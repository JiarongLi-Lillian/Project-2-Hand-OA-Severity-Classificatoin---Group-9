import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import torchvision.transforms.functional as TF
from data import get_dataloaders
from baseline_model import get_model, coral_predict, coral_probs, corn_predict, corn_probs
from loss import get_loss, omega_ord_map_predict, build_asymmetric_loss_matrix


def _tta_on_tensor(model, imgs, is_coral, is_corn, already_probs, use_omega, W):
    def _forward(x):
        out = model(x)
        if is_coral:
            return coral_probs(out, num_classes=5)
        elif is_corn:
            return corn_probs(out, num_classes=5)
        elif already_probs:
            return out
        else:
            return torch.softmax(out, dim=1)

    passes = []
    passes.append(_forward(imgs))
    passes.append(_forward(torch.flip(imgs, dims=[3])))
    passes.append(_forward(TF.rotate(imgs, angle=10)))
    passes.append(_forward(TF.rotate(imgs, angle=-10)))
    passes.append(_forward(TF.adjust_brightness(imgs, brightness_factor=1.2)))

    return torch.stack(passes, dim=0).mean(dim=0)


def evaluate(model, loader, device, loss_name, no_class_weight,
             model_name='', use_tta=False, img_size=224):
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    is_multitask = model_name in [
        'mobilenet_v2_MSFM_multitask',
        'mobilenet_v2_MSFM_multitask_jsn_op',
        'mobilenet_v2_MSFM_multitask_op',
        'mobilenet_v2_MSFM_multitask_jsn',
        'mobilenet_v2_MSFM_multitask_crossattn'
    ]
    is_coral      = 'CORAL' in model_name
    is_corn       = 'CORN'  in model_name
    already_probs = 'ORM' in model_name or model_name == 'mobilenet_v2_MSFM_ordinal'
    use_omega     = (loss_name == 'ordinal_asymmetric')

    if use_omega and not is_coral:
        W = build_asymmetric_loss_matrix(num_classes=5).to(device)
    else:
        W = None

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            gt   = labels['kl'].to(device)

            if use_tta:
                probs = _tta_on_tensor(
                    model, imgs,
                    is_coral, is_corn, already_probs, use_omega, W
                )
            else:
                outputs = model(imgs)

                if is_multitask:
                    probs = torch.softmax(outputs, dim=1)
                elif is_coral:
                    probs = coral_probs(outputs, num_classes=5)
                elif is_corn:
                    probs = corn_probs(outputs, num_classes=5)
                elif already_probs:
                    probs = outputs
                else:
                    probs = torch.softmax(outputs, dim=1)

            if use_omega and W is not None:
                preds = omega_ord_map_predict(probs, W)
            else:
                preds = probs.argmax(dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(gt.cpu().numpy())

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def compute_omega_mae(labels, preds, alpha_L=2.0, alpha_R=0.5):
    W = build_asymmetric_loss_matrix(
        num_classes=5, alpha_L=alpha_L, alpha_R=alpha_R
    ).numpy()
    costs = [W[t, p] for t, p in zip(labels, preds)]
    return float(np.mean(costs))


def compute_metrics(labels, preds, probs):
    diff = preds - labels

    overall_acc        = float((preds == labels).mean())
    omega_mae          = compute_omega_mae(labels, preds)
    underestimate_rate = float((diff < 0).mean())

    per_class_recall = {}
    for kl in range(5):
        mask = labels == kl
        if mask.sum() > 0:
            per_class_recall[f'KL{kl}'] = float((preds[mask] == kl).mean())
        else:
            per_class_recall[f'KL{kl}'] = None

    binary_labels       = (labels >= 3).astype(int)
    disease_score       = probs[:, 3] + probs[:, 4]
    fpr_bin, tpr_bin, _ = roc_curve(binary_labels, disease_score)
    binary_auc          = float(auc(fpr_bin, tpr_bin))

    return {
        'binary_AUC_disease': round(binary_auc, 4),
        'omega_MAE':          round(omega_mae, 4),
        'underestimate_rate': round(underestimate_rate, 4),
        'per_class_recall':   {k: round(v, 4) if v is not None else None
                               for k, v in per_class_recall.items()},
        'overall_accuracy':   round(overall_acc, 4),
    }


def save_confusion_matrix(labels, preds, run_name, normalize=False):
    """
    normalize=False : raw counts  (fmt='d')
    normalize=True  : row-normalized recall ratios  (fmt='.2f')
    """
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt     = '.2f'
        title   = f'Confusion Matrix (normalized) - {run_name}'
    else:
        cm_plot = cm
        fmt     = 'd'
        title   = f'Confusion Matrix - {run_name}'

    disp = ConfusionMatrixDisplay(cm_plot, display_labels=[f'KL{i}' for i in range(5)])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format=fmt)
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(f'results/{run_name}', exist_ok=True)
    plt.savefig(f'results/{run_name}/confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved to results/{run_name}/confusion_matrix.png")


def save_roc_curve(labels, probs, run_name):
    binary_labels = (labels >= 3).astype(int)
    disease_score = probs[:, 3] + probs[:, 4]
    fpr, tpr, _   = roc_curve(binary_labels, disease_score)
    binary_auc    = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='crimson',
            label=f'Disease (KL>=3) vs Non-disease (KL<3)  AUC={binary_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'Binary ROC - {run_name}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.tight_layout()
    os.makedirs(f'results/{run_name}', exist_ok=True)
    plt.savefig(f'results/{run_name}/roc_curve.png')
    plt.close()
    print(f"ROC curve saved to results/{run_name}/roc_curve.png")


def print_metrics(metrics, run_name):
    print(f"\n{'='*50}")
    print(f"  {run_name}")
    print(f"{'='*50}")
    print(f"  [Primary]")
    print(f"  Binary AUC (KL>=3) : {metrics['binary_AUC_disease']}")
    print(f"  omega-MAE          : {metrics['omega_MAE']}")
    print(f"  Underestimate Rate : {metrics['underestimate_rate']}")
    print(f"\n  Per-class Recall (KL0-4):")
    for kl, v in metrics['per_class_recall'].items():
        print(f"    {kl}: {v}")
    print(f"\n  [Supplementary]")
    print(f"  Overall Accuracy   : {metrics['overall_accuracy']}")
    print(f"{'='*50}\n")


def predict(model_name='resnet50', img_size=224, batch_size=32,
            checkpoint='best', loss_name='crossentropy',
            no_class_weight=False, use_tta=False,
            task_weights=None, lp_epochs=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if use_tta:
        print("TTA enabled: 5 passes (identity, h-flip, rot+10, rot-10, brightness+20%)")

    _, _, test_loader = get_dataloaders(img_size=img_size, batch_size=batch_size)

    model = get_model(model_name, num_classes=5).to(device)

    ckpt_name = f"{model_name}_{loss_name}"
    if no_class_weight:
        ckpt_name += "_noweight"
    if task_weights is not None:
        tw_str = "_".join(str(int(w)) for w in task_weights)
        ckpt_name += f"_tw{tw_str}"
    if lp_epochs > 0:
        ckpt_name += f"_lp{lp_epochs}"

    ckpt_path = f'checkpoints/{ckpt_name}_{checkpoint}.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")

    run_name = ckpt_name
    if use_tta:
        run_name += "_tta"
    os.makedirs(f'results/{run_name}', exist_ok=True)

    labels, preds, probs = evaluate(
        model, test_loader, device,
        loss_name, no_class_weight,
        model_name=model_name,
        use_tta=use_tta,
        img_size=img_size,
    )

    metrics = compute_metrics(labels, preds, probs)
    metrics['run_name']   = run_name
    metrics['checkpoint'] = checkpoint
    metrics['use_tta']    = use_tta

    save_confusion_matrix(labels, preds, run_name, normalize=True)
    save_roc_curve(labels, probs, run_name)
    print_metrics(metrics, run_name)

    with open(f'results/{run_name}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to results/{run_name}/metrics.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',           type=str,  default='resnet50')
    parser.add_argument('--img_size',        type=int,  default=224)
    parser.add_argument('--batch_size',      type=int,  default=32)
    parser.add_argument('--checkpoint',      type=str,  default='best',
                        choices=['best', 'last'])
    parser.add_argument('--loss',            type=str,  default='crossentropy')
    parser.add_argument('--no_class_weight', action='store_true', default=False)
    parser.add_argument('--tta',             action='store_true', default=False,
                        help='Enable TTA: identity + h-flip + rot+10 + rot-10 + brightness')
    parser.add_argument('--task_weights',    type=float, nargs='+', default=None)
    parser.add_argument('--lp_epochs',       type=int,   default=0)
    args = parser.parse_args()

    predict(
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        loss_name=args.loss,
        no_class_weight=args.no_class_weight,
        use_tta=args.tta,
        task_weights=args.task_weights,
        lp_epochs=args.lp_epochs,
    )