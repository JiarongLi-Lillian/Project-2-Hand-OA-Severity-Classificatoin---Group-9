import os
import re
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import get_dataloaders
from baseline_model import get_model, coral_predict, coral_probs, corn_predict, MultitaskLoss, MultitaskLossJSNOP, MultitaskLossOP, MultitaskLossJSN
from loss import get_loss, omega_ord_map_predict
from optimizer import get_optimizer, get_scheduler


def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_mixup=False, mixup_alpha=0.4, is_multitask=False):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        if is_multitask:
            imgs = imgs.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            optimizer.zero_grad()
            outputs = model(imgs)
            loss, _ = criterion(outputs, labels)
        else:
            imgs, labels = imgs.to(device), labels['kl'].to(device)
            optimizer.zero_grad()

            if use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(imgs.size(0)).to(device)
                mixed_imgs = lam * imgs + (1 - lam) * imgs[idx]
                outputs = model(mixed_imgs)
                loss = (lam * criterion(outputs, labels)
                        + (1 - lam) * criterion(outputs, labels[idx]))
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device, model_name='', is_multitask=False):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    _base_model_name_v = re.sub(r'_run\d+$', '', model_name)
    is_coral      = 'CORAL' in _base_model_name_v
    is_corn       = 'CORN'  in _base_model_name_v
    use_omega     = hasattr(criterion, 'W')
    already_probs = getattr(criterion, 'already_probs', False)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)

            if is_multitask:
                labels = {k: v.to(device) for k, v in labels.items()}
                outputs = model(imgs)
                loss = criterion.asym(outputs, labels['kl'])
                preds = outputs.argmax(dim=1)
                gt    = labels['kl']
            else:
                labels = labels['kl'].to(device)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

                if is_coral:
                    preds = coral_predict(outputs)
                elif is_corn:
                    preds = corn_predict(outputs)
                elif already_probs:
                    probs = outputs
                    if use_omega:
                        preds = omega_ord_map_predict(probs, criterion.W)
                    else:
                        preds = probs.argmax(dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)
                    if use_omega:
                        preds = omega_ord_map_predict(probs, criterion.W)
                    else:
                        preds = outputs.argmax(dim=1)
                gt = labels

            total_loss += loss.item()
            correct    += (preds == gt).sum().item()
            total      += gt.size(0)

    return total_loss / len(loader), correct / total


def save_learning_curve(train_losses, val_losses, val_accs, run_name):
    os.makedirs(f'results/{run_name}', exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses,   label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, val_accs, label='Val Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'results/{run_name}/learning_curve.png')
    plt.close()
    print(f"Learning curve saved to results/{run_name}/learning_curve.png")


def train(model_name='resnet50', img_size=224, batch_size=32,
          lr=1e-4, num_epochs=50, loss_name='crossentropy',
          optimizer_name='adam', patience=10, no_class_weight=False,
          task_weights=None, lp_epochs=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Strip _runN suffix for all model-name-based logic
    _base_model_name = re.sub(r'_run\d+$', '', model_name)

    is_multitask = _base_model_name in [
        'mobilenet_v2_MSFM_multitask',
        'mobilenet_v2_MSFM_multitask_jsn_op',
        'mobilenet_v2_MSFM_multitask_op',
        'mobilenet_v2_MSFM_multitask_jsn',
        'mobilenet_v2_MSFM_multitask_crossattn'
    ]

    train_loader, val_loader, _ = get_dataloaders(
        img_size=img_size, batch_size=batch_size
    )

    model = get_model(model_name, num_classes=5).to(device)

    run_name = f"{model_name}_{loss_name}"
    if no_class_weight:
        run_name += "_noweight"
    if task_weights is not None:
        tw_str = "_".join(str(int(w)) for w in task_weights)
        run_name += f"_tw{tw_str}"
    if lp_epochs > 0:
        run_name += f"_lp{lp_epochs}"
    print(f"Run name: {run_name}")

    train_labels    = train_loader.dataset.df['KL'].tolist()
    labels_for_loss = None if no_class_weight else train_labels

    already_probs = 'ORM' in _base_model_name or _base_model_name == 'mobilenet_v2_MSFM_ordinal'

    if is_multitask:
        asym_loss_fn = get_loss(
            loss_name,
            labels=labels_for_loss,
            device=device,
            already_probs=False,
            task_weights=task_weights,
        )
        # Use _base_model_name for criterion selection
        if _base_model_name == 'mobilenet_v2_MSFM_multitask_jsn_op':
            criterion = MultitaskLossJSNOP(
                asym_loss_fn=asym_loss_fn,
                w_jsn=0.49, w_op=0.51,
            ).to(device)
        elif _base_model_name == 'mobilenet_v2_MSFM_multitask_op':
            criterion = MultitaskLossOP(
                asym_loss_fn=asym_loss_fn,
                w_op=1.0,
            ).to(device)
        elif _base_model_name == 'mobilenet_v2_MSFM_multitask_jsn':
            criterion = MultitaskLossJSN(
                asym_loss_fn=asym_loss_fn,
                w_jsn=1.0,
            ).to(device)
        else:
            criterion = MultitaskLoss(
                asym_loss_fn=asym_loss_fn,
                w_jsn=0.40, w_op=0.41, w_er=0.19,
            ).to(device)
    else:
        criterion = get_loss(
            loss_name,
            labels=labels_for_loss,
            device=device,
            already_probs=already_probs,
            task_weights=task_weights,
        )

    if lp_epochs > 0 and 'CORN' in _base_model_name:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        optimizer = get_optimizer(optimizer_name, model, lr=lr * 10)
        scheduler = get_scheduler('plateau', optimizer)
        print(f"Phase 1: Linear Probing for {lp_epochs} epochs "
              f"(classifier only, lr={lr * 10})...")
    else:
        orm_lr    = lr * 100 if _base_model_name.endswith('_ORM') else None
        optimizer = get_optimizer(optimizer_name, model, lr=lr, orm_lr=orm_lr)
        scheduler = get_scheduler('plateau', optimizer)

    train_losses, val_losses, val_accs = [], [], []
    best_val_loss      = float('inf')
    early_stop_counter = 0

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(f'results/{run_name}', exist_ok=True)

    print(f"\nStart training {run_name} for {num_epochs} epochs...\n")

    for epoch in range(1, num_epochs + 1):

        if lp_epochs > 0 and 'CORN' in _base_model_name and epoch == lp_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = get_optimizer(optimizer_name, model, lr=lr)
            scheduler = get_scheduler('plateau', optimizer)
            print(f"Phase 2: Full Fine-tuning from epoch {epoch}, lr={lr}...")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            is_multitask=is_multitask
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            model_name=model_name, is_multitask=is_multitask
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       f'checkpoints/{run_name}_best.pth')
            print(f"  -> Best model saved (val_loss: {best_val_loss:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    torch.save(model.state_dict(), f'checkpoints/{run_name}_last.pth')
    print(f"Last model saved to checkpoints/{run_name}_last.pth")

    save_learning_curve(train_losses, val_losses, val_accs, run_name)

    metrics = {
        'run_name':        run_name,
        'model':           model_name,
        'loss':            loss_name,
        'no_class_weight': no_class_weight,
        'already_probs':   already_probs,
        'best_val_loss':   best_val_loss,
        'best_val_acc':    max(val_accs),
        'total_epochs':    len(train_losses),
        'lp_epochs':       lp_epochs,
        'is_multitask':    is_multitask,
    }
    with open(f'results/{run_name}/train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Training metrics saved to results/{run_name}/train_metrics.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',           type=str,   default='resnet50')
    parser.add_argument('--img_size',        type=int,   default=224)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--loss',            type=str,   default='crossentropy')
    parser.add_argument('--optimizer',       type=str,   default='adam')
    parser.add_argument('--patience',        type=int,   default=10)
    parser.add_argument('--no_class_weight', action='store_true', default=False)
    parser.add_argument('--task_weights',    type=float, nargs='+', default=None,
                        help='Per-task weights for CORAL loss')
    parser.add_argument('--lp_epochs',       type=int,   default=0,
                        help='Number of linear probing epochs before full fine-tuning')
    args = parser.parse_args()

    train(
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        loss_name=args.loss,
        optimizer_name=args.optimizer,
        patience=args.patience,
        no_class_weight=args.no_class_weight,
        task_weights=args.task_weights,
        lp_epochs=args.lp_epochs,
    )