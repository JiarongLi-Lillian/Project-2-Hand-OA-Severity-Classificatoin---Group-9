import os
import json
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from train import train
from predict import evaluate, compute_metrics, save_confusion_matrix
from data import get_dataloaders
from baseline_model import get_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def aggregate_metrics(all_metrics: list) -> dict:
    """Compute mean +- std for every metric across runs."""
    scalar_keys = [
        'binary_AUC_disease',
        'omega_MAE',
        'underestimate_rate',
        'overall_accuracy',
    ]
    agg = {}

    for key in scalar_keys:
        vals = [m[key] for m in all_metrics]
        agg[key] = {
            'mean': round(float(np.mean(vals)), 4),
            'std':  round(float(np.std(vals)),  4),
            'all':  [round(v, 4) for v in vals],
        }

    # per_class_recall: KL0-4
    for kl in [f'KL{i}' for i in range(5)]:
        vals = [m['per_class_recall'][kl]
                for m in all_metrics
                if m['per_class_recall'][kl] is not None]
        agg.setdefault('per_class_recall', {})[kl] = {
            'mean': round(float(np.mean(vals)), 4),
            'std':  round(float(np.std(vals)),  4),
            'all':  [round(v, 4) for v in vals],
        }

    return agg


def print_aggregated(agg: dict, exp_name: str):
    print(f"\n{'='*60}")
    print(f"  AGGREGATED RESULTS  ---  {exp_name}")
    print(f"{'='*60}")
    print(f"  [Primary]")
    for key in ['binary_AUC_disease', 'omega_MAE', 'underestimate_rate']:
        v = agg[key]
        print(f"  {key:<25}: {v['mean']:.4f} +- {v['std']:.4f}  {v['all']}")
    print(f"\n  Per-class Recall (KL0-4):")
    for kl, v in agg['per_class_recall'].items():
        print(f"    {kl}: {v['mean']:.4f} +- {v['std']:.4f}  {v['all']}")
    print(f"\n  [Supplementary]")
    v = agg['overall_accuracy']
    print(f"  {'overall_accuracy':<25}: {v['mean']:.4f} +- {v['std']:.4f}  {v['all']}")
    print(f"{'='*60}\n")


def save_aggregated_plot(all_metrics: list, exp_name: str, out_dir: str):
    keys  = ['binary_AUC_disease', 'omega_MAE', 'underestimate_rate', 'overall_accuracy']
    means = [np.mean([m[k] for m in all_metrics]) for k in keys]
    stds  = [np.std( [m[k] for m in all_metrics]) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    x    = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color='steelblue', alpha=0.8, ecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15, ha='right')
    ax.set_title(f'Aggregated Metrics --- {exp_name}')
    ax.set_ylabel('Score')
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'aggregated_metrics.png')
    plt.savefig(path)
    plt.close()
    print(f"Aggregated metrics plot saved to {path}")


def _strip_run_suffix(name: str) -> str:
    """Remove _runN suffix to recover the real model architecture name."""
    import re
    return re.sub(r'_run\d+$', '', name)



def run_experiment(
    model_name      = 'resnet50',
    img_size        = 224,
    batch_size      = 32,
    lr              = 1e-4,
    num_epochs      = 50,
    loss_name       = 'crossentropy',
    optimizer_name  = 'adam',
    patience        = 10,
    no_class_weight = False,
    task_weights    = None,
    lp_epochs       = 0,
    n_runs          = 3,
    base_seed       = 42,
    checkpoint      = 'best',
    use_tta         = False,
):
    # Build experiment name (mirrors train.py run_name convention)
    exp_name = f"{model_name}_{loss_name}"
    if no_class_weight:
        exp_name += "_noweight"
    if task_weights is not None:
        tw_str = "_".join(str(int(w)) for w in task_weights)
        exp_name += f"_tw{tw_str}"
    if lp_epochs > 0:
        exp_name += f"_lp{lp_epochs}"
    if use_tta:
        exp_name += "_tta"

    exp_dir = os.path.join('results', f"{exp_name}_x{n_runs}")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Runs       : {n_runs}  (seeds {base_seed} ... {base_seed + n_runs - 1})")
    print(f"{'#'*60}\n")

    all_metrics  = []
    per_run_info = []
    all_labels_accum = []   # accumulate across runs for aggregated CM
    all_preds_accum  = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for run_idx in range(n_runs):
        seed           = base_seed + run_idx
        # Unique model name per run so checkpoints don't overwrite each other
        # e.g. resnet50_run0, resnet50_run1, resnet50_run2
        run_model_name = f"{model_name}_run{run_idx}"
        set_seed(seed)

        print(f"\n{'--'*30}")
        print(f"  Run {run_idx + 1}/{n_runs}  (seed={seed})  tag={run_model_name}")
        print(f"{'--'*30}")

        import baseline_model as _bm
        import train as _train_mod
        _orig = _bm.get_model

        def _patched(name, num_classes=5, **kw):
            return _orig(_strip_run_suffix(name), num_classes=num_classes, **kw)

        _bm.get_model       = _patched
        _train_mod.get_model = _patched

        try:
            train(
                model_name      = run_model_name,
                img_size        = img_size,
                batch_size      = batch_size,
                lr              = lr,
                num_epochs      = num_epochs,
                loss_name       = loss_name,
                optimizer_name  = optimizer_name,
                patience        = patience,
                no_class_weight = no_class_weight,
                task_weights    = task_weights,
                lp_epochs       = lp_epochs,
            )
        finally:
            _bm.get_model        = _orig
            _train_mod.get_model = _orig

        ckpt_tag = f"{run_model_name}_{loss_name}"
        if no_class_weight:
            ckpt_tag += "_noweight"
        if task_weights is not None:
            tw_str = "_".join(str(int(w)) for w in task_weights)
            ckpt_tag += f"_tw{tw_str}"
        if lp_epochs > 0:
            ckpt_tag += f"_lp{lp_epochs}"
        ckpt_path = f"checkpoints/{ckpt_tag}_{checkpoint}.pth"

        model = get_model(model_name, num_classes=5).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")

        _, _, test_loader = get_dataloaders(img_size=img_size, batch_size=batch_size)

        labels, preds, probs = evaluate(
            model, test_loader, device,
            loss_name       = loss_name,
            no_class_weight = no_class_weight,
            model_name      = model_name,   # real name for CORAL/CORN/multitask checks
            use_tta         = use_tta,
            img_size        = img_size,
        )
        metrics = compute_metrics(labels, preds, probs)
        metrics['run']  = run_idx
        metrics['seed'] = seed

        all_labels_accum.extend(labels.tolist())
        all_preds_accum.extend(preds.tolist())

        all_metrics.append(metrics)
        per_run_info.append({'run': run_idx, 'seed': seed, 'metrics': metrics})

        print(f"  Run {run_idx + 1} results:")
        print(f"    Binary AUC : {metrics['binary_AUC_disease']}")
        print(f"    omega-MAE  : {metrics['omega_MAE']}")
        print(f"    Under Rate : {metrics['underestimate_rate']}")
        print(f"    Accuracy   : {metrics['overall_accuracy']}")
        print(f"    Per-class Recall:")
        for kl, v in metrics['per_class_recall'].items():
            print(f"      {kl}: {v}")

    agg = aggregate_metrics(all_metrics)
    print_aggregated(agg, exp_name)
    save_aggregated_plot(all_metrics, exp_name, exp_dir)

    save_confusion_matrix(
        np.array(all_labels_accum),
        np.array(all_preds_accum),
        f"{exp_name}_x{n_runs}",
        normalize=True,
    )

    results = {
        'experiment': exp_name,
        'n_runs':     n_runs,
        'base_seed':  base_seed,
        'aggregated': agg,
        'per_run':    per_run_info,
    }
    out_path = os.path.join(exp_dir, 'experiment_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Full experiment results saved to {out_path}")

    return agg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ONE model N times and aggregate test metrics.'
    )
    parser.add_argument('--model',           type=str,   default='resnet50')
    parser.add_argument('--img_size',        type=int,   default=224)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--loss',            type=str,   default='crossentropy')
    parser.add_argument('--optimizer',       type=str,   default='adam')
    parser.add_argument('--patience',        type=int,   default=10)
    parser.add_argument('--no_class_weight', action='store_true', default=False)
    parser.add_argument('--task_weights',    type=float, nargs='+', default=None)
    parser.add_argument('--lp_epochs',       type=int,   default=0)
    parser.add_argument('--runs',            type=int,   default=3)
    parser.add_argument('--base_seed',       type=int,   default=42)
    parser.add_argument('--checkpoint',      type=str,   default='best',
                        choices=['best', 'last'])
    parser.add_argument('--tta',             action='store_true', default=False)

    args = parser.parse_args()

    run_experiment(
        model_name      = args.model,
        img_size        = args.img_size,
        batch_size      = args.batch_size,
        lr              = args.lr,
        num_epochs      = args.epochs,
        loss_name       = args.loss,
        optimizer_name  = args.optimizer,
        patience        = args.patience,
        no_class_weight = args.no_class_weight,
        task_weights    = args.task_weights,
        lp_epochs       = args.lp_epochs,
        n_runs          = args.runs,
        base_seed       = args.base_seed,
        checkpoint      = args.checkpoint,
        use_tta         = args.tta,
    )