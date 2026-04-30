# Project-2-Hand-OA-Severity-Classificatoin---Group-9

## 1. File Description

### Core scripts

| File | Description |
|-----|------------|
| `run_experiment.py` | Main entry point. Runs full experiments (train + evaluate + aggregate) |
| `train.py` | Training loop (model training, validation, checkpointing) |
| `predict.py` | Evaluation and metrics (AUC, ω-MAE, confusion matrix) |
| `baseline_model.py` | Model architectures (ResNet, MobileNetV2, MSFM, ORM/CORAL/CORN) |
| `loss.py` | Loss functions (CrossEntropy, Asymmetric, CORAL, CORN) |
| `data.py` | Dataset loading, preprocessing, augmentation |
| `optimizer.py` | Optimizer and learning rate scheduler |

---

### Dataset-related

| File / Folder | Description |
|--------------|------------|
| `train_multitask.csv` | Training split |
| `val_multitask.csv` | Validation split |
| `test_multitask.csv` | Test split |
| `finger_joints_img_*` | Image folders |

---

## 2. Command Line Usage (Experiment Guide)

All experiments are executed using:

```bash
python run_experiment.py [arguments]
```
### Experiment 1 — Loss Ablation
```bash
# CE — no class weight
python run_experiment.py --model resnet50 --loss crossentropy --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# CE — with class weight
python run_experiment.py --model resnet50 --loss crossentropy --epochs 100 --lr 1e-5 --runs 3

# Asymmetric loss (recommended)
python run_experiment.py --model resnet50 --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# Asymmetric + weight
python run_experiment.py --model resnet50 --loss ordinal_asymmetric --epochs 100 --lr 1e-5 --runs 3
```
### Experiment 2 — Backbone Comparison
```bash
# ResNet101
python run_experiment.py --model resnet101 --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# DenseNet201
python run_experiment.py --model densenet201 --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# MobileNetV2
python run_experiment.py --model mobilenet_v2 --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3
```

### Experiment 3 — Ordinal Methods
```bash
# ORM
python run_experiment.py --model mobilenet_v2_ORM --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# CORAL
python run_experiment.py --model mobilenet_v2_CORAL --loss coral --epochs 100 --lr 1e-5 --runs 3

# CORN
python run_experiment.py --model mobilenet_v2_CORN --loss corn --lp_epochs 5 --epochs 100 --lr 1e-5 --runs 3
```

### Experiment 4 — MSFM
```bash
# MSFM + CE
python run_experiment.py --model mobilenet_v2_MSFM --loss crossentropy --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# MSFM + Asymmetric (final model)
python run_experiment.py --model mobilenet_v2_MSFM --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# MSFM + CORN
python run_experiment.py --model mobilenet_v2_MSFM_CORN --loss corn --epochs 100 --lr 1e-5 --runs 3

# MSFM + Ordinal head
python run_experiment.py --model mobilenet_v2_MSFM_ordinal --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3
```

### Experiment 5 — Feature Fusion / Multitask
```bash
# JSN only
python run_experiment.py --model mobilenet_v2_MSFM_multitask_jsn --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# OP only
python run_experiment.py --model mobilenet_v2_MSFM_multitask_op --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# JSN + OP
python run_experiment.py --model mobilenet_v2_MSFM_multitask_jsn_op --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# Full multitask
python run_experiment.py --model mobilenet_v2_MSFM_multitask --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3

# Cross-attention
python run_experiment.py --model mobilenet_v2_MSFM_multitask_crossattn --loss ordinal_asymmetric --no_class_weight --epochs 100 --lr 1e-5 --runs 3
```


