# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project overview

This is a research fork of **FSFM-3C** (CVPR 2025), a self-supervised face security foundation model.

The current course-project target is **DiFF** (Diffusion Facial Forgery Detection). The main problem is that a standard FSFM fine-tuning head is too global: it can miss the sparse, local, and frequency-heavy traces left by diffusion-based forgeries.

Our current downstream improvement direction is **FTF-RVC**:

- **FTF** (Forgery-Aware Token Fusion): mine local anomaly evidence from ViT patch tokens.
- **RVC** (Residual-View Consistency): enforce prediction stability between the original image and a fixed high-pass residual view.

The main idea is to combine spatial token reasoning with frequency-focused calibration so the detector generalizes better across DiFF subsets.

## Key files

- [fsfm-3c/models_vit.py](fsfm-3c/models_vit.py) - contains `VisionTransformer` (baseline) and `VisionTransformerWithTokenFusion` (FTF).
- [fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_finetune_DiFF.py](fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_finetune_DiFF.py) - training entry point for DiFF.
- [fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_test_DiFF.py](fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_test_DiFF.py) - DiFF evaluation entry point.
- [fsfm-3c/finuetune/cross_dataset_unseen_DiFF/engine_finetune.py](fsfm-3c/finuetune/cross_dataset_unseen_DiFF/engine_finetune.py) - training/eval loop, where the residual-view loss should be added.
- [Plan.md](Plan.md) - full research plan, experiment protocol, ablations, and server workflow.

## FTF-RVC architecture

The `VisionTransformerWithTokenFusion` class (registered as `vit_base_patch16_tokenfusion`) augments the ViT-B backbone:

1. ViT backbone -> CLS token `c` + patch tokens `p_1...p_N`
2. Patch anomaly score: `a_i = 1 - cos(p_i, c)`
3. Top-k anomalous patches selected; soft-pooled with `softmax(a_i / tau)` -> local descriptor `l`
4. Global descriptor `g` = `fc_norm(mean(patches))` if `global_pool=True`, else `norm(c)`
5. Fusion gate: `alpha = sigmoid(MLP([g; l]))`, fused: `f = alpha*g + (1-alpha)*l`
6. Final classifier: `head(f)` -> standard logits

RVC is a training-time add-on:

1. Build a high-pass residual view `x_r` from the same input image
2. Run the same model on `x_r`
3. Add a symmetric KL consistency loss between the original-view logits and residual-view logits

Key hyper-parameters:

- `--topk_patches` (default 16)
- `--fusion_tau` (default 0.5)
- `--fusion_gate_hidden_dim` (default 256)
- `--residual_consistency`
- `--rcr_lambda`
- `--rcr_temp`
- `--rcr_filter`

## Running experiments

All commands run from `fsfm-3c/finuetune/cross_dataset_unseen_DiFF/`.

### Data preparation

DiFF needs the raw layout described in the README:

- `datasets/data/DiFF/val`
- `datasets/data/DiFF/test`
- `datasets/data/DiFF/DiFF_real/val`
- `datasets/data/DiFF/DiFF_real/test`

Then run:

```bash
cd /path/to/repo/datasets/finetune/preprocess
python dataset_preprocess.py --dataset FF++_each
python dataset_preprocess.py --dataset DiFF
```

The resulting fine-tuning folders should be:

- training: `datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_each_cls/c23/DeepFakes`
- validation: `datasets/finetune_datasets/diffusion_facial_forgery_detection/DiFF/val_subsets`
- testing: `datasets/finetune_datasets/diffusion_facial_forgery_detection/DiFF/test_subsets`

### Baseline fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune_DiFF.py \
  --model vit_base_patch16 \
  --normalize_from_IMN --apply_simple_augment \
  --batch_size 256 --epochs 50 --blr 5e-4 --layer_decay 0.65 \
  --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --dist_eval \
  --finetune ../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth \
  --data_path ../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_each_cls/c23/DeepFakes \
  --val_data_path ../../../datasets/finetune_datasets/diffusion_facial_forgery_detection/DiFF/val_subsets \
  --output_dir ./checkpoint/DIFF_FSFM_BASELINE
```

### FTF-RVC fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune_DiFF.py \
  --token_fusion \
  --model vit_base_patch16_tokenfusion \
  --normalize_from_IMN --apply_simple_augment \
  --batch_size 256 --epochs 50 --blr 5e-4 --layer_decay 0.65 \
  --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --residual_consistency --rcr_lambda 0.2 --rcr_temp 1.0 --rcr_filter laplacian \
  --dist_eval \
  --finetune ../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth \
  --data_path ../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_each_cls/c23/DeepFakes \
  --val_data_path ../../../datasets/finetune_datasets/diffusion_facial_forgery_detection/DiFF/val_subsets \
  --output_dir ./checkpoint/DIFF_FTF_RVC
```

### Testing

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_test_DiFF.py \
  --eval \
  --model vit_base_patch16_tokenfusion \
  --token_fusion \
  --normalize_from_IMN --apply_simple_augment \
  --nb_classes 2 --batch_size 320 \
  --resume ./checkpoint/DIFF_FTF_RVC \
  --output_dir ./checkpoint/DIFF_FTF_RVC_TEST
```

Evaluation reports subset-wise results on T2I, I2I, FS, and FE.

## Model selection logic

- If `--token_fusion` is set but `--model` is not `vit_base_patch16_tokenfusion`, the script should auto-switch with a warning.
- If `--model vit_base_patch16_tokenfusion` is used without `--token_fusion`, the script should raise `ValueError`.
- FTF-only weight keys (`local_proj`, `fusion_gate`) should be excluded from the missing-key assertion when loading a pretrained backbone checkpoint.
- The residual branch is training-only; evaluation should remain a single forward pass unless a test-time ablation is explicitly requested.

## Research context and open questions

See [Plan.md](Plan.md) for the full ablation plan:

- baseline FSFM vs FTF vs RVC vs FTF-RVC
- sweep `topk_patches` in `{8, 16, 32}`
- sweep `rcr_lambda` in `{0.1, 0.2, 0.4}`
- report frame-level AUC and EER on each DiFF subset

## Git / server workflow

- All development happens on `main`.
- Never commit `datasets/`, checkpoints, logs, or tensorboard files.
- On the server, always `git pull --rebase origin main` before launching experiments.
- Record the git commit hash in every experiment log alongside the command line and checkpoint path.
