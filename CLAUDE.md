# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a research fork of **FSFM-3C** (CVPR 2025), a self-supervised face security foundation model. The fork adds **Forgery-Aware Token Fusion (FTF)**, a lightweight plug-in head that improves diffusion-face forgery detection on the **DiFF** benchmark.

The core problem: the original FSFM fine-tunes a single CLS-token classifier on DiFF, which misses sparse, local forgery artifacts. FTF addresses this without changing the pretraining pipeline.

## Key files

- [fsfm-3c/models_vit.py](fsfm-3c/models_vit.py) — contains both `VisionTransformer` (baseline) and `VisionTransformerWithTokenFusion` (FTF). The FTF class is the primary research contribution.
- [fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_finetune_DiFF.py](fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_finetune_DiFF.py) — training entry point for DiFF evaluation.
- [fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_test_DiFF.py](fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_test_DiFF.py) — evaluation entry point.
- [fsfm-3c/finuetune/cross_dataset_unseen_DiFF/engine_finetune.py](fsfm-3c/finuetune/cross_dataset_unseen_DiFF/engine_finetune.py) — training/eval loop (unchanged from baseline).
- [FSFM_DiFF_FTF_Plan.md](FSFM_DiFF_FTF_Plan.md) — full research plan: motivation, math, experiment protocol, ablations, and server workflow.

## FTF architecture

The `VisionTransformerWithTokenFusion` class (registered as `vit_base_patch16_tokenfusion`) augments the ViT-B backbone:

1. ViT backbone → CLS token `c` + patch tokens `p_1…p_N`
2. Patch anomaly score: `a_i = 1 − cos(p_i, c)`
3. Top-k anomalous patches selected; soft-pooled with `softmax(a_i / τ)` → local descriptor `l`
4. Global descriptor `g` = `fc_norm(mean(patches))` if `global_pool=True`, else `norm(c)`
5. Fusion gate: `α = sigmoid(MLP([g; l]))`, fused: `f = α·g + (1−α)·l`
6. Final classifier: `head(f)` — same cross-entropy loss, no extra supervision

Key hyper-parameters: `--topk_patches` (default 16), `--fusion_tau` (default 0.5), `--fusion_gate_hidden_dim` (default 256).

## Running experiments

All commands run from `fsfm-3c/finuetune/cross_dataset_unseen_DiFF/`.

### Baseline fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune_DiFF.py \
  --model vit_base_patch16 \
  --normalize_from_IMN --apply_simple_augment \
  --batch_size 256 --epochs 50 --blr 5e-4 --layer_decay 0.65 \
  --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --finetune ../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth \
  --data_path ../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_each_cls/c23/DeepFakes \
  --val_data_path ../../../datasets/finetune_datasets/diffusion_facial_forgery_detection/DiFF/val_subsets \
  --output_dir ./checkpoint/DIFF_FSFM_BASELINE
```

### FTF fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_finetune_DiFF.py \
  --token_fusion \
  --model vit_base_patch16_tokenfusion \
  --normalize_from_IMN --apply_simple_augment \
  --batch_size 256 --epochs 50 --blr 5e-4 --layer_decay 0.65 \
  --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
  --finetune ../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth \
  --data_path ../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_each_cls/c23/DeepFakes \
  --val_data_path ../../../datasets/finetune_datasets/diffusion_facial_forgery_detection/DiFF/val_subsets \
  --output_dir ./checkpoint/DIFF_FTF
```

### Testing

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_test_DiFF.py \
  --eval \
  --model vit_base_patch16_tokenfusion \
  --token_fusion \
  --normalize_from_IMN --apply_simple_augment \
  --nb_classes 2 --batch_size 320 \
  --resume ./checkpoint/DIFF_FTF \
  --output_dir ./checkpoint/DIFF_FTF_TEST
```

Evaluation reports frame-level AUC/EER for each DiFF subset: T2I, I2I, FS, FE.

### Data preprocessing

Run once if face crops are not yet prepared:

```bash
python datasets/finetune/preprocess/dataset_preprocess.py --dataset FF++_each
python datasets/finetune/preprocess/dataset_preprocess.py --dataset DiFF
```

## Model selection logic in main_finetune_DiFF.py

- If `--token_fusion` is set but `--model` is not `vit_base_patch16_tokenfusion`, the script auto-switches with a warning.
- If `--model vit_base_patch16_tokenfusion` is used without `--token_fusion`, the script raises `ValueError`.
- FTF-only weight keys (`local_proj`, `fusion_gate`) are excluded from the missing-key assertion when loading a pretrained backbone checkpoint — they are initialized fresh and trained from scratch.

## Research context and open questions

See [FSFM_DiFF_FTF_Plan.md](FSFM_DiFF_FTF_Plan.md) §8 for the full ablation plan:
- Ablate the fusion gate (hard top-k mean vs. gated fusion)
- Sweep `topk_patches` ∈ {8, 16, 32}
- Compare against ImageNet ViT-B baseline and original FSFM baseline
- Metrics: frame-level AUC and EER per DiFF subset

## Git / server workflow

- All development happens on `main`. Push to `main` only after verifying locally.
- Never commit `datasets/`, checkpoints, logs, or tensorboard files.
- On the server, always `git pull --rebase origin main` before launching experiments.
- Record the git commit hash in every experiment log alongside the command line and checkpoint path.
