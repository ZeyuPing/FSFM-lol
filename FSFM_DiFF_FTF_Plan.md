# FSFM DiFF Improvement Plan

## 1. Why this direction

The original FSFM paper learns a strong face-level representation from unlabeled real faces, and it already shows good transfer to DiFF. However, the downstream DiFF fine-tuning pipeline still uses a single ViT classification head, so the final decision mostly comes from global semantics. That is a weak point for unseen diffusion forgeries, because the most discriminative evidence is often sparse, local, and inconsistent across subsets such as T2I, I2I, FS, and FE.

The challenge we want to address is:

- A single global CLS-style decision may miss small diffusion artifacts.
- The model has no explicit way to emphasize patch-level anomaly evidence.
- We want an improvement that is different from the original FSFM pretraining idea, but still easy to train and compare fairly.

## 2. Chosen idea: Forgery-Aware Token Fusion (FTF)

The proposed method keeps the pretrained FSFM backbone, but adds a lightweight token-fusion head for downstream DiFF fine-tuning.

Core intuition:

- Use the final-layer class token as a global semantic reference.
- Score each patch token by how far it deviates from the global reference.
- Select the most suspicious patch tokens and aggregate them into a local forgery descriptor.
- Fuse the global descriptor and local forgery descriptor before the final classifier.

This is a plug-and-play change. It does not need extra annotations, face parsing, or a new dataset. It also keeps the training loss as standard cross-entropy, so the existing fine-tuning loop stays almost unchanged.

## 3. Method design

### 3.1 Feature extraction

For an input face image, the ViT backbone produces a token sequence:

- `c` = class token
- `p_1 ... p_N` = patch tokens

We will expose the final tokens from the backbone without changing the pretrained weights.

### 3.2 Patch anomaly scoring

Use the class token as a global reference and compute a patch anomaly score:

```text
a_i = 1 - cos(p_i, c)
```

High scores indicate that a patch is different from the global face semantics and may contain a forgery trace.

### 3.3 Local token pooling

Select the top-`k` patch tokens with the highest anomaly scores.

```text
S = TopK(a_1 ... a_N, k)
l = sum_{i in S} softmax(a_i / tau) * p_i
```

`l` is the local forgery descriptor. `tau` is a temperature for the softmax weights.

### 3.4 Global-local fusion

The standard FSFM fine-tuning branch already gives a global face descriptor `g`.

We fuse the two descriptors with a small gating MLP:

```text
alpha = sigmoid(MLP([g ; l]))
f = alpha * g + (1 - alpha) * l
logits = Head(f)
```

This makes the classifier use both global face consistency and local artifact evidence.

### 3.5 Why this is different from FSFM

FSFM focuses on learning a universal face representation during self-supervised pretraining. Our method is a downstream adaptation that explicitly mines local forgery evidence from the pretrained tokens. In short:

- FSFM: learn general face features from real faces.
- FTF: use those features to find suspicious local tokens for unseen diffusion forgeries.

That makes the contribution clearly different and suitable as a course project extension.

## 4. Files to change

Recommended minimal code scope:

- `fsfm-3c/models_vit.py`
- `fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_finetune_DiFF.py`
- `fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_test_DiFF.py`

Optional, only if needed for cleanup:

- `fsfm-3c/finuetune/cross_dataset_unseen_DiFF/engine_finetune.py`

### 4.1 `models_vit.py`

Add a new backbone variant, for example:

- `vit_base_patch16_tokenfusion`

Implementation notes:

- Inherit from the current `VisionTransformer`.
- Expose the final tokens before pooling.
- Add a small local projection head and a fusion gate.
- Keep the output as a single logits tensor so the existing training loop does not need a new loss format.

### 4.2 `main_finetune_DiFF.py`

Add CLI arguments:

- `--token_fusion`
- `--topk_patches`
- `--fusion_tau`
- `--fusion_gate_hidden_dim`

When `--token_fusion` is enabled, instantiate the new model variant.

### 4.3 `main_test_DiFF.py`

Add the same model-selection arguments so evaluation can load the new checkpoint with the same architecture.

### 4.4 No change needed in v1

If the model returns logits only, the current training loop in `engine_finetune.py` can stay unchanged.

## 5. Implementation steps

1. Add a token-returning forward path in the ViT backbone.
2. Build the local anomaly scoring and top-k token pooling.
3. Add the fusion gate and final classifier.
4. Wire the new model into the DiFF fine-tuning and testing scripts.
5. Run baseline and FTF experiments under the same data split and hyperparameters.
6. Save logs and checkpoints in separate experiment folders.

## 6. Data and environment

Use the already available assets:

- Pretrained checkpoint: `fsfm-3c/pretrain/checkpoint/pretrained_models/FS-VFM_ViT-B_VF2_600e`
- DiFF dataset: `datasets/finetune_datasets/diffusion_facial_forgery_detection/DiFF`

If you want strict reproduction of the original README protocol, you can later swap in the FF++_o pretrained checkpoint. For now, the VF2 checkpoint is enough to validate the proposed method.

## 7. Experiment plan

### 7.1 Preprocessing

If the DiFF face crops are not already prepared on the server, run:

```bash
cd /path/to/repo
python datasets/finetune/preprocess/dataset_preprocess.py --dataset FF++_each
python datasets/finetune/preprocess/dataset_preprocess.py --dataset DiFF
```

### 7.2 Baseline fine-tuning

Run the original DiFF fine-tuning script as the main baseline:

```bash
cd /path/to/repo/fsfm-3c/finuetune/cross_dataset_unseen_DiFF
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=1 main_finetune_DiFF.py \
  --accum_iter 1 \
  --normalize_from_IMN \
  --apply_simple_augment \
  --batch_size 256 \
  --nb_classes 2 \
  --model vit_base_patch16 \
  --epochs 50 \
  --blr 5e-4 \
  --layer_decay 0.65 \
  --weight_decay 0.05 \
  --drop_path 0.1 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --dist_eval \
  --finetune ../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth \
  --data_path ../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_each_cls/c23/DeepFakes \
  --val_data_path ../../../datasets/finetune_datasets/diffusion_facial_forgery_detection/DiFF/val_subsets \
  --output_dir ./checkpoint/DIFF_FSFM_BASELINE
```

### 7.3 Proposed method fine-tuning

After the new model is coded, run:

```bash
cd /path/to/repo/fsfm-3c/finuetune/cross_dataset_unseen_DiFF
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=1 main_finetune_DiFF.py \
  --accum_iter 1 \
  --normalize_from_IMN \
  --apply_simple_augment \
  --batch_size 256 \
  --nb_classes 2 \
  --model vit_base_patch16_tokenfusion \
  --token_fusion \
  --topk_patches 16 \
  --fusion_tau 0.5 \
  --fusion_gate_hidden_dim 256 \
  --epochs 50 \
  --blr 5e-4 \
  --layer_decay 0.65 \
  --weight_decay 0.05 \
  --drop_path 0.1 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --dist_eval \
  --finetune ../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth \
  --data_path ../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_each_cls/c23/DeepFakes \
  --val_data_path ../../../datasets/finetune_datasets/diffusion_facial_forgery_detection/DiFF/val_subsets \
  --output_dir ./checkpoint/DIFF_FTF
```

### 7.4 Testing

Use the saved fine-tuned folder for subset-wise testing:

```bash
cd /path/to/repo/fsfm-3c/finuetune/cross_dataset_unseen_DiFF
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 main_test_DiFF.py \
  --normalize_from_IMN \
  --apply_simple_augment \
  --eval \
  --model vit_base_patch16_tokenfusion \
  --nb_classes 2 \
  --batch_size 320 \
  --resume ./checkpoint/DIFF_FTF \
  --output_dir ./checkpoint/DIFF_FTF_TEST
```

The script will automatically evaluate T2I, I2I, FS, and FE.

## 8. Baselines and ablations

Report the following comparisons:

- ImageNet-pretrained ViT-B
- Original FSFM fine-tuning baseline
- FTF without fusion gate
- FTF with different `topk_patches` values
- FTF full model

Metrics to report:

- Frame-level AUC
- Video-level AUC
- Frame-level EER
- Video-level EER if needed

## 9. What to write in the paper

The paper should emphasize:

- FSFM is a strong face foundation model, but its downstream DiFF use is still global-classification heavy.
- Diffusion forgeries are sparse and local, so a token-fusion head is a natural extension.
- The method is lightweight and does not require extra supervision.
- The improvement should be judged under the same protocol as the original paper.

## 10. Git workflow for two local developers plus one server clone

Recommended branch structure:

- `main`: stable baseline
- `feat/ftf-diff`: shared feature branch for code and paper updates

### 10.1 Local developer workflow

Before starting:

```bash
git checkout main
git pull --rebase origin main
git checkout -b feat/ftf-diff
```

When making changes:

```bash
git status
git add fsfm-3c/models_vit.py \
        fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_finetune_DiFF.py \
        fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_test_DiFF.py \
        FSFM_DiFF_FTF_Plan.md \
        FTF_DiFF_paper_draft.tex
git commit -m "Add forgery-aware token fusion for DiFF"
git push -u origin feat/ftf-diff
```

If the second collaborator also edits the same branch:

```bash
git fetch origin
git checkout feat/ftf-diff
git pull --rebase origin feat/ftf-diff
```

### 10.2 Server sync workflow

On the server clone, always pull before launching experiments:

```bash
git fetch origin --prune
git checkout feat/ftf-diff
git pull --rebase origin feat/ftf-diff
git status --short
```

Then run preprocessing or training from the same commit.

### 10.3 Conflict rules

- Never commit `datasets/` contents.
- Never commit checkpoints, logs, or tensorboard files.
- If local code and server code diverge, resolve on the local machine first, then push, then pull on the server again.
- If a checkpoint is tied to a specific commit, record the commit hash in the experiment log.

## 11. Suggested experiment log format

For each run, save:

- Git commit hash
- Command line
- Dataset split
- Pretrained checkpoint path
- Best subset checkpoint paths
- Final T2I / I2I / FS / FE scores

That will make the paper table easy to fill later.
