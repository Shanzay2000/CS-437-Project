# CS-437-Project
Semi-Supervised Object Detection for Rock Art Images with Domain Shift

## 1. Project Overview

This project focuses on automated detection of petroglyphs (rock art engravings) in field images from Pakistan. The main challenge is domain shift as many available labeled petroglyph datasets contain clean, cropped, high-contrast images, while the target images contain full rock surfaces, natural lighting variation, low contrast engravings, shadows, weathering, and complex stone backgrounds. The goal of this project is to improve object detection performance on Pakistani field petroglyph images, especially the Gichioda target domain, by comparing baseline detectors with semi-supervised learning, domain transfer, and architecture/augmentation-based improvements.

---

## 2. Dataset Description

The project uses four data sources:

| Dataset | Size | Labels | Description |
|---|---:|---|---|
| Roboflow Petroglyph Dataset | 1014 images | Bounding boxes | Internationally sourced cropped petroglyph images with clean, high-contrast compositions |
| KhomarDas Annotated | 202 images | Bounding boxes | Pakistani field petroglyph images with full rock-surface backgrounds |
| KhomarDas Unannotated | 174 images | No labels | Additional Pakistani field images used for semi-supervised learning |
| GichioDas | 123 images | No labels | Raw Pakistani field images used as the main target/test domain |

The Roboflow dataset provides a strong source-domain baseline, while KhomarDas and GichioDas represent the more difficult local Pakistani field-image domain.

---

## 3. Problem Motivation

Petroglyph detection in real field images is difficult because:

- Engravings are often faint and weathered.
- Rock surfaces contain natural texture that can look similar to motifs.
- Lighting varies due to outdoor conditions and oblique sunlight.
- Target-domain Pakistani images have fewer annotations.
- Models trained only on clean cropped datasets may fail on full field photographs.

This project explores methods for reducing this domain gap.

---

## 4. Repository Structure

```text

├── Baseline Models.ipynb
├── First Improvement - Soft Teacher.ipynb
├── Second Improvement - Cycle GAN.ipynb
├── Second Improvement - Squeeze & Excitation.ipynb
└── README.md
```
---

## Notebooks

### 1. `Baseline Models.ipynb`
Three supervised baselines trained exclusively on the Roboflow labeled dataset with no exposure to KhomarDas or GichioDas images during training.

**Models:**
- **YOLOv8n** — trained for 20 epochs, no augmentation, standard YOLO pipeline
- **YOLOv11n** — trained for 20 epochs, no augmentation, standard YOLO pipeline
- **DeFRCN** — Faster R-CNN with ResNet-50 FPN, two-stage fine-tuning:
  - Stage 1: backbone frozen, detection heads only, 300 iterations
  - Stage 2: full network end-to-end, 3000 iterations, cosine LR with warmup
  - Prototype Calibration Block (PCB) at inference: rescales detection scores by cosine similarity between each image's backbone feature and a mean feature vector of cropped petroglyph support images, suppressing false positives on non-petroglyph rock surfaces

**Evaluation:** All models evaluated on the KhomarDas Annotated test set using mAP@50, mAP@50-95, Precision, Recall, and F1.

---

### 2. `First Improvement - Soft Teacher.ipynb`
Semi-supervised object detection using the Soft Teacher framework, exploiting both labeled and unlabeled images.

**Architecture:** ResNet-50 + FPN (P2–P5) + RPN + RoI Head, implemented from scratch in PyTorch.
![Architecture picture](https://github.com/Shanzay2000/CS-437-Project/blob/main/architecture.jpeg)

**Data split:**
- Labeled: Roboflow + KhomarDas Annotated → supervised loss on ground-truth boxes
- Unlabeled: KhomarDas Unannotated + GichioDas → unsupervised loss on teacher pseudo-labels

**How it works:**
- The **teacher** is an identical copy of the student whose weights are never directly trained. Instead they track the student via Exponential Moving Average (EMA, decay=0.9996), making the teacher more stable and reliable than the student at any given step.
- Unlabeled images are passed through **weak augmentation** (horizontal flip) to the teacher, which generates pseudo-labels by filtering detections above a confidence threshold of 0.35.
- The same unlabeled images are passed through **strong augmentation** (colour jitter, grayscale, Gaussian blur, flip) to the student, which computes unsupervised loss against the teacher's pseudo-labels.
- Labeled images pass through strong augmentation to the student for standard supervised loss.

**Objective:**

$$L = L_{sup} + \lambda L_{unsup}$$


**Training schedule:**
- Round 1: 25 epochs, LR=5e-5, from ImageNet pretrained weights
- Round 2: 15 epochs, LR=2.5e-5, from Round 1 best checkpoint

The two-round strategy exists because pseudo-label quality is poor when the teacher is cold. Starting Round 2 from a strong checkpoint means every pseudo-label is already reliable, making the second round a high-quality refinement pass.

---

### 3. `Second Improvement - CycleGAN.ipynb`
Unpaired image-to-image domain transfer to reduce the visual gap between Roboflow training images and the Pakistani field photography of the target domain.

**How it works:**
- A CycleGAN is trained for 50 epochs using 974 Roboflow images as domain A and 202 KhomarDas Annotated images as domain B.
- No paired correspondence is required — the network learns the style difference between the two domains unsupervised.
- The trained generator translates all Roboflow labeled images into KhomarDas visual style (texture, lighting, weathering), while the petroglyph layout and ground-truth bounding boxes remain unchanged.
- Translated images are appended to the labeled training set, doubling annotated diversity at zero annotation cost.

---

### 4. `Second Improvement - Squeeze & Excitation.ipynb`
Two complementary modifications to the Soft Teacher detection backbone targeting the specific visual challenges of petroglyph imagery.

**Modification 1 — Rock-Art Specific Augmentation:**

Added on top of the standard augmentation pipeline, five stochastic degradations simulate real field photography conditions:

| Operation | What it simulates |
|---|---|
| Random linear illumination gradient | Oblique sunlight casting directional shadows across rock faces |
| JPEG re-compression (quality 40–85) | Lossy compression artefacts from field cameras |
| Contrast reduction toward image mean | Weathered, low-contrast engravings blending into stone |
| Gaussian noise (σ = 3–12) | Sensor noise in shadowed or low-light regions |
| Per-channel brightness shift (±18) | Outdoor white-balance colour casts between capture sessions |

**Modification 2 — Squeeze-and-Excitation (SE) Blocks:**

SE blocks (Hu et al., CVPR 2018) are inserted at each residual stage of the ResNet-50 backbone. Each block:
1. Globally average-pools the feature map to a channel descriptor vector
2. Passes it through a small two-layer bottleneck (squeeze → excite)
3. Uses the output to gate each channel of the original feature map

For petroglyphs, engravings and stone background activate very different feature channels. SE blocks learn to amplify motif-relevant channels and suppress uniform stone texture channels, improving the signal-to-noise ratio of features entering the FPN and RPN — particularly beneficial for low-contrast petroglyphs that are visually similar to their background.

---

## Results

| Model | mAP@50 | Precision | Recall |
|---|---|---|---|
| YOLOv8n | 0.5743 | 0.6963 | 0.5374 |
| YOLOv11n | 0.5529 | 0.6932 | 0.4725 |
| DeFRCN | — | — | — |
| Soft Teacher | — | — | — |
| + CycleGAN | — | — | — |
| + SE Blocks | — | — | — |


---
