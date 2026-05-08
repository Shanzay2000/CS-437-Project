# CS-437-Project
Semi-Supervised Object Detection for Rock Art Images with Domain Shift

## 1. Project Overview

This project focuses on automated detection of petroglyphs, also known as rock art engravings, in field images from Pakistan. The main challenge is domain shift: many available labeled petroglyph datasets contain clean, cropped, high-contrast images, while the target images contain full rock surfaces, natural lighting variation, low contrast engravings, shadows, weathering, and complex stone backgrounds.

The goal of this project is to improve object detection performance on Pakistani field petroglyph images, especially the Gichioda target domain, by comparing baseline detectors with semi-supervised learning, domain transfer, and architecture/augmentation-based improvements.

---

## 2. Dataset Description

The project uses four data sources:

| Dataset | Size | Labels | Description |
|---|---:|---|---|
| Roboflow Petroglyph Dataset | 1014 images | Bounding boxes | Internationally sourced cropped petroglyph images with clean, high-contrast compositions |
| KhomarDas Annotated | 202 images | Bounding boxes | Pakistani field petroglyph images with full rock-surface backgrounds |
| KhomarDas Unannotated | 174 images | No labels | Additional Pakistani field images used for semi-supervised learning |
| Gichioda | 123 images | No labels | Raw Pakistani field images used as the main target/test domain |

The Roboflow dataset provides a strong source-domain baseline, while KhomarDas and Gichioda represent the more difficult local Pakistani field-image domain.

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
.
├── baseline_models.ipynb
├── first_improvement_soft_teacher.ipynb
├── second_improvement_cycle_gan.ipynb
├── second_improvement_se_blocks.ipynb
├── README.md
└── results/
