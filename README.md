# IndiUrbanClean Dataset

A curated image dataset for urban waste classification in Indian cities. We collected and annotated images across five urban waste categories and used them to run diagnostic deep learning experiments — studying how augmentation strategies build invariances, how CNN-Transformer hybrid architectures trade off local and global features, and how different layer freezing strategies affect fine-tuning on a small domain-specific dataset. The focus throughout is on understanding model behaviour, not just maximising accuracy.

---

## Overview

The IndiUrbanClean project is divided into two phases:

- **Phase 1:** Dataset creation, curation, and annotation
- **Phase 2:** Deep learning experiments analyzing model behavior across three diagnostic problems

The goal is not just to achieve high accuracy, but to understand how models learn and respond to different training strategies and architectures.

---

## Dataset

### Classes (5)
| Class | Description |
|---|---|
| `clean_street` | Areas with little or no visible waste, dominated by clean surfaces |
| `open_waste` | Scattered garbage in open areas with high visual variation |
| `construction_waste` | Building site debris with structured shapes, dust, and rubble |
| `dumpyard` | Large organised disposal sites where scene layout is the key signal |
| `overfilled_bins` | Bins at or over capacity where the bin boundary is the key visual cue |

### Statistics
- Train/Validation/Test split: 70/15/15 with stratified sampling to ensure balanced class distribution across all splits.
- Input size: 224×224
- All images manually reviewed for class consistency

### Sources
Images collected from Google Images, Bing, Flickr, Wikimedia Commons, Unsplash, and Pexels.

### Folder Structure
```
IndiUrbanClean/
├── images/
│   ├── clean_street/
│   ├── open_waste/
│   ├── construction_waste/
│   ├── dumpyard/
│   └── overfilled_bins/
├── annotations/
│   └── labels.csv
├── metadata/
│   └── image_metadata.csv
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── docs/
│   ├── dataset_collection.md
│   └── annotation_guidelines.md
├── README.md
├── project1_augmentation
├── coatnet_hybrid_dlcv.ipynb
├── TransferLearning_Part3.ipynb
└── LICENSE
```
---

## Phase 2: Experiments

All experiments use a ResNet-18 pretrained on ImageNet (via `timm`), with identical dataset splits and controlled settings across problems.

## Experiments

### Data Augmentation as Inductive Bias

Five augmentation strategies (baseline, rotation, crop, colour jitter, cutout) were evaluated on clean and four corrupted test sets (blur, dark, rotate, noise) to understand which augmentations build genuine invariances rather than just improving clean accuracy.

| Strategy | Clean | Blur | Dark | Rotate | Noise | Avg (corr.) |
|---|---|---|---|---|---|---|
| None (baseline) | 0.928 | 0.896 | 0.910 | 0.763 | 0.900 | 0.867 |
| Rotation | 0.925 | 0.903 | 0.898 | 0.874 | 0.901 | 0.894 |
| Crop | 0.918 | 0.921 | 0.912 | 0.771 | 0.908 | 0.878 |
| Colour Jitter | 0.910 | 0.889 | 0.924 | 0.724 | 0.893 | 0.858 |
| Cutout | 0.907 | 0.905 | 0.884 | 0.701 | 0.919 | 0.852 |

Each augmentation achieved its highest gain on the corruption it was designed to cover, confirming that augmentation works best when matched to the expected deployment shift.

---

### Hybrid CNN-Transformer Architecture

A CoAtNet-style hybrid model was tested with transition stages 1–4 in both Conv→Attn and Attn→Conv directions, evaluated on local texture tasks (clean test set) and global reasoning tasks (blurred, σ=5).

| Stage | Conv→Attn Local | Conv→Attn Global | Attn→Conv Local | Attn→Conv Global |
|---|---|---|---|---|
| 1 | 0.91 | 0.43 | 0.90 | 0.39 |
| 2 | 0.90 | 0.40 | 0.91 | 0.51 |
| 3 | 0.88 | 0.37 | 0.91 | 0.44 |
| 4 | 0.91 | 0.48 | 0.93 | 0.48 |

Both directions converge at stage 4 for overall accuracy (~93%). Attn→Conv at stage 2 is best for global reasoning. A strong texture bias from the frozen backbone limits global performance regardless of staging.

---

### Transfer Learning — Layer Freezing Strategy

Four freezing strategies were compared on ResNet-18 with 5 epochs and an Adam optimizer, with learning rates tuned per strategy.

| Strategy | Trainable Layers | Train Acc. | Val Acc. | Test Acc. | Overfit Gap |
|---|---|---|---|---|---|
| All But Head | Head only | 0.9061 | 0.9024 | 0.8967 | 0.0037 |
| Freeze Early | Layer3, Layer4, Head | 0.9957 | 0.9495 | 0.9233 | 0.0462 |
| Freeze Late | Layer1, Layer2, Head | 0.9393 | 0.8990 | 0.8567 | 0.0403 |
| Train All | All layers | 0.9949 | 0.9562 | 0.9300 | 0.0387 |

Train All (93.0%) performs best with a low learning rate on a lightweight backbone. Freeze Early (92.3%) is nearly as strong and the safer choice. Freeze Late consistently underperforms by disrupting low-level transferable features.
---

## Key Takeaways

- Models are strongly texture-driven — photometric augmentations outperform geometric ones
- Hybrid staging alone cannot overcome backbone texture bias for global tasks
- Preserving low-level pretrained features and fine-tuning deeper layers is the right balance for small domain-specific datasets
- Recommended setup: colour jitter + random crop, Attn→Conv at stage 2, Freeze Early for fine-tuning

---

## Applications

- Urban cleanliness monitoring
- Waste detection systems
- Smart city analytics
- Environmental awareness tools

---

## License

Refer to the LICENSE file included with the dataset.
