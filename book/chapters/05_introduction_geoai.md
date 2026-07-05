# Chapter 5: Introduction to GeoAI

## 5.1 The Deep Learning Revolution in Earth Observation

Remote sensing analysis was historically dominated by expert-designed feature extraction:
normalised difference indices, object-based image analysis, and spectral mixture models.
Deep learning changed everything by learning hierarchical spatial features directly from raw pixels.

### Key transformative capabilities

**Spatial context:** CNNs capture relationships at multiple scales simultaneously.
A single network sees both rooftop texture and the overall urban grid pattern.

**Transfer learning:** Pre-trained weights from ImageNet or EO datasets reduce
the labelled data needed for new domains by 10-100x.

**End-to-end training:** No separate feature extraction stage. The network learns
representations directly optimised for the final prediction objective.

## 5.2 Unique Challenges of Geospatial AI

### Multi-spectral input

Standard ImageNet models use 3 RGB channels. Sentinel-2 has 13 bands.
Adapting a pre-trained model:

```python
import torch, torch.nn as nn

# Replace first conv: 3ch -> 6ch, copy RGB weights
orig = model.encoder.layer0.conv1
new  = nn.Conv2d(6, orig.out_channels, orig.kernel_size,
                  orig.stride, orig.padding, bias=False)
with torch.no_grad():
    new.weight[:, :3] = orig.weight
    new.weight[:, 3:] = orig.weight.mean(1, keepdim=True).repeat(1,3,1,1)
model.encoder.layer0.conv1 = new
```

### Class imbalance

Damaged buildings may represent 0.1% of pixels. Use FocalDice loss:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, focal_w=0.5, dice_w=0.5):
        super().__init__()
        self.gamma = gamma
        self.fw, self.dw = focal_w, dice_w

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(logits, targets.float(),
                                                     reduction='none')
        pt    = torch.sigmoid(logits)
        focal = ((1-pt)**self.gamma * bce).mean()
        num   = 2*(pt*targets).sum()
        den   = pt.sum() + targets.sum() + 1e-8
        dice  = 1 - num/den
        return self.fw * focal + self.dw * dice
```

### Geographic distribution shift

A model trained on European buildings fails in West Africa.
Always use geographically stratified evaluation splits:

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=grid_cells))
```

## 5.3 The Model Registry

```python
from pygeovision.models import list_models, get_model

print(f'Total: {len(list_models())}')
for task in ['segmentation','detection','change_detection','foundation','sar']:
    print(f'  {task}: {len(list_models(task=task))}')

# Load any model
unet = get_model('unet', encoder_name='resnet50',
                  encoder_weights='imagenet',
                  in_channels=6, num_classes=5)
print(f'Parameters: {sum(p.numel() for p in unet.parameters()):,}')
```

## 5.4 U-Net Family

U-Net's encoder-decoder with skip connections is the workhorse of EO segmentation:

```python
# Available variants
variants = [
    ('unet',         'resnet34',  24.4, 'Classic U-Net'),
    ('unet',         'resnet50',  32.5, 'U-Net ResNet50'),
    ('unetplusplus', 'resnet50',  35.1, 'U-Net++ (nested)'),
    ('deeplabv3plus','resnet101', 59.3, 'DeepLabV3+ (ASPP)'),
]
for arch, enc, params, desc in variants:
    print(f'  {desc}: {params}M parameters')
```

## 5.5 Vision Transformers

SegFormer captures long-range spatial dependencies CNNs miss:

```python
# SegFormer variants (B0-B5, increasing capacity)
for v in ['b0','b1','b2','b3','b4','b5']:
    m = get_model('segformer', variant=v, in_channels=6, num_classes=10)
    n = sum(p.numel() for p in m.parameters())/1e6
    print(f'  SegFormer-{v.upper()}: {n:.1f}M')
# B2 is the best accuracy/speed tradeoff for most EO tasks
```

## 5.6 Foundation Models

```python
from pygeovision.models.foundation.prithvi import PrithviTasks

tasks  = PrithviTasks('prithvi_eo_2_0')
result = tasks.land_cover(ready_array, source='sentinel2')
print(result['class_distribution'])
# Tree cover: 23.1%, Cropland: 41.2%, Urban: 8.9%, ...
```

## 5.7 Transfer Learning Best Practices

```python
# Freeze encoder, train decoder only (data-efficient)
for name, p in model.named_parameters():
    p.requires_grad = 'encoder' not in name

# OneCycleLR scheduler (fast convergence)
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sch = OneCycleLR(opt, max_lr=3e-4, epochs=100,
                  steps_per_epoch=len(train_loader),
                  pct_start=0.1)
```

## 5.8 Model Evaluation

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def mean_iou(true, pred, n_classes):
    cm  = confusion_matrix(true.ravel(), pred.ravel(),
                            labels=list(range(n_classes)))
    ious = []
    for i in range(n_classes):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        ious.append(tp / (tp+fp+fn+1e-8))
    print(f'mIoU: {np.mean(ious):.4f}')
    return float(np.mean(ious))
```

## Summary

- Deep learning outperforms handcrafted methods on most EO tasks.
- Use FocalDice loss for imbalanced classes; geographic splits for honest evaluation.
- PyGeoVision provides 119 models across 8 task categories.
- Foundation models (Prithvi, DINOv3) enable zero-shot inference.

## Exercises

1. List all segmentation models. Compare U-Net vs SegFormer parameter counts.
2. Adapt a ResNet50 model to accept 12-band Sentinel-2 input.
3. Implement geographic k-fold cross-validation using a 1-degree grid.
4. Plot a training curve. When does the validation mIoU plateau?
5. Run Prithvi zero-shot vs fine-tuned comparison on 50 labelled patches.