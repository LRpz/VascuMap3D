# VascuMap3D Technical Documentation

> **A Complete Guide to 3D Vascular Network Segmentation and Analysis**

---

## Table of Contents

1. [Introduction & Background](#1-introduction--background)
   - [1.1 What is VascuMap3D?](#11-what-is-vascumap3d)
   - [1.2 Pipeline Overview](#12-pipeline-overview)
   - [1.3 Key Concepts](#13-key-concepts)
   - [1.4 System Requirements](#14-system-requirements)

2. [Detailed Workflow](#2-detailed-workflow)
   - [2.1 Training](#21-training)
     - [2.1.1 Image Translation (3D Pix2Pix)](#211-image-translation-3d-pix2pix)
     - [2.1.2 Segmentation (2D U-Net)](#212-segmentation-2d-u-net)
   - [2.2 Inference Pipeline](#22-inference-pipeline)
     - [2.2.1 Data Loading & Voxel Normalization](#221-data-loading--voxel-normalization)
     - [2.2.2 Preprocessing & Cropping](#222-preprocessing--cropping)
     - [2.2.3 Image Translation Inference](#223-image-translation-inference)
     - [2.2.4 Segmentation Inference (2.5D)](#224-segmentation-inference-25d)
   - [2.3 Graph Analysis](#23-graph-analysis)
     - [2.3.1 Skeletonization](#231-skeletonization)
     - [2.3.2 Graph Construction](#232-graph-construction)
     - [2.3.3 Metrics Computation](#233-metrics-computation)
     - [2.3.4 Output Files](#234-output-files)

3. [Function Reference](#3-function-reference)
   - [3.1 Image Translation Module](#31-image-translation-module)
   - [3.2 Segmentation Module](#32-segmentation-module)
   - [3.3 Pipeline Module](#33-pipeline-module)
   - [3.4 Graph Analysis Module](#34-graph-analysis-module)
   - [3.5 Utilities](#35-utilities)

4. [Metrics Reference](#4-metrics-reference)
   - [4.1 Global Metrics (CSV)](#41-global-metrics-csv)
   - [4.2 Vessel Metrics (HDF5)](#42-vessel-metrics-hdf5)
   - [4.3 Junction Metrics (HDF5)](#43-junction-metrics-hdf5)

5. [Appendix](#5-appendix)
   - [5.1 Mathematical Formulas](#51-mathematical-formulas)
   - [5.2 Troubleshooting](#52-troubleshooting)
   - [5.3 Contributing](#53-contributing)

---

# 1. Introduction & Background

## 1.1 What is VascuMap3D?

VascuMap3D is an end-to-end computational pipeline for analyzing 3D vascular networks from microscopy images. It solves a common challenge in biological imaging: extracting quantitative morphometric data from complex, branching vessel structures.

**Target users:**
- **Biologists** studying angiogenesis, vasculogenesis, or tumor vasculature
- **Biomedical engineers** developing tissue engineering constructs
- **Computational scientists** building on or extending the pipeline

## 1.2 Pipeline Overview

The pipeline processes raw microscopy images through four main stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VascuMap3D Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STAGE 1: IMAGE TRANSLATION                                                 │
│   ──────────────────────────                                                 │
│   Input:  Brightfield Z-stack (.lif file)                                   │
│   Output: Fluorescence-like volume                                          │
│   Method: 3D Pix2Pix GAN                                                    │
│                                                                              │
│   Why? Brightfield images have low contrast. We translate them to           │
│   fluorescence-like images where vessels are clearly visible.               │
│                                                                              │
│   Skipped when a fluorescence vessel channel is given (--fluo_channel or    │
│   --fluo_index, e.g. the GFP channel of an .nd2 file): that channel goes    │
│   directly to Stage 2.                                                      │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│   STAGE 2: SEGMENTATION                                                      │
│   ─────────────────────                                                      │
│   Input:  Fluorescence-like volume or fluorescence channel                  │
│   Output: Binary vessel mask                                                │
│   Method: 2D U-Net with 2.5D inference (3-plane averaging)                  │
│                                                                              │
│   Why? 2D models are faster and have better pretrained weights.             │
│   We run inference on axial, coronal, and sagittal planes, then average.    │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│   STAGE 3: SKELETONIZATION                                                   │
│   ────────────────────────                                                   │
│   Input:  Binary vessel mask                                                │
│   Output: 1-pixel-wide skeleton                                             │
│   Method: Morphological thinning OR mesh contraction                        │
│                                                                              │
│   Why? We need to reduce vessels to centerlines to extract topology.        │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│   STAGE 4: GRAPH ANALYSIS                                                    │
│   ───────────────────────                                                    │
│   Input:  Skeleton image                                                    │
│   Output: NetworkX graph + CSV/HDF5 metrics in µm, µm², µm³                 │
│   Method: sknw graph extraction, pruning + custom metric computation        │
│                                                                              │
│   Why? Graphs let us compute topological metrics: branching, tortuosity,    │
│   vessel lengths, fractal dimension, etc.                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Data flow summary:**
```
Brightfield input:   .lif file → Brightfield Z-stack → Pseudo-fluorescence → Binary mask → Skeleton → Graph → Metrics
Fluorescence input:  .nd2/.czi/.lif file → Fluorescence channel ───────────→ Binary mask → Skeleton → Graph → Metrics
```

## 1.3 Key Concepts

### Voxel Size and Isotropy

Raw microscopy data has **anisotropic voxels** (Z spacing ≠ XY spacing). This causes problems for skeletonization and metric computation. VascuMap3D handles this with two-stage rescaling:

```
Original acquisition:     Variable (depends on microscope settings)
        ↓ Stage 1
Reference voxel size:     5 × 2 × 2 µm (Z × Y × X) - Standardized
        ↓ Stage 2  
Isotropic for analysis:   2 × 2 × 2 µm (Z × Y × X) - Same in all dimensions
```

**Why 5×2×2 µm as reference?**
- Z = 5 µm: Typical Z-step for confocal/brightfield
- Y = X = 2 µm: Common lateral resolution at 10-20× magnification

**Why make it isotropic?**
- Skeletonization assumes equal spacing in all dimensions
- Length/tortuosity metrics need consistent units
- Cross-sectional area requires isotropic sampling

### Graph Representation

Vascular networks are naturally represented as graphs:
- **Nodes** = Junction points (where vessels meet) + Endpoints (sprout tips)
- **Edges** = Vessel segments between nodes
- **Edge attributes** = Length, cross-sectional area, tortuosity, orientation

### Key Terminology

| Term | Definition |
|------|------------|
| **Vessel segment** | A continuous tube between two junction points |
| **Junction** | Where vessels meet or bifurcate (node degree ≥ 3) |
| **Endpoint/Sprout tip** | Terminal point of a vessel (node degree = 1 in the skeleton graph before pruning) |
| **Sprout** | Vessel segment connected to an endpoint |
| **Branch** | Vessel segment between two junctions |
| **Tortuosity** | Path length / straight-line distance (≥1, higher = more curved) |
| **Chip volume** | Convex-hull volume of the vessel mask inside the border margins; the reference volume of the density metrics |
| **EDT** | Euclidean Distance Transform: distance to nearest boundary |

## 1.4 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.9 (tested) |
| GPU | 8GB VRAM | 16GB+ VRAM |
| RAM | 16GB | 32GB+ |
| CUDA | 11.x | 12.x |

**Installation:**
```bash
git clone <repository_url>
cd VascuMap3D
pip install -r requirements.txt
```

The requirements were tested with Python 3.9, PyTorch 2.8 and CUDA 12.8. Because `requirements.txt` installs `cupy-cuda12x`, a different CUDA toolkit requires the matching CuPy wheel (`cupy-cuda11x` or `cupy-cuda13x`) instead. The packages `tables` (HDF5 output of the per-vessel and per-junction tables) and `nd2` (Nikon `.nd2` input through aicsimageio) are also listed there.

---

# 2. Detailed Workflow

## 2.1 Training

> **Note for beginners:** This section explains how to train the deep learning models. Readers new to deep learning will find the key concepts explained along the way. Training requires a GPU with at least 8 GB of memory.

### 2.1.1 Image Translation (3D Pix2Pix)

#### What is Image Translation?

Image translation teaches a network to "translate" one type of image into another, much as machine translation converts English into French. In our case:

- **Input**: Brightfield microscopy (low contrast, vessels hard to see)
- **Output**: Fluorescence-like image (high contrast, vessels clearly visible)

The model learns this translation from many paired examples, each made of a brightfield image and the fluorescence image of the same sample.

#### How Does Pix2Pix Work?

Pix2Pix is a type of **Generative Adversarial Network (GAN)** - a clever training setup with two neural networks competing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Pix2Pix Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GENERATOR (The Artist)                                                     │
│   ──────────────────────                                                     │
│   • Takes brightfield image as input                                        │
│   • Tries to create a convincing fluorescence-like image                    │
│   • Structure: 3D U-Net (encodes image, then decodes to output)            │
│   • Goal: Fool the discriminator AND match the real fluorescence           │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│   DISCRIMINATOR (The Critic)                                                 │
│   ──────────────────────────                                                 │
│   • Sees pairs: (brightfield + generated) vs (brightfield + real)          │
│   • Tries to tell which fluorescence image is real vs fake                 │
│   • Structure: 3D PatchGAN (looks at small patches, not whole image)       │
│   • Goal: Correctly identify fake images                                   │
│                                                                              │
│   TRAINING LOOP                                                              │
│   ─────────────                                                              │
│   1. Generator creates fake fluorescence from brightfield                   │
│   2. Discriminator tries to spot the fake                                   │
│   3. Generator improves to fool discriminator better                        │
│   4. Discriminator improves to catch fakes better                           │
│   5. Repeat until generator makes realistic images                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why 3D?

Unlike standard Pix2Pix which works on 2D images, we use **3D convolutions** because:
- Vessels are 3D structures, so context from adjacent Z-slices helps
- Maintains spatial coherence across the Z-stack
- Better handles vessels that are tilted or curved through the volume

#### Loss Functions Explained

The generator is trained with multiple loss functions working together:

| Loss | What it Does | Why it Matters |
|------|-------------|----------------|
| **Adversarial (BCE)** | Encourages realistic-looking output | Makes images that "look right" |
| **L1 (Mean Absolute Error)** | Penalizes pixel-by-pixel differences | Ensures correct brightness/structure |
| **MSE (Mean Squared Error)** | Like L1 but penalizes large errors more | Prevents major mistakes |

Combined: `Total Loss = Adversarial + 100×L1 + 100×MSE`

The factor of 100 (λ) gives more weight to matching the actual image than just fooling the discriminator.

#### Input Data Format

Training requires **paired** 3D volumes: each brightfield volume must have a corresponding fluorescence volume:

```
/data/
├── input/                          # Brightfield volumes
│   ├── sample_001.nii.gz
│   ├── sample_002.nii.gz
│   └── ...
└── target/                         # Fluorescence volumes (ground truth)
    ├── sample_001.nii.gz           # MUST match input filenames exactly
    ├── sample_002.nii.gz
    └── ...
```

**Volume requirements:**

| Property | Requirement | Why |
|----------|-------------|-----|
| Format | NIfTI (`.nii.gz`) | Standard medical imaging format |
| Dimensions | 3D (Z, Y, X) | Typically 16-32 × 256 × 256 voxels |
| Values | Normalized to [0, 1] | Neural networks work best with small numbers |
| Data type | Float32 | Sufficient precision, reasonable memory |
| Voxel size | Should match | Input and target must have same voxel dimensions |

**Preparing your data:**

```python
import nibabel as nib
import numpy as np

# Load your 3D volume
volume = ...  # Your data loading

# Normalize to [0, 1]
volume = (volume - volume.min()) / (volume.max() - volume.min())

# Save as NIfTI
nifti_img = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
nib.save(nifti_img, 'sample_001.nii.gz')
```

#### Training Command

```bash
python -m vascumap.models.image_translation.train \
    --input_path /path/to/input \
    --target_path /path/to/target \
    --model_path /path/to/save \
    --batch_size 4 \
    --epochs 200
```

**Key parameters:**

| Parameter | Default | What it Does | Recommendations |
|-----------|---------|--------------|-----------------|
| `batch_size` | 4 | Samples per training step | 4 to 6 only (see note below) |
| `epochs` | 200 | Full passes through dataset | More epochs = longer training |
| `generator_lr` | 1e-3 | Generator learning rate | Change only if needed |
| `discriminator_lr` | 1e-5 | Discriminator learning rate | Much lower than generator |
| `generator_dropout` | 0.1 | Dropout in the generator U-Net | - |
| `discriminator_dropout` | 0.25 | Passed to the discriminator, whose layers contain no dropout | - |
| `val_split` | 0.1 | Fraction of volumes held out for validation | - |
| `num_workers` | 0 | DataLoader worker processes | - |
| `device` | `cuda` | Device for the architecture summaries | - |

**Note on batch size:** `Pix2Pix.training_step` splits every training batch into a discriminator half and a generator half of at most three volumes each (`TRAIN_BATCH_SIZE=6`), and the training loader drops incomplete batches. Consequently, a batch size between 4 and 6 is required, because a smaller batch yields a single half and a larger batch yields more than two.

**Key functions:** [`ImageDataset3D`](#imagedataset3d), [`Generator`](#generator), [`Discriminator`](#discriminator), [`Pix2Pix`](#pix2pix)

#### Monitoring Training

Launch TensorBoard to visualize training progress:

```bash
tensorboard --logdir /path/to/save
```

Then open `http://localhost:6006` in your browser.

**Metrics to watch:**

| Metric | What it Means | Good Values | Concerning Signs |
|--------|---------------|-------------|------------------|
| `train_g_loss` | Generator total loss | Decreasing over time | Stuck high or increasing |
| `train_d_loss` | Discriminator loss | Stable around 0.5-1.0 | Very low (<0.1) or very high (>2) |
| `val_g_psnr` | Image quality (Peak Signal-to-Noise) | > 20 dB | < 15 dB |
| `val_g_ssim` | Structural similarity | > 0.5 | < 0.3 |
| `val_g_accuracy` | How often generator fools discriminator | ~50% | 0% or 100% |
| `val_d_accuracy` | Discriminator correctness | ~50% | 0% or 100% |

**What "50% discriminator accuracy" means:**
- If the discriminator is too good (accuracy > 90%), the generator cannot learn
- If the discriminator is too bad (accuracy < 10%), the generator has no guidance
- 50% means they are evenly matched, which is ideal for learning

**Common problems and solutions:**

| Problem | Symptom | Solution |
|---------|---------|----------|
| Mode collapse | Generator produces same output for all inputs | Lower learning rates, add dropout |
| Discriminator wins | D accuracy > 90%, G loss not decreasing | Reduce discriminator_lr |
| Generator wins | D accuracy < 10%, images look blurry | Increase discriminator_lr |
| Out of memory | CUDA out of memory error | Reduce batch_size (minimum 4) or the volume size |

#### When is Training Done?

Training is complete when:
1. **val_g_psnr** stabilizes above 20 dB
2. **val_g_ssim** stabilizes above 0.5
3. Visual inspection shows good quality images

Validation runs twice per epoch, and the five checkpoints with the highest validation structural similarity (`val_g_ssim`) are kept in the model directory as `{epoch}-{val_g_psnr}-{val_g_ssim}.ckpt`. At the end of training, the final weights are also written as a bare state dictionary to `last_epoch_full.pth`. Since [`load_pix2pix`](#load_pix2pix) reads the `state_dict` entry of a Lightning checkpoint, inference requires one of the `.ckpt` files.

---

### 2.1.2 Segmentation (2D U-Net)

#### What is Segmentation?

Segmentation is labeling each pixel in an image. For vessel segmentation:
- **Input**: Grayscale microscopy image
- **Output**: Binary mask where white = vessel, black = background

#### Why 2D Instead of 3D?

We use a 2D model because:

| Advantage | Explanation |
|-----------|-------------|
| **Pretrained weights** | 2D models have weights trained on millions of images (ImageNet) |
| **Lower memory** | 3D models need 10-100× more GPU memory |
| **More data** | Can extract many 2D slices from each 3D volume |
| **Proven performance** | 2D + 2.5D inference works as well as 3D for vessels |

To get 3D consistency, we use **2.5D inference** (see [Section 2.2.4](#224-segmentation-inference-25d)).

#### Understanding U-Net Architecture

U-Net has two parts: an **encoder** (compresses the image) and a **decoder** (expands back to full resolution):

```
U-Net Architecture
                                                                      
  Input (256×256)                                         Output (256×256)
       │                                                       ▲
       ▼                                                       │
  ┌─────────┐                                             ┌─────────┐
  │ Encoder │ ────────── Skip Connection ─────────────────│ Decoder │
  │ Block 1 │                                             │ Block 1 │
  └────┬────┘                                             └────▲────┘
       │ (Downsample)                               (Upsample) │
       ▼                                                       │
  ┌─────────┐                                             ┌─────────┐
  │ Encoder │ ────────── Skip Connection ─────────────────│ Decoder │
  │ Block 2 │                                             │ Block 2 │
  └────┬────┘                                             └────▲────┘
       │                                                       │
       ▼                                                       │
       ... (more layers) ...                                   ...
       │                                                       │
       ▼                                                       │
  ┌─────────────────────────────────────────────────────────────┐
  │                      Bottleneck                              │
  │              (Smallest, most abstract features)              │
  └─────────────────────────────────────────────────────────────┘

Skip Connections: Pass high-resolution details directly to decoder
Bottleneck: Learns "what is a vessel" at abstract level
```

**The encoder-decoder design:**
- **Encoder**: Progressively shrinks image, learning "what" is present
- **Decoder**: Progressively expands, learning "where" things are
- **Skip connections**: Copy fine details from encoder to decoder (critical for precise boundaries)

#### Choosing an Encoder

The encoder is the "backbone" that extracts features. We use pretrained encoders from `segmentation_models_pytorch`:

| Encoder Family | Examples | Characteristics | Recommendation |
|----------------|----------|-----------------|----------------|
| **MiT (SegFormer)** | `mit_b0` to `mit_b5` | Transformer-based, best accuracy | **Recommended** |
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101` | Classic CNN, fast | Good baseline |
| EfficientNet | `efficientnet-b0` to `efficientnet-b7` | Efficient, good accuracy/speed | Good alternative |
| SE-ResNet | `se_resnet50`, `se_resnext50_32x4d` | ResNet with attention | Slightly better than ResNet |

**MiT-B5 (our recommendation):**
- Based on SegFormer transformer architecture
- Excellent at capturing long-range dependencies (important for connected vessels)
- `b0` to `b5` = increasing model size (b5 is largest/most accurate)

#### Choosing a Model Architecture

The "model" is the overall structure (U-Net is most common):

| Model | Description | When to Use |
|-------|-------------|-------------|
| **Unet** | Classic encoder-decoder with skip connections | Default choice, works for most cases |
| **Unet++** | Nested skip connections | Slightly better for small structures |
| **DeepLabV3+** | Atrous convolutions for multi-scale | Good for varying vessel sizes |
| **FPN** | Feature Pyramid Network | Good for multi-scale detection |
| **MAnet** | Multi-scale attention | Variable feature scales |

**For vessel segmentation, we recommend: `Unet` + `mit_b5`**

The inference pipeline depends on this choice: [`load_segmentation_model`](#load_segmentation_model) always rebuilds a `Unet` with an `mit_b5` encoder and a one-channel input, so a model trained with another architecture or encoder can only be used for inference after the corresponding change in `vascumap/pipeline/prediction.py`.

#### Loss Functions Explained

We use two losses that complement each other:

| Loss | Formula | What it Does | Why Needed |
|------|---------|--------------|------------|
| **Dice Loss** | `1 - 2×(pred∩true)/(pred+true)` | Measures overlap between prediction and ground truth | Handles class imbalance (few vessel pixels vs many background) |
| **BCE Loss** | `-[y×log(p) + (1-y)×log(1-p)]` | Per-pixel classification loss | Provides stable gradients for learning |

**Why both?** Dice alone can be unstable early in training; BCE alone struggles with class imbalance. Together they work well.

#### Input Data Format

You need paired images and masks:

```
/data/
├── images/                         # Grayscale microscopy crops
│   ├── crop_001.tif
│   ├── crop_002.tif
│   └── ...
└── masks/                          # Binary ground truth masks
    ├── crop_001.tif                # MUST have same filename as image
    ├── crop_002.tif
    └── ...
```

**Image requirements:**

| Property | Requirement | Why |
|----------|-------------|-----|
| Format | TIFF (`.tif`) | Preserves quality, no compression artifacts |
| Dimensions | 256 × 256 pixels | Balance of detail and memory usage; also the tile size used at inference |
| Channels | 1 (grayscale) | Microscopy is typically grayscale |
| Values | Raw intensities accepted | Normalized on loading (see below) |
| Preprocessing | Done by the dataset | Each tile is clipped to its 0.1st–99.9th percentiles, rescaled to [0, 1] and cast to float32 |

**Mask requirements:**

| Property | Requirement | Why |
|----------|-------------|-----|
| Format | Same as images | Consistency |
| Dimensions | Same as corresponding image | Must align pixel-by-pixel |
| Values | Any value > 0 is vessel; 0 and NaN are background | Binarized on loading |
| Data type | Any numeric type | Converted to integer labels by the dataset |

Images and masks are paired by their position in the sorted file lists, so identical file names in the two folders guarantee correct pairing. A random 10% of the pairs (`test_size=0.1`, `seed=0`) is held out for validation.

#### Preparing Training Data

**Step 1: Extract 2D crops from your 3D volumes**

```python
import os
import tifffile
import numpy as np

# Load your 3D stack
stack = tifffile.imread('volume.tif')  # Shape: (Z, Y, X)
mask_3d = tifffile.imread('mask.tif')

os.makedirs('images', exist_ok=True)
os.makedirs('masks', exist_ok=True)

# Extract non-overlapping 256×256 crops from each Z-slice
crop_size = 256
_, ny, nx = stack.shape
for z in range(stack.shape[0]):
    i = 0
    for y in range(0, ny - crop_size + 1, crop_size):
        for x in range(0, nx - crop_size + 1, crop_size):
            # Image crop: no normalization needed, SegmentationDataset clips each tile
            # to its 0.1-99.9th percentiles and rescales it to [0, 1] on loading
            img_crop = stack[z, y:y+crop_size, x:x+crop_size].astype(np.float32)

            # Corresponding mask crop
            mask_crop = (mask_3d[z, y:y+crop_size, x:x+crop_size] > 0).astype(np.float32)

            # Save
            tifffile.imwrite(f'images/z{z:03d}_crop{i:03d}.tif', img_crop)
            tifffile.imwrite(f'masks/z{z:03d}_crop{i:03d}.tif', mask_crop)
            i += 1
```

**Step 2: Verify your data**

```python
# Check a few samples
img = tifffile.imread('images/z000_crop000.tif')
mask = tifffile.imread('masks/z000_crop000.tif')

print(f"Image shape: {img.shape}")        # Should be (256, 256)
print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")  # Any range; rescaled on loading
print(f"Mask unique values: {np.unique(mask)}")  # Should be [0, 1]
```

**Key function:** [`SegmentationDataset`](#segmentationdataset)

#### Handling Grayscale with Pretrained Weights

Most pretrained encoders expect 3-channel RGB images. Our microscopy is 1-channel grayscale. The code automatically adapts the first layer:

```python
# Original first layer (MiT patch embedding): Conv2d(3, 64, kernel_size=7, stride=4)
#   - Expects 3 input channels (R, G, B)
#   - Each channel has its own weights

# Adapted layer: Conv2d(1, 64, kernel_size=7, stride=4)
#   - Accepts 1 input channel (grayscale)
#   - Weights = sum of original RGB weights

# For MiT encoders this is done automatically by adapt_input_model(),
# called by build_model() when in_channels=1
```

For the other encoders, `segmentation_models_pytorch` performs the same weight summation itself when the model is built with `in_channels=1`.

**Why summing works:** The RGB weights learned complementary features. Summing them creates a reasonable grayscale feature extractor.

**Key function:** [`adapt_input_model`](#adapt_input_model)

#### Data Augmentation

Training tiles are augmented on the fly with albumentations (version 2.0 or later), whereas validation tiles are only converted to tensors:

| Transform | Parameters | Probability |
|-----------|------------|-------------|
| `HorizontalFlip` | - | 0.5 |
| `VerticalFlip` | - | 0.5 |
| `Affine` | translation ±6.25%, scale 0.9–1.1, rotation ±45°, mirrored border | 0.5 |
| `ToTensorV2` | conversion to a PyTorch tensor | always |

**Key functions:** [`hard_transforms`](#hard_transforms), [`load_data`](#load_data)

#### Training Command

```bash
python -m vascumap.models.segmentation.training \
    --images_dir_path /path/to/images \
    --masks_dir_path /path/to/masks \
    --model_dir_path /path/to/save \
    --model_architecture Unet \
    --encoder_architecture mit_b5 \
    --input_channels 1 \
    --weights imagenet \
    --epochs 200 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --fp16
```

**All parameters explained:**

| Parameter | Default | What it Does | Recommendations |
|-----------|---------|--------------|-----------------|
| `--images_dir_path` | Required | Folder with training images | - |
| `--masks_dir_path` | Required | Folder with training masks | - |
| `--model_dir_path` | Required | Where to save checkpoints | - |
| `--model_architecture` | `Unet` | Model type (see table above) | Start with `Unet` |
| `--encoder_architecture` | `mit_b5` | Encoder backbone | `mit_b5` for best accuracy |
| `--input_channels` | `1` | Number of image channels | `1` for grayscale |
| `--weights` | `imagenet` | Pretrained weights | Always use `imagenet` |
| `--epochs` | `200` | Training iterations | 100-300 depending on dataset |
| `--batch_size` | `16` | Samples per step | Reduce if out of memory |
| `--learning_rate` | `1e-3` | Step size for optimization | Change only if needed |
| `--fp16` | `False` | Use 16-bit mixed precision (`precision="16-mixed"`) | Add flag to save memory |

**Key functions:** [`SegmentationModule`](#segmentationmodule), [`build_model`](#build_model), [`load_data`](#load_data)

#### Monitoring Training

The training logs show these metrics:

| Metric | What it Means | Good Values | Warning Signs |
|--------|---------------|-------------|---------------|
| `train_loss` | Combined Dice+BCE on training data | Decreasing | Stuck or increasing |
| `val_loss` | Loss on validation data | Decreasing, close to train_loss | Much higher than train_loss (overfitting) |
| `train_dice` | Overlap between prediction and truth | Increasing toward 1.0 | Stuck below 0.5 |
| `val_dice` | Dice on validation data | > 0.7 | < 0.5 |
| `train_iou` | Intersection over Union | Increasing toward 1.0 | Stuck low |
| `val_iou` | IoU on validation data | > 0.6 | < 0.4 |

For a binary mask, the Dice coefficient equals the F1 score; the Dice metrics are therefore computed with torchmetrics `BinaryF1Score`, and IoU with the binary `JaccardIndex`, both on sigmoid probabilities thresholded at 0.5. `train_loss` is logged at every step, whereas the Dice and IoU values are averaged over each epoch.

**Understanding Dice and IoU:**

```
Dice = 2 × (Overlap) / (Predicted + Truth)
IoU  = Overlap / (Predicted ∪ Truth)

Example:
  Predicted:  ████░░░░     (4 pixels)
  Truth:      ░░████░░     (4 pixels)
  Overlap:    ░░██░░░░     (2 pixels)
  
  Dice = 2×2 / (4+4) = 0.5
  IoU  = 2 / (4+4-2) = 0.33
```

**Learning rate schedule:**

The learning rate is automatically reduced halfway through training:
- Epochs 1-100: LR = 0.001
- Epochs 101-200: LR = 0.0005

This helps fine-tune the model after initial learning.

**When is training done?**

1. `val_dice` > 0.7 and `val_iou` > 0.6
2. Validation loss has stopped improving for 20+ epochs
3. Visual inspection shows good segmentations

The three checkpoints with the lowest `val_loss` are kept in the model directory as `{epoch}-{val_loss}-{val_dice}.ckpt`, and the learning rate is logged once per epoch. For a `Unet` with an `mit_b5` encoder, these Lightning checkpoints can be passed directly to the inference pipeline.

---

## 2.2 Inference Pipeline

### 2.2.1 Data Loading & Voxel Normalization

The pipeline reads microscopy files and normalizes voxel sizes. Brightfield input is read from Leica LIF files with readlif, all channels included:

```python
from vascumap.pipeline.data import get_stack_from_lif

stack, voxel_size, [nz, ny, nx] = get_stack_from_lif(
    lif_file='sample.lif',
    stack_index=0,
    ref_voxel_size=[5, 2, 2]  # Target: 5×2×2 µm
)
# Returns: (Z, Y, X, C) array, rescaled to reference voxel size
```

When a fluorescence vessel channel is segmented directly, any format readable by aicsimageio (for example `.nd2`, `.czi` or `.lif`) is read scene by scene, and only the requested channel is returned:

```python
from aicsimageio import AICSImage
from vascumap.pipeline.data import get_stack_from_aics

img = AICSImage('sample.nd2')
img.set_scene(img.scenes[0])     # the scene must be selected by the caller

stack, voxel_size, [nz, ny, nx] = get_stack_from_aics(
    img,
    channel_index=1,             # e.g. the GFP channel
    ref_voxel_size=[5, 2, 2]
)
# Returns: (Z, Y, X, 1) array, rescaled to reference voxel size
```

**What happens internally:**
1. Read voxel size from the file metadata (LIF scale, or the aicsimageio physical pixel sizes, which must be present)
2. Validate Z-range ≥ 160 µm (a `ValueError` is raised otherwise)
3. Calculate rescale factors: `original_voxel / ref_voxel`
4. Resample volume using cubic interpolation

**Key functions:** [`get_stack_from_lif`](#get_stack_from_lif), [`get_stack_from_aics`](#get_stack_from_aics)

---

### 2.2.2 Preprocessing & Cropping

The cropping pipeline extracts the region of interest (ROI) of each stack in four stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CROPPING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STAGE 1: AUTO-FOCUS                   Output: Height map + focused image  │
│   ─────────────────────                                                      │
│   Find the best Z for each XY position                                      │
│                                                                              │
│                   ↓                                                          │
│                                                                              │
│   STAGE 2: EDGE DETECTION               Output: Edge map                    │
│   ─────────────────────                                                      │
│   Find sample boundaries in the focused image                               │
│                                                                              │
│                   ↓                                                          │
│                                                                              │
│   STAGE 3: TEMPLATE REGISTRATION        Output: Transformation parameters  │
│   ────────────────────────────                                               │
│   Match a reference template to the detected edges                          │
│                                                                              │
│                   ↓                                                          │
│                                                                              │
│   STAGE 4: GEOMETRIC CROPPING           Output: Cropped stack               │
│   ─────────────────────────                                                  │
│   Extract the registered ROI in both Z and XY                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Stage 1: Auto-Focus (Finding the Curved Surface)

**The Problem:**
The sample surface is curved. If we just take a single Z-slice, some areas would be in focus and others blurry. We need to find the "surface" of the sample.

**The Solution:**
Create a **height map** - for each (X, Y) position, store which Z-slice is in focus.

**Algorithm Walkthrough:**

```
Step 1: Sample sparse grid points
─────────────────────────────────
We don't check every pixel (too slow). Instead, we sample a grid:

Full image (1000×1000 pixels):
┌────────────────────────────────────┐
│  ·     ·     ·     ·     ·     ·   │
│                                     │
│  ·     ·     ·     ·     ·     ·   │    · = sampling point
│                                     │        (n_sampling × n_sampling)
│  ·     ·     ·     ·     ·     ·   │
│                                     │
│  ·     ·     ·     ·     ·     ·   │
└────────────────────────────────────┘

Step 2: At each point, find the best Z
────────────────────────────────────────
For each sampling point, we look at a patch around it across all Z-slices:

Z=0:  ░░░░░    Z=1:  ░░▒░░    Z=2:  ░▒▓▒░    Z=3:  ░░▒░░    Z=4:  ░░░░░
      ░░░░░          ░░▒░░          ▒▓█▓▒          ░░▒░░          ░░░░░
      ░░░░░          ░░▒░░          ░▒▓▒░          ░░▒░░          ░░░░░
      Blurry         Sharper        SHARPEST       Sharper        Blurry

Focus measure = std(sobel(patch))
  - Sobel detects edges
  - Sharp images have more edges → higher std
  - Select Z with maximum focus measure

Step 3: Fit a plane to the sparse points
──────────────────────────────────────────
We have (x, y, z_best) for each sampling point. We fit a plane:

    Z = a·X + b·Y + c

This smooths out noise and gives us a continuous surface.

Step 4: Create dense height map
────────────────────────────────
Apply the plane equation to every pixel to get the full height map.
```

**Code:**

```python
from vascumap.pipeline.preprocessing import auto_focus

focused_image, height_map = auto_focus(
    stack_bf,           # 3D brightfield stack: (Z, Y, X)
    crop_window=50,     # Pixels to exclude from border
    n_sampling=20,      # Grid will be 20×20 = 400 points
    window_size=50      # Each patch is 100×100 pixels (±50)
)

# focused_image: 2D image showing the in-focus surface
# height_map: 2D array of Z-indices (same shape as focused_image)
```

**Parameters:**

| Parameter | What it Does | Typical Value | Notes |
|-----------|--------------|---------------|-------|
| `crop_window` | Excludes image borders | 50-200 | Borders often have artifacts |
| `n_sampling` | Grid density | 5-20 | More = slower but more accurate |
| `window_size` | Patch size for focus | 50-100 | Larger = more robust to noise |

Within [`crop_stack`](#crop_stack), `auto_focus` is called with `crop_window` set to 20% of the stack width (at least 200 pixels), `n_sampling=5` and `window_size=400`.

**Key function:** [`auto_focus`](#auto_focus)

---

#### Stage 2: Edge Detection (Finding Sample Boundaries)

**The Problem:**
We need to find where the sample is in the image so we can align our template to it.

**The Solution:**
Use Sobel edge detection to highlight the sample boundaries.

**How Sobel Works:**

```
Sobel detects intensity changes (edges) in an image:

Original:                    After Sobel:
┌────────────────────┐       ┌────────────────────┐
│████████░░░░░░░░░░░░│       │░░░░░░░█░░░░░░░░░░░░│  ← Edge detected
│████████░░░░░░░░░░░░│       │░░░░░░░█░░░░░░░░░░░░│    at boundary
│████████░░░░░░░░░░░░│  →    │░░░░░░░█░░░░░░░░░░░░│
│████████░░░░░░░░░░░░│       │░░░░░░░█░░░░░░░░░░░░│
│░░░░░░░░░░░░░░░░░░░░│       │░░░░░░░░░░░░░░░░░░░░│
└────────────────────┘       └────────────────────┘
```

**Processing Steps:**

```python
# 0. Clip to the 0.1-99.9th percentiles and rescale to [0, 1]
im = scale(contrast(focused_image, 0.1, 99.9))

# 1. Downsample for speed (10% of original size)
im_r = rescale(im, 0.1)

# 2. Compute Sobel gradients in both directions
edges_x = sobel_h(im_r)  # Horizontal edges
edges_y = sobel_v(im_r)  # Vertical edges

# 3. Combine into gradient magnitude
edges = sqrt(edges_x² + edges_y²)

# 4. Smooth to reduce noise (after trimming a 3-pixel border and
#    edge-padding the rows, which limits border artifacts)
edges = gaussian(edges, sigma=2.5)

# 5. Threshold high values (removes noise)
edges = clip(edges, 0, 0.15)

# 6. Resize back to original
edges = resize(edges, original_shape)
```

---

#### Stage 3: Template Registration (Aligning to a Reference)

**The Problem:**
We have edges showing the sample boundary. We need to figure out exactly where to crop. This is solved by aligning a **pre-defined template** to the detected edges.

**What is a Template?**

A template consists of two files, stored in the crop profile folder (`profile_dir` of `crop_stack`, `--crop_profile_dir` on the command line; default `crop_profiles`, relative to the working directory):

```
crop_profiles/
├── {profile}_fiducials.npy    # Points that should align with edges
└── {profile}_rect.npy         # The ROI we want to extract
```

**Fiducials:** Points along the sample boundary that we expect to see in every image.
**Rectangle:** The four corners of the region we want to extract.

If either file is missing, `crop_stack` raises a `FileNotFoundError` that names both expected paths.


**The Registration Process:**

We find the transformation (scale, rotation, translation) that best aligns the template fiducials with the detected edges.

```
Transformation Parameters:
  - scale: How much to resize the template (bounded to 0.999-1.001, i.e. fixed at ~1.0)
  - theta: Rotation angle in radians (bounded to ±1.5)
  - tx: Translation added to the first (y) coordinate column (bounded to ±500 pixels)
  - ty: Translation added to the second (x) coordinate column (bounded to ±500 pixels)

Mathematical formulation:
  
  transformed_point = scale × (original_point × rotation_matrix) + translation
  
  where rotation_matrix = [cos(θ)  -sin(θ)]
                          [sin(θ)   cos(θ)]
```

**Optimization Strategy:**

Finding the best transformation is an optimization problem. We use a two-stage approach:

```
Stage 1: BASIN-HOPPING (Global Search)
────────────────────────────────────────
Problem: The objective function has many local minima.
Solution: Start from three points (the template centered on the image and two
perturbed variants). From each, make 50 random hops (scale ±0.1, rotation
±0.1 rad, translation ±20 pixels; hops outside the bounds are rejected), each
followed by a bounded Powell local optimization.


Stage 2: POWELL (Local Refinement)
──────────────────────────────────
Refine each basin-hopping result with Powell (up to 2000 iterations) and keep
the lowest score over the three starting points.
Powell method: Optimizes one parameter at a time, then repeats.
```

Because the scale is effectively fixed, the template must be drawn at the pixel size of the stacks it is registered to (2 µm in XY after resampling to the reference voxel size).

**The Objective Function:**

```python
def objective_function(params, template_coords, edge_image):
    """
    Score how well the transformed template matches the edges.
    
    Lower score = better match (we minimize this).
    """
    # Transform template points using current params
    transformed = transform_coords(template_coords, params)
    
    # Check how many points are inside the image
    valid_mask = points_inside_image(transformed)
    valid_ratio = sum(valid_mask) / len(template_coords)
    
    # Penalize heavily if too many points are outside
    if valid_ratio < 0.9:
        return 1e6 * (0.9 - valid_ratio)  # Bad transformation
    
    # Sample edge values at transformed points
    edge_responses = edge_image[transformed_y, transformed_x]
    
    # Weighted mean: points on a non-zero edge count 1.1, the others 1.0
    weights = where(edge_responses > 0, 1.1, 1.0)
    weighted_avg = sum(edge_responses * weights) / sum(weights)
    
    # Good alignment = high edge values = low cost
    penalty_factor = 1.0 - (valid_ratio - 0.9)
    return -(weighted_avg * valid_ratio) * (1 + penalty_factor)
```

**Key functions:** [`register_high_level`](#register_high_level), [`transform_coords`](#transform_coords), [`objective_function`](#objective_function), [`register_neck_coords_advanced`](#register_neck_coords_advanced)

---

#### Stage 4: Geometric Cropping (Extracting the ROI)

**The Problem:**
We now know where the ROI is (transformed rectangle coordinates). We need to extract it from the 3D stack.

**Two Sub-Problems:**

1. **Z-Cropping:** Extract a slab of the right thickness, following the curved surface
2. **XY-Cropping:** Extract the (possibly rotated) quadrilateral and warp it to a rectangle

**Z-Cropping: Following the Surface**

The height map from Stage 1 tells us where the surface is. We extract a slab centered on this surface:


```python
# z_range (µm) is converted to planes: z_range_px = round(z_range / ref_voxel_size[0])
# For each (y, x), we extract Z slices from:
#   grid[y, x] - z_range_px/2  to  grid[y, x] + z_range_px/2
# 
# This creates a "slab" that follows the curved surface. Where the slab would
# leave the stack, it is shifted back inside; if z_range exceeds the stack,
# the whole stack is kept (stacks of fewer than 32 planes raise an error).
```

**XY-Cropping: Warping the Quadrilateral**

The ROI is defined by 4 corners which may form a non-axis-aligned quadrilateral.


```python
from vascumap.pipeline.preprocessing import crop_stack

cropped = crop_stack(
    stack,                      # Input: (Z, Y, X, C)
    crop_profile='myprofile',   # Template name
    bf_index=0,                 # Which channel to use for registration
    mode='3D',                  # '3D' extracts a slab; '2D' extracts single slice
    z_range=200,                # Thickness in µm
    ref_voxel_size=[5, 2, 2],   # Voxel size for Z-range conversion
    lif_path_root='/output',    # Save debug images here
    filename='sample1',         # Prefix for debug images
    profile_dir='crop_profiles' # Folder holding the template files
)
```

The quadrilateral is mapped onto a rectangle by an affine transform estimated from its four corners and resampled with cubic interpolation (`order=3`), plane by plane.

**Key functions:** [`crop_stack`](#crop_stack), [`crop_stack_z`](#crop_stack_z), [`apply_affine_crop_simple`](#apply_affine_crop_simple)

---

#### Creating Custom Crop Profiles

To use the cropping pipeline with your own sample type, you need to create a template. The notebook `notebooks/new_crop_profile.ipynb` implements the steps below interactively, with the fiducials and the rectangle drawn as Shapes layers in napari (an optional dependency, listed as a comment in `requirements.txt`).

**Step 1: Get a Representative Image**

```python
from vascumap.pipeline.data import get_stack_from_lif
from vascumap.pipeline.preprocessing import auto_focus

# Load a typical sample at the reference voxel size (5 × 2 × 2 µm),
# the resolution at which crop_stack registers the template
stack, _, _ = get_stack_from_lif('reference.lif', stack_index=0)

# Get a focused 2D image
focused_image, _ = auto_focus(stack[..., 0])  # Use brightfield channel
```

**Step 2: Annotate Fiducial Points**

Using a tool like Napari, FIJI, or matplotlib, manually click points along the sample boundary:

```python
import matplotlib.pyplot as plt
import numpy as np

# Display the image
fig, ax = plt.subplots()
ax.imshow(focused_image, cmap='gray')

# Collect clicks
fiducials = []
def onclick(event):
    fiducials.append([event.ydata, event.xdata])  # Note: (y, x) order
    ax.plot(event.xdata, event.ydata, 'r.', markersize=5)
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# After clicking, convert to array
fiducials = np.array(fiducials)  # Shape: (N, 2) with columns [y, x]
```

**Tips for good fiducials:**
- Place points along strong edges (sample boundary)
- Space them evenly around the boundary
- Use 50-200 points for good coverage
- Avoid points on features that vary between samples

**Step 3: Define the ROI Rectangle**

Define the 4 corners of your region of interest (clockwise from top-left):

```python
# Corners in (y, x) format
rect = np.array([
    [y_top_left,     x_top_left],
    [y_top_right,    x_top_right],
    [y_bottom_right, x_bottom_right],
    [y_bottom_left,  x_bottom_left],
])
```

**Step 4: Save the Profile**

```python
np.save('crop_profiles/myprofile_fiducials.npy', fiducials)
np.save('crop_profiles/myprofile_rect.npy', rect)
```

**Step 5: Test the Profile**

```python
from vascumap.pipeline.preprocessing import crop_stack

# Test on a few samples
cropped = crop_stack(
    test_stack,
    crop_profile='myprofile',
    bf_index=0,
    mode='3D',
    z_range=200,
    lif_path_root='./debug',  # Check the debug images!
    filename='test',
    profile_dir='crop_profiles'
)
```

Check the saved `test_crop_overview.png` (written only when both `lif_path_root` and `filename` are given) to see if registration worked correctly.

**Common Issues and Solutions:**

| Problem | Symptom | Solution |
|---------|---------|----------|
| Template too large | Points go outside image | Scale down template coordinates (the registration keeps the scale at ~1.0) |
| Not enough fiducials | Poor alignment | Add more fiducial points |
| Fiducials on variable features | Inconsistent alignment | Move fiducials to stable boundaries |
| Optimization stuck | Bad crop location | Call `register_neck_coords_advanced` directly with closer `initial_params` |

---

### 2.2.3 Image Translation Inference

Run the Pix2Pix model to translate brightfield to fluorescence-like:

```python
from vascumap.pipeline.prediction import load_pix2pix, predict_pix2pix

model = load_pix2pix('translation.ckpt')

pseudo_fluo = predict_pix2pix(
    model,
    stack_bf,           # (Z, Y, X)
    device='cuda',
    n_iter=1            # Number of passes averaged
)
```

**Inference details:**
- Uses MONAI SlidingWindowInferer
- ROI size: 32×512×512, overlap: 75%/25%/25%, one window per batch
- Runs under automatic mixed precision (`torch.amp.autocast`)
- Output clipped to [0, 1]
- `load_pix2pix` returns the model in eval mode, so dropout is inactive and `n_iter > 1` averages identical passes

When the pipeline segments a fluorescence channel directly (`fluo_channel` or `fluo_index`), this step and the Pix2Pix checkpoint are skipped.

**Key functions:** [`load_pix2pix`](#load_pix2pix), [`predict_pix2pix`](#predict_pix2pix)

---

### 2.2.4 Segmentation Inference (2.5D)

Run the segmentation model using **2.5D inference** - predictions along three orthogonal planes, averaged:

```python
from vascumap.pipeline.prediction import load_segmentation_model, predict_mask_ortho

model = load_segmentation_model('segmentation.ckpt')   # Lightning .ckpt or .pth

probability_map = predict_mask_ortho(
    model,
    vessel_iso,         # Isotropic volume (Z, Y, X)
    device='cuda'
)
```

**2.5D approach:**
1. **Axial** (XY slices): Primary view, 256×256 tiles
2. **Coronal** (XZ slices): Side view, 256×256 tiles
3. **Sagittal** (YZ slices): Front view, 256×256 tiles
4. **Average**: Final probability = mean of all three

Each plane is tiled by MONAI `SliceInferer` with the training tile size (256×256), 50% overlap, Gaussian blending and batches of 32 tiles. For the coronal and sagittal passes, volumes with fewer than 256 planes in Z are padded with -1 to 256 planes, and the padding is removed from the predictions before averaging.

`load_segmentation_model` accepts a PyTorch Lightning checkpoint (`.ckpt`, whose weights are stored under the `model.` prefix), an exported `.pth` holding a `model_state_dict` entry, or a bare state dictionary. The weights are loaded non-strictly, and any missing or unexpected keys are reported instead of raising an error.

**Post-processing:**
```python
from vascumap.pipeline.utils import process_vessel_mask

binary_mask = process_vessel_mask(probability_map, ortho=True)
# Applies: median filter (size=7) + hysteresis threshold (0.2, 0.5)
```

With `ortho=False` (`--no_ortho`), only the axial plane is segmented ([`predict_mask`](#predict_mask), batches of 16 tiles) on the anisotropic 5 × 2 × 2 µm stack, after clipping the input to [0.05, 1] and rescaling it. The probability map is then thresholded by hysteresis (0.1, 0.5) without median filtering, and the mask is resampled to the isotropic grid.

**Key functions:** [`load_segmentation_model`](#load_segmentation_model), [`predict_mask_ortho`](#predict_mask_ortho), [`predict_mask`](#predict_mask), [`process_vessel_mask`](#process_vessel_mask)

---

### Full Pipeline Command

Brightfield input (Pix2Pix translation, then segmentation):

```bash
python -m vascumap.pipeline.main \
    --input_dir /path/to/lif/files \
    --model_p2p_path /path/to/translation.ckpt \
    --model_smp_path /path/to/segmentation.ckpt
```

Fluorescence input (for example the green fluorescent protein (GFP) channel of `.nd2` files, read with aicsimageio; Pix2Pix is skipped):

```bash
python -m vascumap.pipeline.main \
    --input_dir /path/to/nd2/files \
    --pattern "*.nd2" \
    --fluo_channel GFP \
    --model_smp_path /path/to/segmentation.ckpt
```

**All parameters explained:**

| Parameter | Default | What it Does |
|-----------|---------|--------------|
| `--input_dir` (alias `--lif_dir_path`) | Required | Folder with the microscopy files |
| `--pattern` | `*.lif` | Glob pattern of the files to process, e.g. `*.nd2` |
| `--model_p2p_path` | `None` | Pix2Pix checkpoint; required for brightfield input |
| `--model_smp_path` | Required | Segmentation checkpoint (`.ckpt` or `.pth`) |
| `--fluo_channel` | `None` | Name (case-insensitive substring) of a fluorescence vessel channel; skips Pix2Pix |
| `--fluo_index` | `None` | Index of a fluorescence vessel channel; skips Pix2Pix |
| `--bf_index` | `None` | Brightfield channel index; if omitted, the first channel whose name contains `BF` or `Gray` (case-insensitive), else channel 0 |
| `--no_crop` | off | Skip auto-focus and template cropping |
| `--crop_profile` | None | Name of the crop profile (required unless `--no_crop`) |
| `--crop_profile_dir` | `crop_profiles` | Folder holding the crop profiles |
| `--z_range` | `200` | Z-range (µm) kept around the focal surface when cropping |
| `--no_ortho` | off | Axial-only segmentation instead of 2.5D averaging |
| `--save_all_stacks` | off | Also save the intermediate stacks |

No crop profile is distributed with the repository (`crop_profiles/` is excluded by `.gitignore`). Cropping is on by default, so a profile created as described in [Creating Custom Crop Profiles](#creating-custom-crop-profiles) must be passed with `--crop_profile` (and `--crop_profile_dir`); otherwise `run()` stops with an error before any stack is processed, unless `--no_crop` is set. A stack whose cropping fails is skipped.

Without `--fluo_channel` or `--fluo_index`, LIF files are read with readlif, and every image with exactly three dimensions (X, Y and Z) is processed. With either option, every scene of every matching file is read with aicsimageio, and scenes without a matching channel or with a single Z-plane are skipped. In that case, the stack has a single channel, so auto-focus and template registration also run on the fluorescence image; `--no_crop` avoids this when the crop profile was drawn on brightfield images. Model inference runs on the GPU when CUDA is available and otherwise on the CPU, whereas the median filter of the 2.5D post-processing relies on CuPy and therefore always requires a GPU.

**Output files (per stack, written next to the input files):**

`<stack>` is the file name without extension followed by the LIF image name or the scene identifier, with spaces and slashes replaced by underscores. A stack whose `<stack>_vessel_mask.tif` already exists is skipped.

- `<stack>_vessel_mask.tif`: Binary segmentation (float32, isotropic 2 µm voxels)
- `<stack>_overview.png`: Z-sum projections of the fluorescence channel (or of the brightfield when the stack has no other channel), the probability map and the mask, side by side
- `<stack>_crop_overview.png`: Registration debug (if cropping)
- With `--save_all_stacks`, also on the isotropic grid: `<stack>_vessel_bf.tif` (the input channel, which is the fluorescence channel in fluorescence mode), `<stack>_vessel_pred.tif` (the volume passed to segmentation), `<stack>_vessel_proba.tif` (probability map) and `<stack>_vessel_fluo.tif` (the fluorescence channel; for multi-channel brightfield input, the first channel other than the brightfield)

**Key function:** [`run`](#run)

---

## 2.3 Graph Analysis

### 2.3.1 Skeletonization

Convert binary mask to 1-pixel-wide skeleton. Two methods available, both applied after the holes of the mask have been filled (`binary_fill_holes`, run on the GPU in chunks of Z × 512 × 512 voxels with a 128-voxel overlap; the mask is not smoothed beforehand):

#### Method 1: Morphological Thinning

Fast, iterative erosion preserving topology:

```python
from vascumap.pipeline.skeletonization import skeletonize_3d_parallel

skeleton = skeletonize_3d_parallel(
    binary_mask,
    chunk_size=(512, 512, 512),
    overlap=128          # voxels; must exceed the largest vessel radius
)
```

Chunks are skeletonized in parallel with Dask, each with a halo of `overlap` voxels on every side (clamped to the volume and chunk size along each axis), because thinning is not a local operation and too small an overlap corrupts thick vessels near chunk seams. [`construct_vessel_network`](#construct_vessel_network) uses chunks of Z × 512 × 512 voxels with the same 128-voxel overlap.

**Best for:** Dense networks (vessel volume fraction > 0.3, the threshold used by `skel_method="auto"`)

**Key function:** [`skeletonize_3d_parallel`](#skeletonize_3d_parallel)

#### Method 2: Mesh Contraction

GPU-accelerated Laplacian smoothing on mesh:

```python
from vascumap.pipeline.mesh_contraction import create_graph_contraction

graph, mesh, simple_mesh, contracted = create_graph_contraction(
    binary_edt,
    time_lim=300  # seconds
)
```

**Algorithm:**
1. Generate surface mesh (Marching Cubes) and simplify it (quadric decimation)
2. Iteratively contract toward centerline on the GPU ([`contract_gpu`](#contract_gpu))
3. Simplify and repair the contracted mesh
4. Extract skeleton from contracted mesh (PyGEL3D) and convert it to a NetworkX graph, removing terminal nodes attached to nodes of degree > 6

Within [`construct_vessel_network`](#construct_vessel_network), the graph nodes are then moved toward the vessel centerline along the gradient of the smoothed EDT ([`reposition_graph_edges`](#reposition_graph_edges), at most 10 voxels), and [`finalize_graph`](#finalize_graph) merges degree-2 nodes and rasterizes the edges. The pruning steps of Section 2.3.2 apply to the thinning method only, whereas border trimming applies to both methods.

**Best for:** Sparse networks, varying vessel diameters

**Key functions:** [`create_graph_contraction`](#create_graph_contraction), [`contract_gpu`](#contract_gpu)

---

### 2.3.2 Graph Construction

Convert skeleton to NetworkX graph:

```python
import sknw

# Build graph from skeleton (as called by construct_vessel_network)
graph = sknw.build_sknw(skeleton_image, multi=False, iso=False, ring=False)

# Nodes have 'pts' attribute (coordinates)
# Edges have 'pts' attribute (path coordinates) and 'weight' (path length in voxels)
```

Each node keeps its full voxel set in `pts_all`, while `pts` is reduced to one representative voxel, and every node of degree 1 is flagged as a sprout tip (`sprout=True`). This flag is not updated afterwards, so a junction that loses branches during pruning or border trimming is not relabeled as a sprout tip.

**Post-processing** (thinning method, in this order):
1. **Tip/origin area pruning** ([`prune_graph`](#prune_graph)): Remove terminal branches that do not taper (tip/origin cross-sectional area ratio > 0.20) or that are at most 50 µm long
2. **Self-loop removal**: Remove self-loops shorter than 40 µm
3. **Spur removal**: Repeatedly remove terminal branches shorter than 80 µm
4. **Small-component removal**: Remove connected components with a total length below 20 µm
5. **Merge degree-2 nodes**: Join the two edges of every degree-2 node into one continuous edge (run even without pruning)
6. **Border trimming**: Remove every edge with a point closer than 100 µm (XY) or 2 µm (Z) to the volume border, then remove the isolated nodes

Steps 1 to 4 run only when `do_prune_graph=True` (the default). Every threshold is given in µm and converted to voxels with `voxel_size_um`; at the default 2 µm, the length thresholds of steps 1 to 4 correspond to 25, 20, 40 and 10 voxels, respectively, and the border margins of step 6 to 50 voxels in XY and 1 voxel in Z.

**How the tip/origin area ratio works:** Thinning behaves well on tubular segments, but thick, blob-like regions produce many spurious terminal branches. For each terminal branch, `prune_graph` resamples the cross-sectional area profile along the branch to 100 points and takes the thinner half as the tip side. The ratio between the mean area over the 20% of the branch nearest the tip and the area at the origin end measures the taper: a genuine sprout narrows toward its tip (small ratio), whereas a false branch inside a blob remains as thick at its tip as at its origin. The procedure is repeated until no terminal branch is removed.

| Parameter of `construct_vessel_network` | Default | Step or effect |
|------------------------------------------|---------|----------------|
| `tip_area_ratio_cutoff` | 0.20 | 1 |
| `prune_length_cutoff_um` | 50 µm | 1 |
| `min_loop_um` | 40 µm | 2 |
| `min_spur_um` | 80 µm | 3 |
| `min_component_um` | 20 µm | 4 |
| `vicinity_xy_um` | 100 µm | 6 |
| `vicinity_z_um` | 2 µm | 6 |
| `keep_rings` | `False` | Keep closed rings without a node (sknw `ring=True`) |
| `keep_isolated` | `False` | Keep degree-0 nodes, the only skeleton of compact, near-spherical objects |

The individual steps can also be called directly, with thresholds in voxels:

```python
from vascumap.pipeline.skeletonization import prune_graph, collect_border_vicinity_edges

# 2 µm voxels: 25 voxels = 50 µm, 50 voxels = 100 µm
pruned = prune_graph(graph, area_3d, tip_area_ratio_cutoff=0.20, length_cutoff=25)
trimmed = collect_border_vicinity_edges(pruned, binary_mask.shape, vicinity_z=1, vicinity_xy=50)
```

In practice, [`construct_vessel_network`](#construct_vessel_network) runs the whole sequence, from hole filling to border trimming:

```python
import pandas as pd
from vascumap.pipeline.graph_main import construct_vessel_network

global_df = pd.DataFrame({'sample': ['sample1']})
graph, area_image, area_image_exact, filled_mask = construct_vessel_network(
    binary_mask,
    global_df,               # receives chip_volume, vessel_volume, vessel_volume_fraction
    skel_method="thinning",
    voxel_size_um=2.0,
)
```

**Key functions:** [`construct_vessel_network`](#construct_vessel_network), [`prune_graph`](#prune_graph), [`collect_border_vicinity_edges`](#collect_border_vicinity_edges), [`finalize_graph`](#finalize_graph) (contraction method)

---

### 2.3.3 Metrics Computation

Compute morphometric features from the graph:

```python
import pandas as pd
from vascumap.pipeline.metrics import (
    compute_cross_sectional_areas,
    compute_cross_sectional_areas_exact,
    compute_vessel_metrics,
    compute_junction_metrics,
    fractal_dimension_and_lacunarity,
    convex_hull_volume
)
from vascumap.pipeline.visualization import graph2image

# Cross-sectional areas at skeleton points (voxel², fast ellipse approximation)
area_3d = compute_cross_sectional_areas(mask, skeleton, edt)

# Local vessel volume per skeleton voxel (voxel³, nearest-centerline partition on the GPU)
area_3d_exact = compute_cross_sectional_areas_exact(mask, skeleton, overlap=128)

# Per-vessel metrics (µm, µm², µm³)
vessel_df = compute_vessel_metrics(graph, area_3d, pd.DataFrame(),
                                   area_image_exact=area_3d_exact, voxel_size_um=2.0)

# Per-junction metrics (distances in µm; neighbors counted within 500 µm)
junction_df = compute_junction_metrics(graph, pd.DataFrame(),
                                       distance_threshold=500, voxel_size_um=2.0)

# Global metrics on the final skeleton, rasterized from the graph
fd, lacunarity = fractal_dimension_and_lacunarity(graph2image(graph, mask.shape) > 0)
chip_volume_um3 = convex_hull_volume(mask) * 2.0 ** 3
```

**Two cross-sectional area estimates:** [`compute_cross_sectional_areas`](#compute_cross_sectional_areas) models each cross-section as an ellipse whose semi-axes are the 3D EDT and the EDT of the Z-maximum projection of the mask. This estimate is fast, but vessels that overlap in the projection are merged, so the area is overestimated in dense regions. [`compute_cross_sectional_areas_exact`](#compute_cross_sectional_areas_exact) avoids the projection: a Euclidean feature transform assigns every foreground voxel, in full 3D, to its nearest skeleton voxel (a Voronoi partition of the vessel around its centerline), so that the number of voxels assigned to a skeleton voxel equals the vessel volume carried by that centerline point. Dividing this local volume by the local centerline step yields the perpendicular cross-sectional area, and summing the local volumes along an edge yields the vessel volume. The partition is computed in chunks on the GPU, with an overlap that must exceed the largest vessel radius; `construct_vessel_network(compute_exact_area=False)` or `--no_exact_area` skips it.

In practice, [`calculate_graph_metrics`](#calculate_graph_metrics) runs these functions on the output of `construct_vessel_network` and adds the global columns listed in [Section 4.1](#41-global-metrics-csv).

**Key functions:** [`calculate_graph_metrics`](#calculate_graph_metrics), [`compute_cross_sectional_areas`](#compute_cross_sectional_areas), [`compute_cross_sectional_areas_exact`](#compute_cross_sectional_areas_exact), [`compute_vessel_metrics`](#compute_vessel_metrics), [`compute_junction_metrics`](#compute_junction_metrics), [`fractal_dimension_and_lacunarity`](#fractal_dimension_and_lacunarity), [`convex_hull_volume`](#convex_hull_volume)

---

### 2.3.4 Output Files

Graph analysis writes its output files into the mask folder:

| File | Format | Content |
|------|--------|---------|
| `global_metrics_{method}.csv` | CSV | One row per sample, aggregate statistics |
| `{sample}_vessel_metrics_{method}.h5` | HDF5 | One row per vessel segment (with `--save_local_metrics`) |
| `{sample}_junction_metrics_{method}.h5` | HDF5 | One row per node (with `--save_local_metrics`) |
| `{sample}_graph_{method}.png` | PNG | Graph drawn over the Z-sum projection of the mask (with `--graph_visualization`) |

`{sample}` is the mask file name without extension (for example `stack1_vessel_mask`), and `{method}` is the skeletonization method. Masks are found with the pattern `*{suffix}.tif`. The global CSV is rewritten after every mask, so that it holds all masks processed so far; a mask without foreground contributes a row with its name only, and a mask that raises an error is reported and skipped.

In the graph figure, each edge is colored by the median equivalent diameter of its cross-sections, d = 2√(A/π), in µm (magma color map, from half the minimum diameter to the 90th percentile), and the nodes are colored by type: sprout tips in lime, other free ends in orange and junctions in white.

#### Reading Output Files

```python
import pandas as pd

# Global metrics
global_df = pd.read_csv('global_metrics_thinning.csv')

# Vessel metrics (requires: pip install tables)
vessel_df = pd.read_hdf('stack1_vessel_mask_vessel_metrics_thinning.h5', key='df')

# Junction metrics
junction_df = pd.read_hdf('stack1_vessel_mask_junction_metrics_thinning.h5', key='df')
```

#### Full Pipeline Command

```bash
python -m vascumap.pipeline.graph_main \
    --masks_dir_path /path/to/masks \
    --file_suffix _vessel_mask \
    --method thinning \
    --voxel_size_um 2.0 \
    --save_local_metrics \
    --graph_visualization
```

**All parameters explained:**

| Parameter | Default | What it Does |
|-----------|---------|--------------|
| `--masks_dir_path` | Required | Folder with the mask files; the outputs are written there |
| `--file_suffix` | `_mask` | Masks are the files matching `*{suffix}.tif` |
| `--method` | `thinning` | Skeletonization: `thinning` or `contraction` |
| `--voxel_size_um` | `2.0` | Isotropic voxel size of the masks in µm; converts every threshold and metric to physical units |
| `--no_exact_area` | off | Skip the nearest-centerline (exact) cross-sectional areas |
| `--save_local_metrics` | off | Save the per-vessel and per-junction HDF5 tables |
| `--graph_visualization` | off | Save the graph figure |

Masks written by the inference pipeline are isotropic 2 µm grids, so the default voxel size applies to them; masks on another grid require `--voxel_size_um`, because the pruning thresholds are also defined in µm.

**Key function:** [`process_directory`](#process_directory)

---

# 3. Function Reference

This section documents all public functions organized by module. Each entry includes:
- **Purpose**: What the function does
- **Arguments**: Input parameters with types
- **Returns**: Output values
- **Source**: Link to source file

---

## 3.1 Image Translation Module

**Location:** [`vascumap/models/image_translation/`](vascumap/models/image_translation/)

### `ImageDataset3D`

PyTorch Dataset for paired 3D NIfTI volumes.

**Source:** [`utils.py`](vascumap/models/image_translation/utils.py)

```python
dataset = ImageDataset3D(
    input_paths: List[str],      # Paths to input volumes
    target_paths: List[str],     # Paths to target volumes (optional)
    split: str = 'train',        # 'train', 'val', or 'test'
    transform: callable = None   # Custom TorchIO transform
)
```

**Returns:** `tuple(input_tensor, target_tensor)` or `input_tensor`

Without a custom transform, the volumes are converted to float tensors, and training pairs are additionally flipped at random along the TorchIO `LR` and `AP` axes (the same flip for input and target).

---

### `Generator`

3D U-Net generator based on MONAI.

**Source:** [`utils.py`](vascumap/models/image_translation/utils.py)

```python
generator = Generator(
    dropout_p: float = 0.4    # Dropout probability
)
```

**Architecture:**
- Channels: (32, 64, 128, 256, 512)
- Strides: (1, 2, 2, 2, 1)
- Residual units: 3 per level
- Output: one channel, passed through a ReLU

---

### `Discriminator`

3D PatchGAN discriminator.

**Source:** [`utils.py`](vascumap/models/image_translation/utils.py)

```python
discriminator = Discriminator(
    dropout_p: float = 0.4    # Stored only: the layers contain no dropout
)
```

**Input:** Concatenated input+target (2 channels)

**Architecture:** Five 3D convolutions (kernel 4; 64, 128, 256, 512 and 1 channels), the first four followed by batch normalization and LeakyReLU (0.2) and the last by a sigmoid that yields one real/fake probability per patch.

---

### `Pix2Pix`

PyTorch Lightning module for GAN training.

**Source:** [`utils.py`](vascumap/models/image_translation/utils.py)

```python
model = Pix2Pix(
    generator_dropout_p: float = 0.4,
    discriminator_dropout_p: float = 0.4,
    generator_lr: float = 1e-3,
    discriminator_lr: float = 1e-6,
    weight_decay: float = 1e-5,
    lr_scheduler_T_0: float = 5e3,
    lr_scheduler_T_mult: float = 2
)
```

**Loss functions:**
- Generator: `BCE + λ×L1 + λ×MSE` (λ=100)
- Discriminator: `BCE`

**Training config:**
- Optimizers: Adam for each network, with `weight_decay`
- Schedulers: `CosineAnnealingWarmRestarts` (`T_0`, `T_mult`), stepped after every batch
- Manual optimization: every training batch is split into chunks of three volumes (`TRAIN_BATCH_SIZE=6`); the first chunk updates the discriminator and the second the generator, so batches must hold 4 to 6 volumes
- Validation metrics: `val_g_psnr`, `val_g_ssim`, `val_g_accuracy`, `val_d_accuracy`

---

### `train` (image translation)

Train the Pix2Pix model from two folders of paired NIfTI volumes; called by the command line of `train.py`.

**Source:** [`train.py`](vascumap/models/image_translation/train.py)

```python
train(
    input_path: str,                 # Folder of input volumes (.nii.gz)
    target_path: str,                # Folder of target volumes (.nii.gz)
    model_path: str,                 # Checkpoint and log folder
    batch_size: int = 4,
    epochs: int = 200,
    generator_lr: float = 1e-3,
    discriminator_lr: float = 1e-5,
    generator_dropout: float = 0.1,
    discriminator_dropout: float = 0.25,
    val_split: float = 0.1,
    num_workers: int = 0,
    device: str = 'cuda'             # Used for the architecture summaries
)
```

Sets `lr_scheduler_T_0` to two epochs of batches, keeps the five checkpoints with the highest `val_g_ssim` and saves the final weights to `last_epoch_full.pth`.

---

## 3.2 Segmentation Module

**Location:** [`vascumap/models/segmentation/`](vascumap/models/segmentation/)

### `SegmentationModule`

PyTorch Lightning module for 2D segmentation.

**Source:** [`model.py`](vascumap/models/segmentation/model.py)

```python
model = SegmentationModule(
    model_name: str,             # 'Unet', 'FPN', 'DeepLabV3+', etc.
    encoder_name: str,           # 'mit_b5', 'resnet50', etc.
    in_channels: int,            # Usually 1 for grayscale
    encoder_weights: str = 'imagenet',
    learning_rate: float = 1e-3,
    max_epochs: int = 200        # For scheduler calculation
)
```

**Training config:**
- Loss: Dice + BCE
- Optimizer: RAdam (weight_decay=1e-4)
- Scheduler: StepLR (halves at max_epochs/2)
- Metrics: `BinaryF1Score` (logged as Dice, `train_dice`/`val_dice`) and binary `JaccardIndex` (`train_iou`/`val_iou`)

---

### `train` (segmentation)

Train a segmentation model; called by the command line of `training.py`.

**Source:** [`training.py`](vascumap/models/segmentation/training.py)

```python
train(
    images_path: str,
    masks_path: str,
    model_path: str,                 # Checkpoint and log folder
    model_str: str,                  # e.g. 'Unet'
    encoder_str: str,                # e.g. 'mit_b5'
    weights: str,                    # e.g. 'imagenet'
    in_channels: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    fp16: bool                       # precision="16-mixed" if True, else 32
)
```

Keeps the three checkpoints with the lowest `val_loss` and logs the learning rate once per epoch.

---

### `build_model`

Factory function for segmentation models.

**Source:** [`model_utils.py`](vascumap/models/segmentation/model_utils.py)

```python
model = build_model(
    model_str: str,              # Architecture name
    encoder_str: str,            # Encoder backbone
    in_channels: int = 1,
    encoder_weights: str = None  # 'imagenet' or None
)
```

**Supported architectures:**
`Unet`, `Unet++`, `MAnet`, `Linknet`, `FPN`, `PSPNet`, `PAN`, `DeepLabV3`, `DeepLabV3+`

For an MiT encoder with `in_channels=1`, the model is built with three input channels and then passed to [`adapt_input_model`](#adapt_input_model); every model has one output class.

**Returns:** `torch.nn.Module`

---

### `adapt_input_model`

Adapts pretrained RGB weights for grayscale input.

**Source:** [`model_utils.py`](vascumap/models/segmentation/model_utils.py)

```python
model = adapt_input_model(
    model: nn.Module    # Model with 3-channel first layer
)
```

**Returns:** `nn.Module` with 1-channel first layer

Only models whose encoder has an MiT patch embedding (`encoder.patch_embed1`) are changed; any other model is returned unchanged. The replacement layer is fixed at 64 output channels, the patch-embedding width of `mit_b1` to `mit_b5`.

---

### `load_data`

Create train/val data loaders.

**Source:** [`dataset.py`](vascumap/models/segmentation/dataset.py)

```python
loaders = load_data(
    images_path_str: str,        # Directory of images
    masks_path_str: str = None,  # Directory of masks (None for inference)
    batch_size: int = 16,
    seed: int = 0,
    test_size: float = 0.1,
    format: str = 'tif'
)
```

**Returns:** `OrderedDict` with 'train' and 'valid' DataLoaders when a mask folder is given, otherwise a single inference DataLoader

Training batches are shuffled and incomplete training batches are dropped; all loaders use `num_workers=0`. The training set receives [`hard_transforms`](#hard_transforms) followed by tensor conversion, and the validation set receives tensor conversion only.

---

### `hard_transforms`

Training augmentations (albumentations ≥ 2.0).

**Source:** [`transforms.py`](vascumap/models/segmentation/transforms.py)

```python
transforms = hard_transforms()
```

**Returns:** `list` of `HorizontalFlip(p=0.5)`, `VerticalFlip(p=0.5)` and `Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-45, 45), border_mode=cv2.BORDER_REFLECT_101, p=0.5)`, combined with `post_transforms()` (`ToTensorV2`) by `compose()`

---

### `SegmentationDataset`

PyTorch Dataset for 2D image-mask pairs.

**Source:** [`dataset.py`](vascumap/models/segmentation/dataset.py)

```python
dataset = SegmentationDataset(
    images: List[Path],          # Image file paths
    masks: List[Path] = None,    # Mask file paths
    transforms: callable = None  # Albumentations composition
)
```

**Returns:** `dict` with 'image', 'mask', 'filename'

Each image is clipped to its 0.1st–99.9th percentiles, rescaled to [0, 1], cast to float32 and given a trailing channel axis; each mask is binarized (values > 0 become 1, while 0 and NaN become 0) before the transforms are applied.

---

## 3.3 Pipeline Module

**Location:** [`vascumap/pipeline/`](vascumap/pipeline/)

### Data Loading

#### `get_stack_from_lif`

Extract and rescale 3D stack from LIF file.

**Source:** [`data.py`](vascumap/pipeline/data.py)

```python
stack, voxel_size, [nz, ny, nx] = get_stack_from_lif(
    lif_file: str,                       # Path to LIF file
    stack_index: int = 0,                # Image index in LIF
    ref_voxel_size: list = [5, 2, 2]     # Target voxel size [Z, Y, X] µm
)
```

**Returns:**
- `stack`: `np.ndarray` (Z, Y, X, C)
- `voxel_size`: `np.ndarray` [Z, Y, X] original
- `[nz, ny, nx]`: Original dimensions

Raises a `ValueError` when the stack spans less than 160 µm in Z.

---

#### `get_stack_from_aics`

Extract one channel of an aicsimageio scene as a 3D stack and rescale it; used for fluorescence input in any format aicsimageio can read (e.g. `.nd2`).

**Source:** [`data.py`](vascumap/pipeline/data.py)

```python
stack, voxel_size, [nz, ny, nx] = get_stack_from_aics(
    img: AICSImage,                      # Scene already selected with img.set_scene(...)
    channel_index: int,                  # Channel to extract (e.g. GFP)
    ref_voxel_size: list = [5, 2, 2]     # Target voxel size [Z, Y, X] µm
)
```

**Returns:**
- `stack`: `np.ndarray` (Z, Y, X, 1), float32 before resampling
- `voxel_size`: `np.ndarray` [Z, Y, X] original, in µm
- `[nz, ny, nx]`: Original dimensions

Reads the first time point when the file has a time axis. Raises a `ValueError` when the physical pixel sizes are missing or when the stack spans less than 160 µm in Z.

---

### Preprocessing

#### `auto_focus`

Find optimal focal plane across XY.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
focus, grid = auto_focus(
    stack_bf: np.ndarray,        # 3D stack (Z, Y, X)
    crop_window: int = 50,       # Border margin
    n_sampling: int = 20,        # Grid density
    window_size: int = 50,       # Focus patch size
    viz: bool = False,           # Show visualization
    stack_chan: np.ndarray = None  # Optional extra stack(s) sampled on the same surface
)
```

**Returns:**
- `focus`: 2D focused image (Y, X)
- `grid`: Height map of Z-indices (Y, X)

When `stack_chan` is given, the function returns `(focus, focused_channels, grid)` instead.

---

#### `crop_stack`

Complete cropping pipeline.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
cropped = crop_stack(
    stack: np.ndarray,           # Input (Z, Y, X, C)
    crop_profile: str,           # Profile name
    bf_index: int,               # Brightfield channel
    mode: str = '3D',            # '2D' or '3D'
    z_range: float = 200,        # Z-thickness in µm
    ref_voxel_size: list = [5, 2, 2],
    lif_path_root: str = None,   # Save debug images here
    filename: str = None,
    profile_dir: str = 'crop_profiles'  # Folder holding the profile files
)
```

**Returns:** `np.ndarray` cropped stack

Loads `<crop_profile>_rect.npy` and `<crop_profile>_fiducials.npy` from `profile_dir` and raises a `FileNotFoundError` when either is missing. The debug figure `<filename>_crop_overview.png` is written to `lif_path_root` only when both arguments are given.

---

#### `crop_stack_z`

Crop Z-range around focal surface.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
cropped = crop_stack_z(
    stack: np.ndarray,           # Full stack
    grid: np.ndarray,            # Height map from auto_focus
    z_range: float,              # Thickness in µm
    ref_voxel_size: tuple        # Voxel size [Z, Y, X]
)
```

**Returns:** `np.ndarray` Z-cropped stack

---

#### `apply_affine_crop_simple`

Warp quadrilateral ROI to rectangle.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
warped = apply_affine_crop_simple(
    image: np.ndarray,           # 3D (Z,Y,X,C) or 2D (Y,X,C)
    registered_rect: np.ndarray, # 4×2 corners (y, x)
    order: int = 1               # Interpolation order
)
```

**Returns:** `np.ndarray` warped image

---

#### `transform_coords`

Apply affine transformation to coordinates.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
transformed = transform_coords(
    coords: np.ndarray,          # (N, 2) coordinates
    params: list                 # [scale, theta, tx, ty]
)
```

**Returns:** `np.ndarray` transformed coordinates, `scale × (coords @ R(theta)) + [tx, ty]`

---

#### `register_high_level`

High-level registration wrapper.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
registered_coords, optimal_params, best_score = register_high_level(
    edges: np.ndarray,           # Edge map
    neck_coords_all: np.ndarray  # Template coordinates
)
```

**Returns:** `tuple` (transformed_coords, params, score)

Initializes the parameters with scale 1, no rotation and a translation that centers the template on the edge map, then calls [`register_neck_coords_advanced`](#register_neck_coords_advanced) with the Powell method.

---

#### `register_neck_coords_advanced`

Global and local optimization of the template transformation.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
transformed_coords, optimal_params, best_score = register_neck_coords_advanced(
    neck_coords: np.ndarray,     # Template fiducials (N, 2), (y, x)
    canny_image: np.ndarray,     # Edge map
    initial_params: list,        # [scale, theta, tx, ty]
    method: str = 'Powell'       # Local minimizer
)
```

**Returns:** `tuple` (transformed_coords, params, score), or `(None, None, None)` if every optimization fails

Runs basin-hopping (50 hops) from three starting points, refines each result with the local minimizer and keeps the lowest score. Bounds: scale 0.999–1.001, rotation ±1.5 rad, translations ±500 pixels.

---

#### `objective_function`

Registration cost of a set of transformation parameters.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
score = objective_function(
    params: list,                # [scale, theta, tx, ty]
    coords: np.ndarray,          # Template fiducials (N, 2)
    canny_image: np.ndarray      # Edge map
)
```

**Returns:** `float`, the negative weighted mean edge response at the transformed fiducials (lower is better), or a penalty of `1e6 × (0.9 − valid_ratio)` when fewer than 90% of the fiducials fall inside the image

---

### Prediction

#### `load_pix2pix`

Load trained Pix2Pix model.

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
model = load_pix2pix(
    model_path: str,             # Lightning checkpoint (.ckpt)
    device: str = 'cuda'         # Accepted but unused: the model is loaded on the CPU
)
```

**Returns:** `Pix2Pix` in eval mode

Loads the `state_dict` entry of the checkpoint strictly, so a bare state dictionary such as `last_epoch_full.pth` is not accepted.

---

#### `load_segmentation_model`

Load trained segmentation model.

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
model = load_segmentation_model(
    model_path: str,             # Checkpoint path (.pth or .ckpt)
    device: str = 'cuda'         # Accepted but unused: the model is loaded on the CPU
)
```

**Returns:** `nn.Module` in eval mode

Builds a `Unet` with an `mit_b5` encoder adapted to one input channel, then loads, non-strictly, the weights of a Lightning `.ckpt` (keys under the `model.` prefix), of a `.pth` with a `model_state_dict` entry or of a bare state dictionary. Missing or unexpected keys are printed instead of raising an error.

---

#### `predict_pix2pix`

Run translation inference with sliding window.

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
prediction = predict_pix2pix(
    model_p2p: nn.Module,
    stack_bf: np.ndarray,        # (Z, Y, X) brightfield
    device: str,
    n_iter: int = 1              # Number of passes averaged
)
```

**Returns:** `np.ndarray` pseudo-fluorescence (Z, Y, X), clipped to [0, 1]

Sliding-window inference with ROI 32 × 512 × 512, overlap 0.75/0.25/0.25 and one window per batch, under automatic mixed precision. In eval mode, the `n_iter` passes are identical.

---

#### `predict_mask`

Run 2D segmentation (axial slices only).

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
proba = predict_mask(
    model_smp: nn.Module,
    vessel_pred: np.ndarray,     # (Z, Y, X)
    device: str
)
```

**Returns:** `np.ndarray` probability map (Z, Y, X)

Axial `SliceInferer` with 256 × 256 tiles, 50% overlap, Gaussian blending and `sw_batch_size=16`.

---

#### `predict_mask_ortho`

Run 2.5D segmentation (3-plane averaging).

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
proba = predict_mask_ortho(
    model_smp: nn.Module,
    vessel_pred_iso: np.ndarray, # Isotropic volume
    device: str
)
```

**Returns:** `np.ndarray` averaged probability map

Axial, coronal and sagittal `SliceInferer` passes with 256 × 256 tiles, 50% overlap, Gaussian blending and `sw_batch_size=32`. For the coronal and sagittal passes, volumes with fewer than 256 Z-planes are padded with -1 to 256 planes, and the padding is cropped from the predictions.

---

### Main Pipeline

#### `run`

Run full inference pipeline.

**Source:** [`main.py`](vascumap/pipeline/main.py)

```python
run(
    input_dir: str,                       # Directory with the microscopy files
    pattern: str = '*.lif',               # Glob pattern of the files to process
    ortho: bool = True,                   # Use 2.5D inference
    bf_index: int = None,                 # Brightfield channel (found by name if None)
    fluo_channel: str = None,             # Fluorescence channel name, e.g. 'GFP' (skips translation)
    fluo_index: int = None,               # Fluorescence channel index (skips translation)
    crop_profile: str = None,
    crop_profile_dir: str = 'crop_profiles',
    z_range: int = 200,                   # µm kept around the focal surface
    save_all_stacks: bool = False,
    crop: bool = True,
    model_p2p_path: str = None,           # Required for brightfield input
    model_smp_path: str = None            # Required
)
```

Raises a `ValueError` when `model_smp_path` is missing, or when `model_p2p_path` is missing for brightfield input. Each stack is loaded, optionally cropped, translated (brightfield input only), segmented and saved as described in [Full Pipeline Command](#full-pipeline-command); an error in one stack is printed and the next stack is processed.

---

## 3.4 Graph Analysis Module

### Skeletonization

#### `skeletonize_3d_parallel`

Parallel thinning using Dask.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
skeleton = skeletonize_3d_parallel(
    binary_volume: np.ndarray,
    chunk_size: tuple = (512, 512, 512),
    overlap: int | tuple = 128,  # Halo in voxels; scalar or per-axis (z, y, x)
    iter: int = 1
)
```

**Returns:** `np.ndarray` skeleton image

The overlap is clamped per axis so that it never exceeds the volume or the chunk size, which keeps the XY overlap at 128 voxels for thin-Z volumes.

---

#### `prune_graph`

Remove spurious branches.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
pruned = prune_graph(
    graph: nx.Graph,
    area_3d: np.ndarray,                 # Cross-sectional areas (voxel²) at skeleton voxels
    tip_area_ratio_cutoff: float = 0.25, # Maximum tip/origin area ratio of a kept sprout
    length_cutoff: int = 25              # Terminal branches up to this length (voxels) are removed
)
```

**Returns:** `nx.Graph` pruned

A terminal branch is removed when its tip/origin cross-sectional area ratio exceeds `tip_area_ratio_cutoff` (it does not taper) or when its length is at most `length_cutoff`; tapering sprouts of sufficient length are kept. The pass is repeated until no terminal branch is removed. [`construct_vessel_network`](#construct_vessel_network) calls it with a cutoff of 0.20 and a length of 50 µm.

---

#### `finalize_graph`

Clean graph: merge degree-2 nodes, clip coordinates. Used by the mesh contraction method; it also flags degree-1 nodes as sprout tips, removes edges whose two nodes both have degree 1 and rasterizes every edge into a continuous voxel path.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
final = finalize_graph(
    G_repositioned: nx.Graph,
    binary_edt: np.ndarray
)
```

**Returns:** `nx.Graph` cleaned

---

#### `collect_border_vicinity_edges`

Remove edges near image borders.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
trimmed = collect_border_vicinity_edges(
    graph: nx.Graph,
    image_shape: tuple,
    vicinity_z: int = 1,         # Z-margin in voxels
    vicinity_xy: int = 50,       # XY-margin in voxels
    keep_isolated: bool = False  # Keep degree-0 nodes
)
```

**Returns:** `nx.Graph` (a copy) with border edges removed

An edge is removed when any of its points lies within the margins. The nodes left with degree 0 are then removed, unless `keep_isolated=True`, in which case they are kept as the only skeleton of compact, near-spherical objects.

---

#### `measure_edge_length`

Compute path length of edge coordinates.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
length = measure_edge_length(
    coordinates: np.ndarray      # (N, 3) points
)
```

**Returns:** `float` total path length

---

### Mesh Contraction

#### `create_graph_contraction`

Full mesh contraction pipeline.

**Source:** [`mesh_contraction.py`](vascumap/pipeline/mesh_contraction.py)

```python
graph, mesh, simple_mesh, contracted = create_graph_contraction(
    binary_edt: np.ndarray,
    time_lim: int = 300          # Seconds
)
```

**Returns:** `tuple` (graph, original_mesh, simplified_mesh, contracted_mesh)

Calls [`contract_gpu`](#contract_gpu) with `precision=1e-9`, `SL=4` and `WH0=2`.

---

#### `contract_gpu`

GPU-accelerated Laplacian mesh contraction.

**Source:** [`mesh_contraction.py`](vascumap/pipeline/mesh_contraction.py)

```python
contracted = contract_gpu(
    mesh: trimesh.Trimesh,
    epsilon: float = 1e-6,       # Convergence threshold
    iter_lim: int = 100,
    time_lim: int = None,        # Seconds
    precision: float = 1e-7,     # Solver tolerance (LSMR)
    SL: float = 2,               # Contraction weight multiplier
    WH0: float = 1,              # Initial attraction weight
    WL0: str | float = 'auto',   # Initial contraction weight ('auto' from mesh statistics)
    operator: str = 'cotangent', # 'cotangent' or 'umbrella'
    progress: bool = True,
    validate: bool = True
)
```

**Returns:** `trimesh.Trimesh` contracted

---

#### `reposition_graph_edges`

Move nodes toward vessel centerline using EDT gradient.

**Source:** [`mesh_contraction.py`](vascumap/pipeline/mesh_contraction.py)

```python
repositioned = reposition_graph_edges(
    graph: nx.Graph,
    binary_edt: np.ndarray,
    min_segment_length: float = 5.0,
    max_disp: float = 12.5,
    step_size: float = 1.0,
    num_iterations: int = 1000,
    tol: float = 1e-3            # Gradient norm at which a node stops
)
```

**Returns:** `nx.Graph` repositioned

[`construct_vessel_network`](#construct_vessel_network) calls it with `min_segment_length=10`, `max_disp=10`, `step_size=1`, `num_iterations=1000` and `tol=1e-20`.

---

### Metrics

#### `compute_cross_sectional_areas`

Compute vessel area at skeleton points.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
area_3d = compute_cross_sectional_areas(
    mask: np.ndarray,            # Binary mask
    skeleton_image: np.ndarray,  # Skeleton
    binary_edt: np.ndarray       # EDT
)
```

**Formula:** `Area = π × EDT_2D(y,x) × EDT_3D(z,y,x)`

**Returns:** `np.ndarray` with areas (voxel²) at skeleton points

Fast approximation: the 2D EDT is computed on the Z-maximum projection of the mask, so vessels that overlap in projection are merged and the area is overestimated in dense regions.

---

#### `compute_cross_sectional_areas_exact`

Compute the local vessel volume carried by each skeleton voxel (nearest-centerline partition).

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
local_volume = compute_cross_sectional_areas_exact(
    mask: np.ndarray,            # Binary mask (foreground > 0)
    skeleton_image: np.ndarray,  # Skeleton
    chunk_size: tuple = None,    # Defaults to (Z, 512, 512)
    overlap: int | tuple = 128   # Halo in voxels; scalar or per-axis (z, y, x)
)
```

**Returns:** `np.ndarray` volume (voxel³) carried by each skeleton voxel, zero elsewhere

Every foreground voxel is assigned in 3D to its nearest skeleton voxel with a Euclidean feature transform computed in chunks on the GPU (CuPy). The overlap must exceed the largest vessel radius, so that the nearest centerline of a foreground voxel is never in an unseen neighboring chunk; it is clamped per axis to the volume and chunk size. [`compute_vessel_metrics`](#compute_vessel_metrics) converts the local volumes into areas by dividing them by the local centerline step.

---

#### `compute_vessel_metrics`

Compute per-edge metrics.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
vessel_df = compute_vessel_metrics(
    graph: nx.Graph,
    area_image: np.ndarray,              # Cross-sectional areas (voxel²)
    vessel_metrics_df: pd.DataFrame,     # Unused; a new DataFrame is returned
    area_image_exact: np.ndarray = None, # Local volumes from compute_cross_sectional_areas_exact
    voxel_size_um: float = 2.0           # Isotropic voxel size for unit conversion
)
```

**Returns:** `pd.DataFrame` indexed by the edge end nodes (`node1`, `node2`), with columns: `z, y, x, vessel_volume, vessel_length, end_to_end_distance, tortuosity, is_sprout, mean_cross_sectional_area, median_cross_sectional_area, std_cross_sectional_area, node1_degree, node2_degree, orientation_z, orientation_y, orientation_x`, plus `mean_cross_sectional_area_exact, median_cross_sectional_area_exact, std_cross_sectional_area_exact, vessel_volume_exact` when `area_image_exact` is given. Lengths are in µm, areas in µm² and volumes in µm³ (see [Section 4.2](#42-vessel-metrics-hdf5)).

---

#### `compute_junction_metrics`

Compute per-node metrics.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
junction_df = compute_junction_metrics(
    graph: nx.Graph,
    junction_metrics_df: pd.DataFrame,  # Unused; a new DataFrame is returned
    distance_threshold: float = 50,     # Neighbor-counting radius in µm
    voxel_size_um: float = 2.0          # Distances are reported in µm
)
```

**Returns:** `pd.DataFrame` indexed by node, with columns: `z, y, x, node_degree, is_sprout_tip, is_junction, nearest_junction_distance, nearest_sprout_tip_distance, junction_neighbor_count, sprout_tip_neighbor_count` (see [Section 4.3](#43-junction-metrics-hdf5))

[`calculate_graph_metrics`](#calculate_graph_metrics) calls it with `distance_threshold=500` (µm).

---

#### `fractal_dimension_and_lacunarity`

Box-counting fractal analysis.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
fd, lacunarity = fractal_dimension_and_lacunarity(
    array: np.ndarray,           # Binary
    max_box_size: int = None,    # log2 of the largest box; default floor(log2(min(array.shape)))
    min_box_size: int = 1,       # log2 of the smallest box (2 voxels)
    n_samples: int = 20,         # Number of box sizes
    n_offsets: int = 0,          # Grid offsets per box size (0 = a single, unshifted grid)
    plot: bool = False           # Accepted; no plot is drawn
)
```

**Returns:** `tuple` (fractal_dimension, lacunarity)

---

#### `convex_hull_volume`

Compute convex hull volume.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
volume = convex_hull_volume(
    binary_image: np.ndarray
)
```

**Returns:** `float` hull volume in voxels³ (0 for fewer than four foreground voxels)

---

### Main Graph Pipeline

#### `process_directory`

Process all masks in a directory.

**Source:** [`graph_main.py`](vascumap/pipeline/graph_main.py)

```python
process_directory(
    masks_dir_path: str,
    suffix: str,                     # e.g., '_vessel_mask'
    save_local_metrics: bool,
    visualization: bool = True,
    skel_method: str = "thinning",   # 'thinning', 'contraction' or 'auto'
    contraction_timelim: int = 300,  # Seconds
    voxel_size_um: float = 2.0,      # Isotropic voxel size of the masks in µm
    compute_exact_area: bool = True  # Also compute the nearest-centerline areas
)
```

Writes the files listed in [Section 2.3.4](#234-output-files) into `masks_dir_path`.

---

#### `construct_vessel_network`

Build graph from binary mask.

**Source:** [`graph_main.py`](vascumap/pipeline/graph_main.py)

```python
graph, area_image, area_image_exact, binary_filled_holes = construct_vessel_network(
    binary_image: np.ndarray,              # Binary vessel mask on an isotropic grid
    global_metrics_df: pd.DataFrame,       # Receives the global volume columns
    skel_method: str = "thinning",         # 'thinning', 'contraction' or 'auto'
    contraction_timelim: int = 300,        # Seconds
    do_prune_graph: bool = True,           # Tip/origin, spur, loop and component pruning
    voxel_size_um: float = 2.0,            # Isotropic voxel size in µm
    compute_exact_area: bool = True,       # Also compute the nearest-centerline areas
    min_spur_um: float = 80.0,             # Terminal branches shorter than this are removed
    min_loop_um: float = 40.0,             # Self-loops shorter than this are removed
    min_component_um: float = 20.0,        # Components shorter in total are removed
    prune_length_cutoff_um: float = 50.0,  # prune_graph length cutoff
    vicinity_xy_um: float = 100.0,         # XY border margin
    vicinity_z_um: float = 2.0,            # Z border margin
    keep_rings: bool = False,              # Keep closed rings without a node
    tip_area_ratio_cutoff: float = 0.20,   # prune_graph tip/origin area ratio
    keep_isolated: bool = False,           # Keep degree-0 nodes (compact objects)
    chunk_size: tuple = None,              # GPU chunk size; default (Z, 512, 512)
    overlap: tuple = None                  # Chunk overlap; default (128, 128, 128)
)
```

**Returns:** `tuple`
- `graph`: NetworkX graph, pruned and border-trimmed
- `area_image`: Cross-sectional areas (voxel²) at skeleton voxels
- `area_image_exact`: Local vessel volume (voxel³) per skeleton voxel, or `None` if `compute_exact_area=False`
- `binary_filled_holes`: The hole-filled binary mask (float32)

Every threshold is given in µm and converted to voxels with `voxel_size_um` (Section 2.3.2). The overlap, clamped per axis to the volume size, must exceed the largest vessel radius, because hole filling, the EDT and skeletonization are not local operations; a single chunk covering the whole volume removes the seams altogether when it fits in GPU memory. The function writes `chip_volume`, `vessel_volume` and `vessel_volume_fraction` into `global_metrics_df`. With `skel_method="auto"`, thinning is used when the vessel volume fraction exceeds 0.3, and mesh contraction otherwise.

---

#### `calculate_graph_metrics`

Compute all metrics from graph.

**Source:** [`graph_main.py`](vascumap/pipeline/graph_main.py)

```python
global_df, junction_df, vessel_df = calculate_graph_metrics(
    graph: nx.Graph,
    sample_name: str,
    area_image: np.ndarray,               # Cross-sectional areas (voxel²)
    global_metrics_df: pd.DataFrame,      # Must already hold chip_volume
    area_image_exact: np.ndarray = None,  # Local volumes (voxel³)
    voxel_size_um: float = 2.0
)
```

**Returns:** `tuple` of DataFrames (global, junction, vessel); the global columns are listed in [Section 4.1](#41-global-metrics-csv)

---

### Visualization

#### `visualize_vessel_network`

Draw the vessel graph over a Z-sum projection of the mask.

**Source:** [`visualization.py`](vascumap/pipeline/visualization.py)

```python
visualize_vessel_network(
    binary_image: np.ndarray,          # Binary mask (background projection)
    graph: nx.Graph,
    area_image: np.ndarray = None,     # Cross-sectional areas (voxel²) at skeleton voxels
    sample_name: str = "Vessel Network",  # Plot title
    voxel_size_um: float = 2.0
)
```

Edges are colored by the median equivalent diameter of their cross-sections, d = 2√(A/π), in µm, with a horizontal color bar below the panel; without `area_image`, every edge is drawn in one color. Nodes are colored by type: sprout tips in lime, other free ends in orange and junctions in white. The figure is left open for the caller to save.

---

#### `graph2image`

Rasterize a graph back into a binary skeleton image.

**Source:** [`visualization.py`](vascumap/pipeline/visualization.py)

```python
skeleton = graph2image(
    graph: nx.Graph,
    shape: tuple,                # Output shape (Z, Y, X)
    include_nodes: bool = False  # Also draw node voxels ('pts_all' when kept)
)
```

**Returns:** `np.ndarray` binary skeleton image

---

## 3.5 Utilities

**Location:** [`vascumap/pipeline/utils.py`](vascumap/pipeline/utils.py)

### `contrast`

Percentile contrast adjustment.

```python
adjusted = contrast(
    arr: np.ndarray,
    low: float,                  # Lower percentile (0-100)
    high: float                  # Upper percentile (0-100)
)
```

**Returns:** `np.ndarray` clipped to percentile range

---

### `scale`

Min-max normalization to [0, 1].

```python
scaled = scale(
    arr: np.ndarray
)
```

**Returns:** `np.ndarray` (float32) in [0, 1]

---

### `resize_dask`

Dask-based 3D rescaling.

```python
resized = resize_dask(
    stack: np.ndarray,           # (Z, Y, X) or (Z, Y, X, C)
    rescale_factor: list         # [z, y, x] factors
)
```

**Returns:** `np.ndarray` resized with cubic interpolation (`order=3`); a trailing channel axis is kept

---

### `cupy_chunk_processing`

Apply GPU function to volume in chunks.

```python
result = cupy_chunk_processing(
    volume: np.ndarray,
    processing_func: callable,   # Takes a CuPy array as first argument
    chunk_size: tuple = (64, 512, 512),
    overlap: tuple = (15, 15, 15),
    *args, **kwargs              # Passed to processing_func (NumPy arrays moved to the GPU)
)
```

**Returns:** `np.ndarray` processed

---

### `median_filter_3d_gpu`

GPU-accelerated 3D median filter.

```python
filtered = median_filter_3d_gpu(
    volume: np.ndarray,
    size: int = 3,
    chunk_size: tuple = (64, 64, 64)
)
```

**Returns:** `np.ndarray` filtered

---

### `process_vessel_mask`

Convert probability map to binary mask.

```python
mask = process_vessel_mask(
    vessel_proba: np.ndarray,    # Probabilities [0, 1]
    ortho: bool = False          # Apply extra filtering
)
```

**Processing:**
- If `ortho=True`: median filter (size=7, GPU chunks of 32 × 1024 × 1024) + hysteresis threshold (0.2, 0.5)
- If `ortho=False`: hysteresis threshold (0.1, 0.5)

**Returns:** `np.ndarray` binary mask

---

### `remove_false_positives`

Remove artifacts using Gaussian fitting.

```python
remove_false_positives(
    mask_p: str                  # Path to mask file
)
```

Fits a Gaussian to the vessel area per Z-plane and, when an excess of detections is found below the fitted peak, sets the planes below the cutoff to zero. In that case, the original file is renamed with every occurrence of `mask` in its path replaced by `mask_backup` (for example `sample_vessel_mask_backup.tif`), and the corrected mask is written in its place. A diagnostic figure, `*mask_correction.png`, is saved whenever the fit succeeds. This function is not called by the pipeline.

---

# 4. Metrics Reference

## 4.1 Global Metrics (CSV)

Stored in `global_metrics_{method}.csv`, one row per sample. Lengths are in µm, areas in µm² and volumes in µm³, converted with `voxel_size_um` (`v` below); "heterogeneity" denotes a standard deviation across vessels or nodes.

### Volume

| Column | Unit | Description | Formula |
|--------|------|-------------|---------|
| `sample` | - | Sample identifier (mask file name without extension) | - |
| `chip_volume` | µm³ | Convex-hull volume of the hole-filled mask inside the border margins (100 µm in XY, 2 µm in Z by default) | `ConvexHull(interior foreground).volume × v³` |
| `vessel_volume` | µm³ | Foreground volume of the hole-filled mask (whole volume) | `sum(filled mask) × v³` |
| `vessel_volume_fraction` | ratio | Vessel volume relative to chip volume | `vessel_volume / chip_volume` |

### Network Topology

All counts and lengths refer to the final graph, after pruning and border trimming.

| Column | Unit | Description | Formula |
|--------|------|-------------|---------|
| `total_vessel_length` | µm | Sum of edge lengths | `Σ length(edge) × v` |
| `total_sprout_count` | count | Edges with at least one sprout-tip end | Vessels connected to tips |
| `total_branch_count` | count | Edges with no sprout-tip end | Edges between junctions |
| `total_junction_count` | count | Nodes with degree ≥ 3 | Branch points |
| `vessel_length_density` | µm⁻² | Vessel length per chip volume | `total_vessel_length / chip_volume` |
| `sprout_density_per_vessel_length` | µm⁻¹ | Sprouts per vessel length | `total_sprout_count / total_vessel_length` |
| `junction_density_per_vessel_length` | µm⁻¹ | Junctions per vessel length | `total_junction_count / total_vessel_length` |
| `sprout_density_per_chip_volume` | µm⁻³ | Sprouts per chip volume | `total_sprout_count / chip_volume` |
| `junction_density_per_chip_volume` | µm⁻³ | Junctions per chip volume | `total_junction_count / chip_volume` |

A sprout tip is a node flagged when the skeleton graph is built (degree 1 before pruning), so an edge cut by border trimming does not turn its remaining end into a sprout tip.

### Fractal Analysis

Computed on the final, pruned and border-trimmed skeleton, rasterized from the graph edges, so that it describes the same structure as the length and count metrics.

| Column | Unit | Description | Formula |
|--------|------|-------------|---------|
| `skeleton_fractal_dimension` | - | Box-counting dimension | `slope(log(N) vs log(1/ε))` |
| `skeleton_lacunarity` | - | Heterogeneity measure | `mean(Var(H)/Mean(H)²)` |

**Interpretation:**
- Fractal dimension ≈ 1: Linear structure
- Fractal dimension ≈ 2: Area-filling
- Fractal dimension ≈ 3: Volume-filling

### Morphology and Spacing

| Column | Unit | Description | Formula |
|--------|------|-------------|---------|
| `median_vessel_tortuosity` | ratio | Median tortuosity of the edges | `median(tortuosity)` |
| `vessel_tortuosity_heterogeneity` | ratio | Spread of edge tortuosity | `std(tortuosity)` |
| `median_vessel_cross_sectional_area` | µm² | Median over edges of the per-edge median area | `median(median_cross_sectional_area)` |
| `vessel_cross_sectional_area_heterogeneity` | µm² | Spread of the per-edge median area | `std(median_cross_sectional_area)` |
| `median_vessel_cross_sectional_area_exact` | µm² | As above, from the nearest-centerline areas | `median(median_cross_sectional_area_exact)` |
| `vessel_cross_sectional_area_heterogeneity_exact` | µm² | As above, from the nearest-centerline areas | `std(median_cross_sectional_area_exact)` |
| `median_junction_nearest_neighbor_distance` | µm | Median distance from a junction to the nearest other junction | `median(nearest_junction_distance)` over `is_junction` nodes |
| `junction_spacing_heterogeneity` | µm | Spread of that distance | `std(nearest_junction_distance)` over `is_junction` nodes |
| `median_sprout_tip_nearest_neighbor_distance` | µm | Median distance from a sprout tip to the nearest other sprout tip | `median(nearest_sprout_tip_distance)` over `is_sprout_tip` nodes |
| `sprout_tip_spacing_heterogeneity` | µm | Spread of that distance | `std(nearest_sprout_tip_distance)` over `is_sprout_tip` nodes |

The two `_exact` columns are present only when the nearest-centerline areas were computed (the default; omitted with `--no_exact_area`). Medians and standard deviations ignore NaN values. The spacing metrics use the `is_junction` flag of the junction table (every node that is not a sprout tip), which can include nodes of degree below 3 that were junctions before pruning or border trimming, whereas `total_junction_count` counts nodes of degree ≥ 3 only.

---

## 4.2 Vessel Metrics (HDF5)

Stored in `{sample}_vessel_metrics_{method}.h5`, one row per edge, indexed by the end nodes of the edge (`node1`, `node2`). `pᵢ` are the centerline voxels of the edge and `v` the voxel size in µm.

| Column | Type | Unit | Description | Formula |
|--------|------|------|-------------|---------|
| `z`, `y`, `x` | array | voxels | Coordinates along vessel | Path points |
| `vessel_volume` | float | µm³ | Segment volume (approximation; biased low on oblique runs) | `Σ area(pᵢ) × v³` (assumes a 1-voxel step) |
| `vessel_length` | float | µm | Path length | `Σ ‖pᵢ₊₁ - pᵢ‖ × v` |
| `end_to_end_distance` | float | µm | Endpoint distance | `‖p_last - p_first‖ × v` |
| `tortuosity` | float | ratio | Curvature measure | `vessel_length / end_to_end_distance` (clipped [0,5]) |
| `is_sprout` | bool | - | Connected to a sprout tip | `sprout(n1) OR sprout(n2)` |
| `mean_cross_sectional_area` | float | µm² | Mean cross-section | `mean(π × r_3D × r_2D) × v²` |
| `median_cross_sectional_area` | float | µm² | Median cross-section | `median(areas) × v²` |
| `std_cross_sectional_area` | float | µm² | Area variability | `std(areas) × v²` |
| `mean_cross_sectional_area_exact` | float | µm² | Mean nearest-centerline cross-section | `mean(V(pᵢ) / Δᵢ) × v²` |
| `median_cross_sectional_area_exact` | float | µm² | Median nearest-centerline cross-section | `median(V(pᵢ) / Δᵢ) × v²` |
| `std_cross_sectional_area_exact` | float | µm² | Variability of the nearest-centerline cross-section | `std(V(pᵢ) / Δᵢ) × v²` |
| `vessel_volume_exact` | float | µm³ | Segment volume from the nearest-centerline partition | `Σ V(pᵢ) × v³` |
| `node1_degree` | int | count | First node degree | - |
| `node2_degree` | int | count | Second node degree | - |
| `orientation_z/y/x` | float | ratio | Direction vector | `(p_last - p_first) / ‖·‖` |
| `sample` | str | - | Sample identifier | - |

`V(pᵢ)` is the local vessel volume (voxel³) assigned to centerline voxel `pᵢ` and `Δᵢ` the local centerline step (Section 5.1). The four `_exact` columns are present only when the nearest-centerline areas were computed. Edges with fewer than two centerline points are skipped.

**Tortuosity interpretation:**
- 1.0: Perfectly straight
- \>1.5: Significantly curved
- Clipped to [0, 5] for robustness

---

## 4.3 Junction Metrics (HDF5)

Stored in `{sample}_junction_metrics_{method}.h5`, one row per node, indexed by the node identifier. Distances are straight-line Euclidean distances between node positions, in µm.

| Column | Type | Unit | Description | Formula |
|--------|------|------|-------------|---------|
| `z`, `y`, `x` | float | voxels | Node position | - |
| `node_degree` | int | count | Node degree | Edges connected |
| `is_sprout_tip` | bool | - | Is sprout tip | Degree 1 in the skeleton graph before pruning |
| `is_junction` | bool | - | Is not a sprout tip | `NOT is_sprout_tip` |
| `nearest_junction_distance` | float | µm | To closest other junction | `min(‖pos - pos_j‖) × v` |
| `nearest_sprout_tip_distance` | float | µm | To closest other sprout tip | `min(‖pos - pos_e‖) × v` |
| `junction_neighbor_count` | int | count | Junctions within threshold | Within 500 µm |
| `sprout_tip_neighbor_count` | int | count | Sprout tips within threshold | Within 500 µm |
| `sample` | str | - | Sample identifier | - |

When a sample has nodes of only one type, the columns of the missing type are NaN (distances) or 0 (counts).

**Degree interpretation:**
- Degree 1: Sprout tip (terminal) when `is_sprout_tip` is True; otherwise a free end left by pruning or border trimming
- Degree 3: Typical bifurcation
- Degree ≥ 4: Complex junction

---

## Unit Conversions

All lengths, areas and volumes are reported in physical units (**µm, µm², µm³**), converted with `voxel_size_um` (`--voxel_size_um`, default 2.0 µm, the isotropic grid of the inference output). Only the coordinate columns `z`, `y` and `x` of both tables remain in voxel indices. To convert them, or to derive a diameter from an area:

```python
import numpy as np

voxel_size_um = 2.0  # the value passed as --voxel_size_um

# Node coordinates in µm
junction_df[['z_um', 'y_um', 'x_um']] = junction_df[['z', 'y', 'x']] * voxel_size_um

# Equivalent diameter (µm) from a cross-sectional area (µm²), as in the graph figure
diameter_um = 2 * np.sqrt(vessel_df['median_cross_sectional_area'] / np.pi)
```

---

# 5. Appendix

## 5.1 Mathematical Formulas

### Cross-Sectional Area

Modeled as ellipse at each skeleton point:

```
Area(p) = π × r_major × r_minor
        = π × EDT_2D(y, x) × EDT_3D(z, y, x)

where:
  EDT_3D = distance to nearest boundary in 3D
  EDT_2D = distance in XY max-projection
```

### Nearest-Centerline (Exact) Cross-Sectional Area

```
V(s)  = number of foreground voxels whose nearest skeleton voxel is s   (voxel³)
Δ(s)  = mean length of the two centerline steps adjacent to s
        (the single adjacent step at either end of an edge)
A(s)  = V(s) / Δ(s)                                                     (voxel²)

vessel_volume_exact = Σ V(s) × v³   over the centerline voxels s of an edge
```

### Sprout Taper Ratio (Pruning)

```
r = mean(A over the 20% of the branch nearest the tip) / A(origin)

The area profile A along a terminal branch is resampled to 100 points;
the half with the smaller mean area is the tip side, and the origin is the
voxel at the opposite end. The branch is removed when r > tip_area_ratio_cutoff
or when its length ≤ length_cutoff.
```

### Tortuosity

```
τ = L_path / L_straight
  = Σ‖pᵢ₊₁ - pᵢ‖ / ‖p_last - p_first‖     (clipped to [0, 5])

Interpretation:
  τ = 1: Straight
  τ > 1: Curved
  τ = 5: Closed loop or extremely curved segment (clipped value)
```

### Fractal Dimension (Box-Counting)

```
D = lim(ε→0) [log N(ε) / log(1/ε)]

Implementation:
  scales = unique(floor(2^k)), k = 20 values evenly spaced from
           floor(log₂(min_dim)) down to 1   (box sizes from 2 voxels)
  N(ε) = count of non-empty boxes at scale ε
  scales that give the same N(ε) are kept once
  D = slope of linear fit: log(N) vs log(1/ε)
```

### Lacunarity

```
λ(ε) = Var(H) / Mean(H)²

where H = number of skeleton voxels in each non-empty box at scale ε;
the reported lacunarity is the mean of λ(ε) over the scales kept for D

Interpretation:
  λ ≈ 0: Homogeneous
  λ > 1: Heterogeneous/clustered
```

---

## 5.2 Troubleshooting

### CUDA Out of Memory

**Solution:** Reduce chunk sizes:
```python
cupy_chunk_processing(volume, func, chunk_size=(32, 256, 256))
```

For graph analysis, smaller chunks are passed to `construct_vessel_network(chunk_size=(Z, 256, 256))`, while the overlap must remain larger than the largest vessel radius; `--no_exact_area` (`compute_exact_area=False`) also skips the nearest-centerline partition. During segmentation, the sliding-window batch sizes are fixed in `prediction.py` (16 or 32 tiles).

### Empty Graph Output

**Causes:** No foreground, over-pruning, vessels lying entirely within the border margins, a wrong voxel size  
**Solutions:**
1. Check `np.sum(mask) > 0`
2. Increase `tip_area_ratio_cutoff` (e.g., 0.5), so that fewer terminal branches are treated as blob artifacts
3. Decrease `prune_length_cutoff_um` and `min_spur_um`, or set `do_prune_graph=False` to inspect the unpruned topology
4. Decrease `vicinity_xy_um` and `vicinity_z_um`
5. Check that `--voxel_size_um` matches the mask grid, because every threshold is defined in µm

### Mesh Contraction Timeout

**Solutions:**
1. Increase `contraction_timelim` of `process_directory` or `construct_vessel_network` (e.g., 600); the command line keeps the default of 300 s
2. Use `skel_method='thinning'` for dense networks

### NaN in Metrics

| Metric | Cause | Solution |
|--------|-------|----------|
| `orientation_*` | Zero-length vector | Remove degenerate edges |
| `skeleton_fractal_dimension`, `skeleton_lacunarity` | Graph without edges | Check mask |
| `*_density_*` | `chip_volume` or `total_vessel_length` is 0 | Check mask and border margins |
| Morphology and spacing columns | Graph without edges (columns left empty) | Check mask |
| `nearest_*_distance` | No nodes of that type (NaN); a single node of that type (inf) | Expected for some samples |

`tortuosity` is never NaN: a small constant (10⁻⁸ voxel) is added to the end-to-end distance, and the ratio is clipped to 5.

---

## 5.3 Contributing

### Code Style

- PEP 8 compliant
- Google-style docstrings
- Line length: 100 characters

```bash
# Format before committing
black vascumap/
isort vascumap/
```

### Pull Request Process

1. Fork and create feature branch
2. Make changes with clear commits
3. Update documentation if needed
4. Submit PR with description

---

*Last updated: October 2026*
