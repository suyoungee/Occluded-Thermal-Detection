# Occluded Thermal Detection

This repository contains the research code for **“Data Augmentation for
Improving Thermal Image Detection under Partial Occlusion,”** presented at the
38th Workshop on Image Processing and Image Understanding (IPIU 2026).

The work targets missed detections caused by vegetation and other partial
occlusions in thermal surveillance imagery. It combines two complementary data
augmentation methods:

- **Inpainting augmentation:** randomly masks part of a labeled object and uses
  SimpleLaMa to reconstruct the masked region from the surrounding background.
- **FLUX augmentation:** creates irregular fractal-tree masks and uses
  FLUX.1-Fill-dev to synthesize thermal-looking branches, fog, and sensor noise.

The original, inpainted, and FLUX-generated images are mixed to improve the
occlusion robustness of a YOLO11m detector.

## Paper results

The following values are the results reported in the paper. They have not been
rerun as part of the repository documentation update.

| Training data | Original test recall | Original test mAP50 | Original test mAP50–95 |
| --- | ---: | ---: | ---: |
| Original baseline | 0.820 | 0.895 | 0.580 |
| All Mixed (ours) | **0.869** | **0.909** | 0.561 |

The All Mixed model also achieved the following results on occlusion-focused
test sets:

| Test set | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: |
| Inpainting | 0.848 | 0.892 | 0.541 |
| FLUX | 0.797 | 0.818 | 0.488 |
| All Mixed | 0.821 | 0.873 | 0.528 |

The paper used the following training configuration:

| Hyperparameter | Value |
| --- | --- |
| Model | YOLO11m |
| Input resolution | 640 × 640 |
| Batch size | 64 |
| Epochs | 40 |
| Optimizer | SGD |
| Initial learning rate | 1e-4 |
| Momentum | 0.937 |
| Weight decay | 5e-4 |
| Inpainting probability | 0.8 |

Some scripts retain experiment-specific values from later repository work.
Review their configuration constants before attempting an exact reproduction
of the paper.

## Repository structure

```text
configs/
  flir_multi.yaml              # Multi-source training configuration
  roboflow_data.yaml           # Original FLIR dataset configuration
scripts/
  1_convert_data.py            # Optional MOT-style CSV to YOLO conversion
  2_train.py                   # YOLO11 training
  3_val.py                     # Validation
  4_test.py                    # Test-set evaluation
  5_predict.py                 # Qualitative prediction
  6_check_data_integrity.py    # Dataset split-overlap check
  7_check_labels_integrity.py  # Image-label pairing check
  8_random_inpainting.py       # SimpleLaMa augmentation
  9_split_train_images.py      # Training-image sharding
  10_generate_occlusion.py     # FLUX occlusion generation
data/
  FLUX1-Fill-dev/              # Generated FLUX samples and YOLO labels
```

The scripts are research utilities rather than a packaged command-line
application. Dataset paths, checkpoint names, and generation shards are set
directly in each script or exposed through the existing arguments.

## Installation

Create an isolated Python environment and install the dependencies:

```bash
git clone https://github.com/suyoungee/Occluded-Thermal-Detection.git
cd Occluded-Thermal-Detection

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The paper experiments were performed on Ubuntu 20.04 with an NVIDIA RTX A6000.
FLUX generation requires a CUDA-capable environment with enough memory to load
the quantized model components.

FLUX generation also requires access to
[FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev).
Accept the model terms and authenticate with Hugging Face before running the
generation script.

## Dataset

The experiments use the
[FLIR Thermal Dataset v27](https://universe.roboflow.com/thermal-imaging-0hwfw/flir-data-set/dataset/27)
with the configured classes `car`, `dog`, and `person`. Place the
downloaded dataset under `data/` and verify that the paths in the selected YAML
configuration match the local directory layout.

The Roboflow YOLO export does not need conversion. `1_convert_data.py` is an
optional utility for a compatible raw dataset that uses the MOT-style CSV
schema expected by that script.

Check the original dataset before training:

```bash
python scripts/6_check_data_integrity.py
python scripts/7_check_labels_integrity.py
```

## Data augmentation

### Inpainting

`8_random_inpainting.py` accepts input and output paths as arguments. For
example:

```bash
python scripts/8_random_inpainting.py \
  --input_images data/RoboFlow_FLIR_Dataset_v27_yolo11/train/images \
  --input_labels data/RoboFlow_FLIR_Dataset_v27_yolo11/train/labels \
  --output_images data/inpainting/train/images \
  --output_labels data/inpainting/train/labels \
  --prob 0.8
```

The original YOLO labels are retained because the detector is expected to infer
the complete object from its visible regions.

### FLUX occlusion generation

Split the source training images into the three shards expected by the
generation script:

```bash
python scripts/9_split_train_images.py
```

Set `SPLIT_FOLDER` in `10_generate_occlusion.py` to the shard to process,
then run:

```bash
python scripts/10_generate_occlusion.py
```

The script uses the thermal-domain prompt described in the paper and fills
procedural branch masks with FLUX-generated content.

## Training and evaluation

Confirm `DATA_CONFIG`, the training parameters, and GPU IDs in
`2_train.py`, then run:

```bash
python scripts/2_train.py
```

Before validation, set `BEST_WEIGHTS` in `3_val.py`:

```bash
python scripts/3_val.py
```

For test evaluation, review `SELECT_WEIGHTS` in `4_test.py`:

```bash
python scripts/4_test.py
```

For qualitative predictions, set `BEST_WEIGHTS` in `5_predict.py` and
place input images in `assets/`:

```bash
python scripts/5_predict.py
```

## Citation

```bibtex
@inproceedings{gong2026occludedthermal,
  title     = {Data Augmentation for Improving Thermal Image Detection
               under Partial Occlusion},
  author    = {Gong, Minjoong and Cho, Suyoung and Park, Jinsun},
  booktitle = {Proceedings of the 38th Workshop on Image Processing
               and Image Understanding (IPIU)},
  year      = {2026}
}
```

## Acknowledgements

This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
[SimpleLaMa](https://github.com/enesmsahin/simple-lama-inpainting), and
[FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev).
