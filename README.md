# RoD

Code for the manuscript: RoDNet: Spatial-Frequency Collaboration and Artifact-Aware Stabilization for Multi-Scale Dental Anomaly Detection.

## RoD Release

This repository does not include:

- Dataset images and labels
- Training weights
- Experiment artifacts such as `runs/`, `outputs/`, and `result/`
- Demo projects, test caches, and unrelated large files

## Included Content

- `ultralytics/`
  Customized Ultralytics source code, including the custom module registration and implementations required by the experiments.
- `scripts/train.py`
  Parameterized training entry point.
- `scripts/val.py`
  Parameterized validation entry point.
- `tools/artifact_experiment/`
  Artifact-source stratified evaluation scripts for the `Dental_X` dataset.
- `tools/artifact_experiment_v2_PDR-10/`
  Stratified evaluation, fair evaluation, and tuned inference scripts for the `PDR-10` dataset.
- `docs/artifact-experimen.md`
  Experiment plan for the `Dental_X` version.
- `docs/artifact-experimen-v2.md`
  Experiment plan for the `PDR-10` version.

## Environment

Python 3.10+ is recommended. Install the dependencies first:

```bash
pip install -r requirements.txt
```

If you are using a clean environment, you may also need to make sure these packages are available:

```bash
pip install ultralytics prettytable pywavelets dill timm
```

## Training Example

```bash
python scripts/train.py \
  --model ultralytics/cfg/models/12/our_WTconv_DYT.yaml \
  --data path/to/PDR-10/data.yaml \
  --name PDR10_our_WTconv_DYT
```

## Validation Example

```bash
python scripts/val.py \
  --model path/to/best_fp32.pt \
  --data path/to/PDR-10/data.yaml \
  --split test \
  --save-json
```

## Stratified Evaluation

Both sets of stratified evaluation scripts are retained:

- `tools/artifact_experiment_v2_PDR-10/`: three-model fair evaluation and tuned inference follow-up tests for `PDR-10`.

By default, these scripts require you to provide local dataset paths and weight paths.

## Dataset Notes

The manuscript evaluates RoDNet on six datasets covering panoramic dental radiographs, chest X-ray images, and natural drone imagery. The released scripts in this repository primarily target `Dental_X` and `PDR-10`; the remaining datasets are documented here to make the paper's evaluation setup traceable.

- `ODet3`
  A 31-class oral disease detection dataset based on panoramic dental radiographs. The manuscript uses 9,676 training images, 2,764 validation images, and 1,378 test images. It can be obtained from the Roboflow Universe release: [ODet3: Oral Disease Detection Dataset, Release 3](https://universe.roboflow.com/di-hastalklar/dis-hastaliklarinin-tespiti3-vukvr).

- `Dental_X` / `DXR-Pano`
  A 31-class panoramic dental X-ray detection dataset. In the manuscript, this benchmark is reported as `DXR-Pano`; in the retained scripts, the same dataset family is referred to as `Dental_X`. The manuscript uses 9,674 training images, 2,760 validation images, and 1,380 test images. It can be obtained from Roboflow Universe: [Dental X-Ray Panoramic Dataset](https://universe.roboflow.com/celldetection-ok5sm/dental-x-ray-panoramic-dataset).

- `PDR-10`
  A 10-class panoramic dental radiograph detection dataset used as the main benchmark for ablation, efficiency, visualization, and artifact-source robustness analysis. The manuscript uses 9,382 training images, 2,854 validation images, and 1,577 test images. It can be obtained from Roboflow Universe: [PDR-10: Ten-Class Panoramic Dental Radiograph Dataset](https://universe.roboflow.com/p-jwewf/30classes-uxmcx).

- `PerioXrays`
  A single-class panoramic radiograph benchmark for clinical-oriented apical periodontitis detection. The manuscript uses 3,000 training images, 637 validation images, and 637 test images. The dataset is introduced by the PerioDet benchmark paper; obtain it from the official PerioDet/PerioXrays release channel associated with that work: [PerioDet: Large-Scale Panoramic Radiograph Benchmark for Clinical-Oriented Apical Periodontitis Detection](https://papers.miccai.org/miccai-2025/0687-Paper1336.html).

- `CXR-Rad`
  A 5-class chest X-ray radiograph detection dataset used to evaluate transferability to non-dental medical anomaly detection. The manuscript uses 4,493 training images, 1,242 validation images, and 496 test images. It can be obtained from Roboflow Universe: [A Dataset of Chest X-Ray Radiographs](https://universe.roboflow.com/willy-aapee/chest-x-ray-images-u9us8).

- `VisDrone`
  A 12-class natural-image multi-scale object detection benchmark collected from drone-mounted cameras. The manuscript uses 6,471 training images, 548 validation images, and 548 test images. It can be obtained from the official VisDrone dataset release: [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset).

For `PDR-10`, the paper additionally performs artifact-source stratification under clinician guidance. Categories associated with restorative components, treatment traces, or implant-related structures are assigned to the with-artifact group; the remaining categories are assigned to the without-artifact group. A test image is treated as with-artifact if it contains at least one object from the with-artifact categories.

The repository does not redistribute the datasets themselves. Download or request each dataset from its original provider, follow the provider's license and access terms, and prepare a YOLO-format `data.yaml`. For the retained scripts, the expected directory structure is `images/{train,val,test}` and `labels/{train,val,test}`.

For more detailed usage, see [docs/DATASET.md](docs/DATASET.md).
