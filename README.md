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

The experiments use a YOLO-format datasets:

- `PDR-10`
  A 10-class dental panoramic radiograph detection dataset using the same standard YOLO detection format.

The repository does not provide the datasets themselves. If you already have the corresponding data, prepare your own `data.yaml`. Otherwise, obtain the data from the original annotation source or your project data management location, then organize it into the standard YOLO detection format.

For more detailed usage, see [docs/DATASET.md](docs/DATASET.md).
