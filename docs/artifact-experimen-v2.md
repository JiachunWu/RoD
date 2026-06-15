# artifact-experimen.md

# Artifact-source-based Evaluation Plan for RoDNet

## 0. Purpose of this document

This document is a complete task specification for Codex to implement and run an artifact-source-based evaluation experiment for the current dental panoramic radiograph object detection dataset.

The goal is to add a rigorous, reproducible, and minimally subjective artifact-related quantitative analysis to the RoDNet paper without inventing new artifact metrics.

The experiment must use the current YOLO labels and class definitions to split test images into:

```text
0 = artifact absent / negligible
1 = artifact present
```

Important terminology:

```text
"artifact present" in this experiment means:
the image contains at least one radiopaque / metallic / restorative / implant-related / endodontic / prosthetic object class that is likely to act as an artifact source or high-response distractor in panoramic radiographs.

It does NOT mean:
a radiologist manually confirmed every possible image artifact such as ghost jaw, spine overlay, air-gap shadow, motion blur, or exposure artifact.
```

Therefore, the experiment must be described in the code, logs, and generated report as:

```text
artifact-source-based stratification
```

or:

```text
radiopaque artifact-source stratification
```

Do not call the automatically generated label a definitive artifact ground truth.

The final outputs must allow the paper to report standard existing detection metrics separately on:

```text
artifact absent / negligible subset
artifact present subset
```

and optionally report the relative performance drop from the artifact-absent subset to the artifact-present subset, following common robustness-evaluation practice.

No novel artifact metric should be introduced in the main analysis.

---

## 1. Dataset information available to Codex

The user provided a YOLO dataset configuration file named:

```text
data(1).yaml
```

For implementation, Codex should support any user-provided YOLO YAML path passed through `--data`. The file name above is only the current uploaded file name.

The current uploaded dataset contains **10 object classes**.

The expected class definitions are:

```yaml
names:
  0: Crown
  1: Filling
  2: Fracture teeth
  3: Implant
  4: Retained root
  5: Root Canal Treatment
  6: Root Piece
  7: abutment
  8: gingival former
  9: impacted tooth
```

If `data(1).yaml` uses a `path`, `train`, `val`, and `test` structure, Codex must parse it exactly as a standard YOLO dataset YAML.

A typical YOLO YAML structure may be:

```yaml
path: /path/to/dataset/root
train: images/train
val: images/val
test: images/test

names:
  0: Crown
  1: Filling
  2: Fracture teeth
  3: Implant
  4: Retained root
  5: Root Canal Treatment
  6: Root Piece
  7: abutment
  8: gingival former
  9: impacted tooth
```

The dataset uses standard YOLO detection label format:

```text
class_id x_center y_center width height
```

where all coordinates are normalized to `[0, 1]`.

Two example labels were provided.

Example A contains classes:

```text
0 Crown
5 Root Canal Treatment
1 Filling
```

Because it contains class `0 Crown`, class `5 Root Canal Treatment`, and class `1 Filling`, it must be assigned:

```text
artifact_source_label = 1
```

Example B contains only class:

```text
9 impacted tooth
```

Because `impacted tooth` is not an artifact-source class under this protocol, it must be assigned:

```text
artifact_source_label = 0
```

Codex must implement these sample checks as unit tests or sanity checks.

---

## 2. Core definition: class-level artifact-source mapping

### 2.1 Artifact-source-positive classes

The following classes must be mapped to:

```text
artifact_source_class = 1
```

These classes represent radiopaque restorative, implant-related, endodontic, prosthetic, or artificial dental structures that can serve as artifact sources or high-response distractors in dental panoramic radiographs.

```python
ARTIFACT_SOURCE_CLASS_IDS = {
    0,  # Crown
    1,  # Filling
    3,  # Implant
    5,  # Root Canal Treatment
    7,  # abutment
    8,  # gingival former
}
```

Expected class names for artifact-source-positive classes:

| Class ID | Class name | Artifact-source label | Reason |
|---:|---|---:|---|
| 0 | Crown | 1 | A crown is a restorative/prosthetic structure that commonly appears as a high-density radiopaque region in panoramic radiographs. It may create strong model responses and interfere with nearby lesion or small-object detection. |
| 1 | Filling | 1 | A filling is a restorative material and is commonly radiopaque. It is a typical high-density restorative source that may attract model attention or produce artifact-like distractors. |
| 3 | Implant | 1 | An implant is usually metallic or highly radiopaque. It is a strong artificial structure and a plausible source of local high-response interference in panoramic radiographs. |
| 5 | Root Canal Treatment | 1 | Root canal treatment commonly includes radiopaque endodontic filling material. It often appears as a line-like or column-like high-density structure and can influence local detector responses. |
| 7 | abutment | 1 | An abutment is related to implant/prosthetic restoration and is generally an artificial high-density structure. It should be treated as an artifact-source class. |
| 8 | gingival former | 1 | A gingival former is an implant-related artificial component and is generally radiopaque. It should be treated as an artifact-source class. |

### 2.2 Artifact-source-negative classes

The following classes must be mapped to:

```text
artifact_source_class = 0
```

They are tooth abnormalities, retained/root structures, tooth-position abnormalities, or non-artificial pathological/anatomical targets. Their presence alone should not assign an image to the artifact-present subset.

```python
ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS = {
    2,  # Fracture teeth
    4,  # Retained root
    6,  # Root Piece
    9,  # impacted tooth
}
```

Expected class names for artifact-source-negative classes:

| Class ID | Class name | Artifact-source label | Reason |
|---:|---|---:|---|
| 2 | Fracture teeth | 0 | Tooth fracture is a dental abnormality or structural tooth condition, not a metallic/restorative/implant/endodontic artifact-source object. |
| 4 | Retained root | 0 | Retained root is a retained dental root or pathological/status target. It should not be automatically treated as an artificial high-density artifact source. |
| 6 | Root Piece | 0 | Root piece is a tooth/root fragment target. It is not automatically equivalent to metallic or restorative material. It should remain artifact-source-negative unless future manual review confirms otherwise. |
| 9 | impacted tooth | 0 | Impacted tooth is a tooth-position or anatomical/pathological condition. It is not an artificial radiopaque object and should not be used as evidence of artifact-source presence. |

### 2.3 Required consistency check

Codex must verify that:

```python
ARTIFACT_SOURCE_CLASS_IDS.union(ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS) == set(range(10))
ARTIFACT_SOURCE_CLASS_IDS.intersection(ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS) == set()
```

If this check fails, stop execution and raise a clear error.

Codex must also verify that each class ID in the labels is within:

```python
0 <= class_id < 10
```

If a label file contains any class ID outside this range, stop execution in strict mode or log a serious error in non-strict mode.

---

## 3. Image-level artifact-source label generation

### 3.1 Image-level rule

For each image:

```text
artifact_source_label(image) = 1
```

if the image has at least one ground-truth object whose class ID belongs to:

```python
ARTIFACT_SOURCE_CLASS_IDS
```

Otherwise:

```text
artifact_source_label(image) = 0
```

Formal definition:

```python
if any(class_id in ARTIFACT_SOURCE_CLASS_IDS for class_id in image_gt_class_ids):
    artifact_source_label = 1
else:
    artifact_source_label = 0
```

### 3.2 Images with empty label files

If an image has an empty YOLO label file, assign:

```text
artifact_source_label = 0
```

Reason:

```text
No annotated artifact-source object exists in that image.
```

However, Codex must also record:

```text
has_no_gt = True
```

in the manifest.

### 3.3 Images with missing label files

If an image exists but the corresponding label file is missing:

- Do not silently ignore it.
- Add it to a warning list.
- Assign:

```text
artifact_source_label = 0
```

only if the project’s existing dataset/evaluation code treats missing label files as empty labels.

Recommended behavior:

```text
Default: treat missing labels as empty labels but log a warning.
Strict mode: if --strict-labels is passed, stop with an error.
```

### 3.4 Image-to-label path conversion

Use YOLO convention:

```text
/images/ -> /labels/
image extension -> .txt
```

Examples:

```text
/path/to/dataset/images/test/example.jpg
/path/to/dataset/labels/test/example.txt
```

Codex must support image extensions:

```text
.jpg
.jpeg
.png
.bmp
.tif
.tiff
.webp
```

Case-insensitive.

---

## 4. Required scripts to implement

Codex must implement the following scripts.

Recommended directory:

```text
tools/artifact_experiment/
```

Required scripts:

```text
tools/artifact_experiment/01_build_artifact_manifest.py
tools/artifact_experiment/02_make_artifact_subset_yamls.py
tools/artifact_experiment/03_eval_artifact_subsets.py
tools/artifact_experiment/04_summarize_artifact_results.py
```

Optionally, if the existing project structure prefers a single script, Codex may implement:

```text
tools/run_artifact_experiment.py
```

but the single script must still clearly execute the four stages above.

---

## 5. Script 01: build artifact manifest

### 5.1 Purpose

Create an image-level manifest for each split, especially the test split, with artifact-source labels derived from YOLO ground-truth classes.

### 5.2 Command-line interface

Implement:

```bash
python tools/artifact_experiment/01_build_artifact_manifest.py \
  --data /path/to/data.yaml \
  --splits train val test \
  --out-dir outputs/artifact_experiment \
  --strict-labels false
```

Required arguments:

| Argument | Type | Required | Description |
|---|---|---:|---|
| `--data` | str | yes | Path to YOLO `data.yaml` or `data(1).yaml`. |
| `--splits` | list[str] | no | Splits to process. Default: `test`. Allowed values: `train`, `val`, `test`. |
| `--out-dir` | str | yes | Output directory. |
| `--strict-labels` | bool | no | If true, missing label files cause failure. Default: false. |

### 5.3 Inputs

The script must parse:

```text
data.yaml
```

or any other YAML path passed through `--data`, including:

```text
data(1).yaml
```

The script must support both relative paths under the dataset root and absolute paths.

If the YAML contains:

```yaml
path: /path/to/dataset/root
train: images/train
val: images/val
test: images/test
```

The script must resolve:

```text
train image directory = /path/to/dataset/root/images/train
val image directory   = /path/to/dataset/root/images/val
test image directory  = /path/to/dataset/root/images/test
```

If a split value is already an absolute path or a text file containing image paths, Codex must support it.

### 5.4 Outputs

Create:

```text
outputs/artifact_experiment/manifests/artifact_manifest_train.csv
outputs/artifact_experiment/manifests/artifact_manifest_val.csv
outputs/artifact_experiment/manifests/artifact_manifest_test.csv
outputs/artifact_experiment/manifests/artifact_manifest_all.csv
outputs/artifact_experiment/stats/artifact_class_mapping.json
outputs/artifact_experiment/stats/artifact_manifest_summary.json
outputs/artifact_experiment/logs/build_artifact_manifest.log
```

If only the test split is processed, create only:

```text
outputs/artifact_experiment/manifests/artifact_manifest_test.csv
outputs/artifact_experiment/manifests/artifact_manifest_all.csv
```

plus the same stats and log files.

### 5.5 Manifest columns

Each manifest CSV must contain exactly these columns, in this order:

```text
split
image_path
label_path
image_stem
image_ext
label_exists
label_empty
num_gt_boxes
gt_class_ids
gt_class_names
artifact_source_class_ids
artifact_source_class_names
artifact_source_box_count
artifact_source_label
has_no_gt
warning
```

Column definitions:

| Column | Description |
|---|---|
| `split` | `train`, `val`, or `test`. |
| `image_path` | Absolute path to image. |
| `label_path` | Absolute path to corresponding YOLO label file. |
| `image_stem` | File stem without extension. |
| `image_ext` | Image extension. |
| `label_exists` | Boolean. |
| `label_empty` | Boolean. |
| `num_gt_boxes` | Number of valid YOLO boxes. |
| `gt_class_ids` | Semicolon-separated class IDs present in the image. Example: `0;1;5`. Unique IDs only, sorted ascending. |
| `gt_class_names` | Semicolon-separated class names corresponding to `gt_class_ids`. |
| `artifact_source_class_ids` | Semicolon-separated artifact-source class IDs found in the image. Empty string if none. |
| `artifact_source_class_names` | Semicolon-separated artifact-source class names found in the image. Empty string if none. |
| `artifact_source_box_count` | Count of GT boxes whose class is in artifact-source classes. |
| `artifact_source_label` | Integer, 0 or 1. |
| `has_no_gt` | Boolean, true if no GT boxes are available. |
| `warning` | Empty string if no issue; otherwise a concise warning message. |

### 5.6 JSON class mapping

Create:

```text
outputs/artifact_experiment/stats/artifact_class_mapping.json
```

Content must include:

```json
{
  "artifact_source_label_definition": {
    "0": "artifact absent / negligible based on absence of radiopaque artifact-source classes",
    "1": "artifact present based on presence of at least one radiopaque artifact-source class"
  },
  "important_note": "This is an artifact-source-based stratification, not exhaustive manual artifact ground truth.",
  "artifact_source_class_ids": [0, 1, 3, 5, 7, 8],
  "artifact_absent_or_negligible_class_ids": [2, 4, 6, 9],
  "class_names": {
    "0": "Crown",
    "1": "Filling",
    "2": "Fracture teeth",
    "3": "Implant",
    "4": "Retained root",
    "5": "Root Canal Treatment",
    "6": "Root Piece",
    "7": "abutment",
    "8": "gingival former",
    "9": "impacted tooth"
  }
}
```

### 5.7 Summary JSON

Create:

```text
outputs/artifact_experiment/stats/artifact_manifest_summary.json
```

It must contain, for each split and for all splits combined:

```json
{
  "test": {
    "num_images_total": 0,
    "num_images_artifact_absent": 0,
    "num_images_artifact_present": 0,
    "percent_images_artifact_absent": 0.0,
    "percent_images_artifact_present": 0.0,
    "num_gt_boxes_total": 0,
    "num_artifact_source_boxes_total": 0,
    "artifact_source_box_percent": 0.0,
    "artifact_source_class_counts": {
      "0 Crown": 0,
      "1 Filling": 0,
      "3 Implant": 0,
      "5 Root Canal Treatment": 0,
      "7 abutment": 0,
      "8 gingival former": 0
    },
    "non_artifact_source_class_counts": {
      "2 Fracture teeth": 0,
      "4 Retained root": 0,
      "6 Root Piece": 0,
      "9 impacted tooth": 0
    },
    "missing_label_files": 0,
    "empty_label_files": 0
  }
}
```

### 5.8 Required sanity checks

After script execution, print to console and log:

```text
Number of images in each split
Number and percentage of artifact_source_label = 0
Number and percentage of artifact_source_label = 1
Top 10 classes by GT box count
Top artifact-source classes by GT box count
Number of missing label files
Number of empty label files
```

### 5.9 Sample-label tests

Add a function or small test block that verifies the current uploaded sample labels.

#### Example A

The first uploaded label sample contains the following class IDs:

```python
[
    0, 0,
    5, 5, 5,
    0, 0, 0,
    5,
    0,
    5,
    1,
    0, 0, 0,
    5,
    0,
    5,
    1, 1, 1, 1, 1, 1, 1,
    5, 5, 5,
    1
]
```

It must produce:

```python
artifact_source_label == 1
artifact_source_class_ids == [0, 1, 5]
artifact_source_box_count == 30
```

Explanation:

```text
All boxes in this example belong to Crown, Root Canal Treatment, or Filling, and all three classes are artifact-source-positive classes.
```

#### Example B

The second uploaded label sample contains the following class IDs:

```python
[9, 9, 9, 9]
```

It must produce:

```python
artifact_source_label == 0
artifact_source_class_ids == []
artifact_source_box_count == 0
```

Explanation:

```text
All boxes are impacted tooth, which is artifact-source-negative.
```

If tests fail, stop with an error.

---

## 6. Script 02: make artifact subset YAMLs

### 6.1 Purpose

Create subset image lists and temporary YOLO data YAMLs for standard YOLO/Ultralytics validation on artifact-absent and artifact-present subsets.

### 6.2 Command-line interface

Implement:

```bash
python tools/artifact_experiment/02_make_artifact_subset_yamls.py \
  --data /path/to/data.yaml \
  --manifest outputs/artifact_experiment/manifests/artifact_manifest_test.csv \
  --split test \
  --out-dir outputs/artifact_experiment
```

### 6.3 Outputs

Create:

```text
outputs/artifact_experiment/splits/test_all.txt
outputs/artifact_experiment/splits/test_artifact_absent.txt
outputs/artifact_experiment/splits/test_artifact_present.txt

outputs/artifact_experiment/data_yaml/test_all.yaml
outputs/artifact_experiment/data_yaml/test_artifact_absent.yaml
outputs/artifact_experiment/data_yaml/test_artifact_present.yaml
```

### 6.4 Text file contents

Each `.txt` file must contain one absolute image path per line.

Definitions:

```text
test_all.txt:
all test images

test_artifact_absent.txt:
only images with artifact_source_label == 0

test_artifact_present.txt:
only images with artifact_source_label == 1
```

### 6.5 YAML contents

Each subset YAML must preserve the original class names and dataset root.

Example:

```yaml
path: /path/to/dataset/root
train: images/train
val: images/val
test: /absolute/path/to/outputs/artifact_experiment/splits/test_artifact_present.txt

names:
  0: Crown
  1: Filling
  2: Fracture teeth
  3: Implant
  4: Retained root
  5: Root Canal Treatment
  6: Root Piece
  7: abutment
  8: gingival former
  9: impacted tooth
```

Codex must ensure that the `test` field points to the correct subset `.txt` file.

### 6.6 Required checks

Before writing subset YAMLs, verify:

```text
artifact_absent subset has at least 1 image
artifact_present subset has at least 1 image
```

If either subset is empty, stop and produce a clear error.

Also report:

```text
num_test_all
num_test_artifact_absent
num_test_artifact_present
```

---

## 7. Script 03: evaluate artifact subsets using existing detection metrics

### 7.1 Purpose

Run standard object detection evaluation on:

```text
test_all
test_artifact_absent
test_artifact_present
```

for each model.

The metrics must be existing standard object detection metrics:

```text
AP / mAP50-95
AP50 / mAP@0.5
AP75 / mAP@0.75
Precision
Recall
F1
```

No new artifact metric should be introduced in this main evaluation.

### 7.2 Required model inputs

The script must accept one or more model checkpoints.

Example:

```bash
python tools/artifact_experiment/03_eval_artifact_subsets.py \
  --models /path/to/yolov12n.pt /path/to/rodnet.pt /path/to/rodnet_wo_bacm.pt \
  --model-names YOLOv12n RoDNet RoDNet_wo_BACM \
  --data-yamls \
      outputs/artifact_experiment/data_yaml/test_all.yaml \
      outputs/artifact_experiment/data_yaml/test_artifact_absent.yaml \
      outputs/artifact_experiment/data_yaml/test_artifact_present.yaml \
  --subset-names all artifact_absent artifact_present \
  --imgsz 1280 \
  --batch 16 \
  --device 0 \
  --out-dir outputs/artifact_experiment
```

### 7.3 Required defaults

Default values:

```text
imgsz = 1280
batch = 16
device = 0
conf = 0.001
iou = 0.7
max_det = 300
```

If the main project uses different validation settings, Codex must allow the user to override every setting.

### 7.4 Preferred implementation

Use the same evaluation path as the existing project whenever possible.

If the project uses Ultralytics-style validation, use Python API:

```python
from ultralytics import YOLO
model = YOLO(model_path)
metrics = model.val(
    data=subset_yaml,
    split="test",
    imgsz=imgsz,
    batch=batch,
    device=device,
    conf=conf,
    iou=iou,
    max_det=max_det,
    save_json=True,
    project=project_dir,
    name=run_name,
    exist_ok=True,
)
```

Extract metrics robustly:

```python
precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map5095 = metrics.box.map
```

For AP75, try the following in order:

1. If `metrics.box.all_ap` exists and has IoU-index dimension, use the IoU index corresponding to 0.75.
2. Otherwise, parse the saved JSON using pycocotools and compute AP at IoU=0.75.
3. If neither is available, record `AP75` as `NaN` and log a clear warning.

F1 must be computed as the standard harmonic mean:

```python
f1 = 2 * precision * recall / (precision + recall + 1e-16)
```

Do not call F1 a novel metric. It is a standard detection/classification summary metric.

### 7.5 Output files

Create:

```text
outputs/artifact_experiment/metrics/per_run_metrics.csv
outputs/artifact_experiment/metrics/per_run_metrics.json
outputs/artifact_experiment/metrics/combined_subset_metrics.csv
outputs/artifact_experiment/logs/eval_artifact_subsets.log
```

`per_run_metrics.csv` columns:

```text
model_name
model_path
subset_name
data_yaml
num_images
precision
recall
f1
ap
ap50
ap75
imgsz
batch
device
conf
iou
max_det
runtime_seconds
notes
```

Definitions:

| Column | Definition |
|---|---|
| `ap` | COCO-style AP averaged over IoU thresholds 0.50:0.95. |
| `ap50` | AP at IoU=0.50. |
| `ap75` | AP at IoU=0.75. |
| `precision` | Mean precision returned by the validator. |
| `recall` | Mean recall returned by the validator. |
| `f1` | Harmonic mean of precision and recall. |
| `notes` | Warning or implementation note, e.g., AP75 unavailable. |

### 7.6 Required validation checks

For every model and subset, verify:

```text
metrics are finite where available
num_images matches subset text file length
model path exists
data yaml exists
```

If a model fails on one subset, continue evaluating other models/subsets if possible, but mark the failed run with `notes` and a clear error message.

---

## 8. Script 04: summarize artifact results

### 8.1 Purpose

Generate final tables and summaries for paper writing.

### 8.2 Command-line interface

Implement:

```bash
python tools/artifact_experiment/04_summarize_artifact_results.py \
  --metrics outputs/artifact_experiment/metrics/per_run_metrics.csv \
  --manifest-summary outputs/artifact_experiment/stats/artifact_manifest_summary.json \
  --out-dir outputs/artifact_experiment
```

### 8.3 Outputs

Create:

```text
outputs/artifact_experiment/tables/table_artifact_stratification.md
outputs/artifact_experiment/tables/table_artifact_stratification.csv

outputs/artifact_experiment/tables/table_artifact_stratified_performance.md
outputs/artifact_experiment/tables/table_artifact_stratified_performance.csv

outputs/artifact_experiment/tables/table_relative_performance_drop.md
outputs/artifact_experiment/tables/table_relative_performance_drop.csv

outputs/artifact_experiment/report/artifact_experiment_summary.md
```

### 8.4 Table 1: artifact stratification

`table_artifact_stratification.csv` columns:

```text
split
num_images_total
num_images_artifact_absent
num_images_artifact_present
percent_images_artifact_absent
percent_images_artifact_present
num_gt_boxes_total
num_artifact_source_boxes_total
artifact_source_box_percent
top_artifact_source_classes
```

### 8.5 Table 2: artifact-stratified detection performance

`table_artifact_stratified_performance.csv` columns:

```text
model_name
subset_name
num_images
ap
ap50
ap75
precision
recall
f1
```

Rows must be sorted by:

```text
model_name
subset_name order: all, artifact_absent, artifact_present
```

### 8.6 Table 3: relative performance drop

This table is optional but recommended. It uses existing robustness reporting style.

For each model:

```text
relative drop from artifact_absent to artifact_present
```

Formula:

```text
Relative drop (%) = (Metric_artifact_absent - Metric_artifact_present) / Metric_artifact_absent * 100
```

Compute for:

```text
AP
AP50
AP75
Precision
Recall
F1
```

If denominator is zero or NaN, record NaN and log warning.

Do not present relative drop as a novel artifact metric. Present it as a robustness summary derived from standard detection metrics.

`table_relative_performance_drop.csv` columns:

```text
model_name
relative_drop_ap_percent
relative_drop_ap50_percent
relative_drop_ap75_percent
relative_drop_precision_percent
relative_drop_recall_percent
relative_drop_f1_percent
```

### 8.7 Summary report

Create:

```text
outputs/artifact_experiment/report/artifact_experiment_summary.md
```

It must include:

1. Dataset path.
2. Split evaluated.
3. Class mapping.
4. Artifact-source-positive class list.
5. Artifact-source-negative class list.
6. Number of images in each subset.
7. Detection metrics table.
8. Relative performance drop table.
9. Exact commands used.
10. Important limitations text.

The limitations text must include:

```text
This experiment uses artifact-source-based stratification derived from existing object labels. The artifact-present group indicates that at least one radiopaque restorative, implant-related, endodontic, prosthetic, or artificial dental structure is annotated in the image. It does not exhaustively capture all image artifacts such as ghost jaw, spine overlay, pharyngeal air-gap, motion blur, exposure artifacts, or other unannotated projection artifacts.
```

---

## 9. Optional: TIDE error analysis using existing error types

This is optional and should only be implemented if time permits.

The user specifically does not want invented artifact metrics. TIDE is acceptable because it uses existing standard error categories.

### 9.1 Purpose

Run TIDE on:

```text
artifact_absent subset
artifact_present subset
```

for each model, and compare standard TIDE error types:

```text
Classification error
Localization error
Both classification and localization error
Duplicate detection error
Background error
Missed ground-truth error
```

### 9.2 Implementation requirements

If implementing TIDE:

1. Convert YOLO ground truth to COCO format for each subset.
2. Convert model predictions to COCO detection JSON.
3. Run TIDE using its official or installed API.
4. Do not define any new artifact-specific error category.

### 9.3 Output files

Create:

```text
outputs/artifact_experiment/tide/tide_artifact_absent.csv
outputs/artifact_experiment/tide/tide_artifact_present.csv
outputs/artifact_experiment/tide/tide_error_comparison.md
```

Columns:

```text
model_name
subset_name
classification_error
localization_error
both_error
duplicate_error
background_error
missed_error
```

If TIDE cannot be installed or run cleanly, do not block the main experiment. Log the issue and continue.

---

## 10. Exact expected workflow

Codex must make the following workflow possible.

### 10.1 Step 1: Build manifest

```bash
python tools/artifact_experiment/01_build_artifact_manifest.py \
  --data data\(1\).yaml \
  --splits train val test \
  --out-dir outputs/artifact_experiment
```

If the shell does not require escaping parentheses, this is also valid:

```bash
python tools/artifact_experiment/01_build_artifact_manifest.py \
  --data "data(1).yaml" \
  --splits train val test \
  --out-dir outputs/artifact_experiment
```

Expected outputs:

```text
outputs/artifact_experiment/manifests/artifact_manifest_train.csv
outputs/artifact_experiment/manifests/artifact_manifest_val.csv
outputs/artifact_experiment/manifests/artifact_manifest_test.csv
outputs/artifact_experiment/manifests/artifact_manifest_all.csv
outputs/artifact_experiment/stats/artifact_class_mapping.json
outputs/artifact_experiment/stats/artifact_manifest_summary.json
```

### 10.2 Step 2: Build subset YAMLs

```bash
python tools/artifact_experiment/02_make_artifact_subset_yamls.py \
  --data "data(1).yaml" \
  --manifest outputs/artifact_experiment/manifests/artifact_manifest_test.csv \
  --split test \
  --out-dir outputs/artifact_experiment
```

Expected outputs:

```text
outputs/artifact_experiment/splits/test_all.txt
outputs/artifact_experiment/splits/test_artifact_absent.txt
outputs/artifact_experiment/splits/test_artifact_present.txt
outputs/artifact_experiment/data_yaml/test_all.yaml
outputs/artifact_experiment/data_yaml/test_artifact_absent.yaml
outputs/artifact_experiment/data_yaml/test_artifact_present.yaml
```

### 10.3 Step 3: Evaluate models

Example command:

```bash
python tools/artifact_experiment/03_eval_artifact_subsets.py \
  --models runs/detect/yolov12n/weights/best.pt runs/detect/rodnet/weights/best.pt runs/detect/rodnet_wo_bacm/weights/best.pt \
  --model-names YOLOv12n RoDNet RoDNet_wo_BACM \
  --data-yamls \
      outputs/artifact_experiment/data_yaml/test_all.yaml \
      outputs/artifact_experiment/data_yaml/test_artifact_absent.yaml \
      outputs/artifact_experiment/data_yaml/test_artifact_present.yaml \
  --subset-names all artifact_absent artifact_present \
  --imgsz 1280 \
  --batch 16 \
  --device 0 \
  --out-dir outputs/artifact_experiment
```

If only two models are available, the command can be:

```bash
python tools/artifact_experiment/03_eval_artifact_subsets.py \
  --models runs/detect/yolov12n/weights/best.pt runs/detect/rodnet/weights/best.pt \
  --model-names YOLOv12n RoDNet \
  --data-yamls \
      outputs/artifact_experiment/data_yaml/test_all.yaml \
      outputs/artifact_experiment/data_yaml/test_artifact_absent.yaml \
      outputs/artifact_experiment/data_yaml/test_artifact_present.yaml \
  --subset-names all artifact_absent artifact_present \
  --imgsz 1280 \
  --batch 16 \
  --device 0 \
  --out-dir outputs/artifact_experiment
```

### 10.4 Step 4: Summarize

```bash
python tools/artifact_experiment/04_summarize_artifact_results.py \
  --metrics outputs/artifact_experiment/metrics/per_run_metrics.csv \
  --manifest-summary outputs/artifact_experiment/stats/artifact_manifest_summary.json \
  --out-dir outputs/artifact_experiment
```

Expected outputs:

```text
outputs/artifact_experiment/tables/table_artifact_stratification.md
outputs/artifact_experiment/tables/table_artifact_stratified_performance.md
outputs/artifact_experiment/tables/table_relative_performance_drop.md
outputs/artifact_experiment/report/artifact_experiment_summary.md
```

---

## 11. Implementation details Codex must follow

### 11.1 Do not modify the original dataset

Codex must not move, rename, edit, or delete any original image or label file.

All generated files must be placed under:

```text
outputs/artifact_experiment/
```

### 11.2 Use absolute paths in split text files

To avoid path resolution issues, split text files must contain absolute image paths.

### 11.3 Preserve original class names exactly

The class name:

```text
Root Canal Treatment
```

must be preserved exactly as written in `data(1).yaml`.

The class name:

```text
gingival former
```

must be preserved exactly as written in `data(1).yaml`.

Do not silently convert class names to lowercase, uppercase, snake_case, or hyphenated forms.

### 11.4 Deterministic output

All scripts must be deterministic.

If any sorting is required, sort by absolute image path.

### 11.5 Logging

Each script must create a log file in:

```text
outputs/artifact_experiment/logs/
```

The logs must include:

```text
start time
end time
command-line arguments
resolved dataset paths
number of images processed
warnings
errors
```

### 11.6 Error handling

Use clear errors.

Examples:

```text
ERROR: data.yaml does not contain key 'names'.
ERROR: label file contains class_id 10, but valid range is 0-9.
ERROR: artifact_present subset is empty.
ERROR: number of model names does not match number of model paths.
```

### 11.7 Required Python packages

Use only common packages where possible:

```text
pyyaml
pandas
numpy
tqdm
ultralytics
```

Optional:

```text
pycocotools
tidecv
matplotlib
```

If optional packages are missing, do not break the main workflow unless the user explicitly requested that optional function.

---

## 12. Required output validation checklist

Codex must ensure the final experiment output passes this checklist.

### 12.1 Manifest checklist

- [ ] `artifact_manifest_test.csv` exists.
- [ ] Every test image has one row.
- [ ] `artifact_source_label` contains only 0 or 1.
- [ ] At least one image has label 0.
- [ ] At least one image has label 1.
- [ ] Class IDs are valid.
- [ ] Artifact-source classes match the fixed mapping.

### 12.2 Subset checklist

- [ ] `test_artifact_absent.txt` exists.
- [ ] `test_artifact_present.txt` exists.
- [ ] Sum of subset image counts equals all test images.
- [ ] No image path appears in both absent and present subsets.
- [ ] Every image path exists.

### 12.3 Evaluation checklist

- [ ] Each model has results for `all`.
- [ ] Each model has results for `artifact_absent`.
- [ ] Each model has results for `artifact_present`.
- [ ] Metrics CSV contains AP, AP50, AP75, precision, recall, and F1.
- [ ] If AP75 is unavailable, the reason is logged clearly.
- [ ] The same validation settings are used across all subsets and models.

### 12.4 Report checklist

- [ ] Class mapping is included.
- [ ] Artifact-source definition is included.
- [ ] Dataset subset sizes are included.
- [ ] Detection performance table is included.
- [ ] Relative performance drop table is included.
- [ ] Limitations are included.
- [ ] Commands used are included.

---

## 13. Paper-ready wording generated from the experiment

The summary report must include a paper-ready paragraph similar to the following, with numerical placeholders filled after running the experiment.

```text
To examine whether the proposed model remains robust in the presence of radiopaque artifact sources, we performed an artifact-source-based stratified evaluation on the test set. Based on the available object annotations, restorative, implant-related, endodontic, prosthetic, and artificial component categories were treated as radiopaque artifact-source classes. Specifically, Crown, Filling, Implant, Root Canal Treatment, abutment, and gingival former were assigned as artifact-source-positive classes. Images containing at least one such class were assigned to the artifact-present subset, whereas images containing only Fracture teeth, Retained root, Root Piece, or impacted tooth were assigned to the artifact-absent/negligible subset. This stratification was used only for robustness analysis and should be interpreted as artifact-source presence rather than exhaustive manual annotation of all imaging artifacts.

We evaluated each model on the full test set and on the two artifact-source subsets using standard detection metrics, including AP, AP50, AP75, precision, recall, and F1-score. The relative performance drop from the artifact-absent/negligible subset to the artifact-present subset was also reported as a robustness summary derived from the same standard metrics.
```

Do not automatically claim that RoDNet is better unless the computed results support it.

If RoDNet is better on the artifact-present subset, the report may say:

```text
Compared with the baseline detector, RoDNet achieved higher AP50 and recall on the artifact-present subset and exhibited a smaller relative performance drop, suggesting improved robustness under radiopaque artifact-source conditions.
```

If RoDNet is not better, the report must state the result honestly.

---

## 14. Why this experiment is scientifically defensible

Codex should include this rationale in the generated report, but keep it concise.

1. The current dataset already contains labels for several radiopaque artificial or restorative structures.
2. These structures are not artifacts themselves, but they are plausible artifact sources or high-response distractors in panoramic radiographs.
3. Using them to stratify images is objective and reproducible because it depends on existing ground-truth class labels.
4. The evaluation uses existing standard object detection metrics rather than invented artifact metrics.
5. The limitation is that projection artifacts without corresponding object labels are not captured.

---

## 15. Final notes for Codex

Do not overcomplicate the first implementation.

The minimum successful implementation is:

```text
1. Build manifest.
2. Generate artifact absent/present test split files and YAMLs.
3. Evaluate YOLOv12n and RoDNet on both subsets.
4. Produce standard metric tables.
```

Only implement TIDE and optional plots after the core experiment works.

Do not change model code.

Do not retrain models.

Do not change dataset labels.

Do not remove any class from evaluation.

Do not exclude artifact-source objects from detection evaluation.

The artifact-source label is used only to split images into subsets for stratified evaluation.
