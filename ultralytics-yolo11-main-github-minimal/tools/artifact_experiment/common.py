import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ARTIFACT_SOURCE_CLASS_IDS = {2, 4, 6, 14, 18, 19, 22, 25, 26, 27, 28, 29, 30}
ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS = {0, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20, 21, 23, 24}

MANIFEST_COLUMNS = [
    "split",
    "image_path",
    "label_path",
    "image_stem",
    "image_ext",
    "label_exists",
    "label_empty",
    "num_gt_boxes",
    "gt_class_ids",
    "gt_class_names",
    "artifact_source_class_ids",
    "artifact_source_class_names",
    "artifact_source_box_count",
    "artifact_source_label",
    "has_no_gt",
    "warning",
]

METRIC_COLUMNS = [
    "model_name",
    "model_path",
    "subset_name",
    "data_yaml",
    "num_images",
    "precision",
    "recall",
    "f1",
    "ap",
    "ap50",
    "ap75",
    "imgsz",
    "batch",
    "device",
    "conf",
    "iou",
    "max_det",
    "runtime_seconds",
    "notes",
]


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(out_dir, log_name, args=None):
    out_dir = Path(out_dir)
    logs_dir = ensure_dir(out_dir / "logs")
    logger = logging.getLogger(log_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logs_dir / log_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Start time: %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("Command: %s", " ".join([sys.executable] + sys.argv))
    if args is not None:
        logger.info("Arguments: %s", vars(args))

    commands_path = logs_dir / "commands.txt"
    with commands_path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} | {' '.join([sys.executable] + sys.argv)}\n")
    return logger


def finish_logger(logger):
    logger.info("End time: %s", datetime.now().isoformat(timespec="seconds"))


def load_yaml(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"ERROR: {path} is not a YAML mapping.")
    return data


def write_yaml(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_names(names):
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    if not isinstance(names, dict):
        raise ValueError("ERROR: data.yaml does not contain key 'names' as a list or mapping.")
    normalized = {}
    for key, value in names.items():
        try:
            class_id = int(key)
        except Exception as exc:
            raise ValueError(f"ERROR: class id {key!r} in names is not an integer.") from exc
        normalized[class_id] = str(value)
    expected = set(range(31))
    got = set(normalized)
    if got != expected:
        raise ValueError(f"ERROR: data.yaml names must contain class ids 0-30. Missing={sorted(expected - got)}, extra={sorted(got - expected)}.")
    return dict(sorted(normalized.items()))


def validate_artifact_mapping():
    union = ARTIFACT_SOURCE_CLASS_IDS.union(ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS)
    inter = ARTIFACT_SOURCE_CLASS_IDS.intersection(ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS)
    expected = set(range(31))
    if union != expected:
        raise ValueError(f"ERROR: artifact class mapping does not cover all classes 0-30. Missing={sorted(expected - union)}, extra={sorted(union - expected)}.")
    if inter:
        raise ValueError(f"ERROR: artifact class mapping overlaps for class ids {sorted(inter)}.")


def run_sample_label_tests():
    example_a = [4, 4, 4, 14, 1, 1, 1, 1, 15, 9, 9, 9, 9]
    label_a, ids_a, count_a = artifact_summary_from_class_ids(example_a)
    if label_a != 1 or ids_a != [4, 14] or count_a != 4:
        raise AssertionError("ERROR: sample-label test A failed.")

    example_b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 1, 10, 10, 10, 1]
    label_b, ids_b, count_b = artifact_summary_from_class_ids(example_b)
    if label_b != 0 or ids_b != [] or count_b != 0:
        raise AssertionError("ERROR: sample-label test B failed.")


def artifact_summary_from_class_ids(class_ids):
    artifact_box_count = sum(1 for class_id in class_ids if class_id in ARTIFACT_SOURCE_CLASS_IDS)
    artifact_class_ids = sorted({class_id for class_id in class_ids if class_id in ARTIFACT_SOURCE_CLASS_IDS})
    artifact_source_label = 1 if artifact_box_count > 0 else 0
    return artifact_source_label, artifact_class_ids, artifact_box_count


def resolve_dataset_root(data_yaml_path, data):
    data_yaml_path = Path(data_yaml_path).resolve()
    root_value = data.get("path", data_yaml_path.parent)
    root = Path(root_value)
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()
    return root


def resolve_split_path(data_yaml_path, data, split):
    if split not in data:
        raise ValueError(f"ERROR: data.yaml does not contain key {split!r}.")
    value = data[split]
    root = resolve_dataset_root(data_yaml_path, data)
    if isinstance(value, str):
        split_path = Path(value)
        if not split_path.is_absolute():
            split_path = root / split_path
        return split_path.resolve()
    raise ValueError(f"ERROR: split {split!r} must be a string path in data.yaml.")


def collect_images(split_path):
    split_path = Path(split_path)
    if split_path.is_dir():
        images = [p.resolve() for p in split_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(images, key=lambda p: str(p))
    if split_path.is_file():
        images = []
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    images.append(Path(line).expanduser().resolve())
        return sorted(images, key=lambda p: str(p))
    raise FileNotFoundError(f"ERROR: split path does not exist: {split_path}")


def image_to_label_path(image_path):
    image_path = Path(image_path).resolve()
    parts = list(image_path.parts)
    label_parts = parts[:]
    indices = [i for i, part in enumerate(label_parts) if part == "images"]
    if not indices:
        raise ValueError(f"ERROR: cannot derive label path because image path has no 'images' segment: {image_path}")
    label_parts[indices[-1]] = "labels"
    return Path(*label_parts).with_suffix(".txt")


def parse_label_file(label_path, strict_labels=False):
    label_path = Path(label_path)
    if not label_path.exists():
        if strict_labels:
            raise FileNotFoundError(f"ERROR: missing label file: {label_path}")
        return [], False, True, "missing label file"

    raw_lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    non_empty = [line.strip() for line in raw_lines if line.strip()]
    if not non_empty:
        return [], True, True, ""

    class_ids = []
    for line_number, line in enumerate(non_empty, start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except Exception as exc:
            raise ValueError(f"ERROR: invalid class_id in {label_path}:{line_number}: {parts[0]!r}") from exc
        if class_id < 0 or class_id >= 31:
            raise ValueError(f"ERROR: label file contains class_id {class_id}, but valid range is 0-30: {label_path}:{line_number}")
        class_ids.append(class_id)
    return class_ids, True, False, ""


def join_ids(values):
    return ";".join(str(v) for v in values)


def join_names(ids, class_names):
    return ";".join(class_names[class_id] for class_id in ids)


def make_manifest_row(split, image_path, class_names, strict_labels=False):
    image_path = Path(image_path).resolve()
    label_path = image_to_label_path(image_path)
    class_ids, label_exists, label_empty, warning = parse_label_file(label_path, strict_labels=strict_labels)
    artifact_label, artifact_ids, artifact_box_count = artifact_summary_from_class_ids(class_ids)
    gt_ids = sorted(set(class_ids))
    return {
        "split": split,
        "image_path": str(image_path),
        "label_path": str(label_path.resolve()),
        "image_stem": image_path.stem,
        "image_ext": image_path.suffix,
        "label_exists": bool(label_exists),
        "label_empty": bool(label_empty),
        "num_gt_boxes": int(len(class_ids)),
        "gt_class_ids": join_ids(gt_ids),
        "gt_class_names": join_names(gt_ids, class_names),
        "artifact_source_class_ids": join_ids(artifact_ids),
        "artifact_source_class_names": join_names(artifact_ids, class_names),
        "artifact_source_box_count": int(artifact_box_count),
        "artifact_source_label": int(artifact_label),
        "has_no_gt": bool(len(class_ids) == 0),
        "warning": warning,
    }


def write_csv(path, rows, columns):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def count_classes_from_manifest_rows(rows):
    counts = Counter()
    for row in rows:
        ids = parse_semicolon_ints(row.get("gt_class_ids", ""))
        for class_id in ids:
            # This counts classes present per image, not boxes. Use labels for box-level counts.
            counts[class_id] += 1
    return counts


def parse_semicolon_ints(value):
    if value is None or value == "":
        return []
    return [int(x) for x in str(value).split(";") if x != ""]


def safe_percent(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def build_summary(rows, class_names):
    summary = {}
    groups = {}
    for row in rows:
        groups.setdefault(row["split"], []).append(row)
    groups["all"] = rows

    for split, split_rows in groups.items():
        class_box_counts = Counter()
        artifact_box_counts = Counter()
        non_artifact_box_counts = Counter()
        for row in split_rows:
            label_path = row["label_path"]
            if row["label_exists"] in {True, "True", "true", "1"} and Path(label_path).exists():
                class_ids, _, _, _ = parse_label_file(label_path)
                for class_id in class_ids:
                    class_box_counts[class_id] += 1
                    if class_id in ARTIFACT_SOURCE_CLASS_IDS:
                        artifact_box_counts[class_id] += 1
                    else:
                        non_artifact_box_counts[class_id] += 1

        num_total = len(split_rows)
        num_present = sum(int(row["artifact_source_label"]) == 1 for row in split_rows)
        num_absent = num_total - num_present
        total_boxes = sum(int(row["num_gt_boxes"]) for row in split_rows)
        artifact_boxes = sum(int(row["artifact_source_box_count"]) for row in split_rows)
        summary[split] = {
            "num_images_total": num_total,
            "num_images_artifact_absent": num_absent,
            "num_images_artifact_present": num_present,
            "percent_images_artifact_absent": safe_percent(num_absent, num_total),
            "percent_images_artifact_present": safe_percent(num_present, num_total),
            "num_gt_boxes_total": total_boxes,
            "num_artifact_source_boxes_total": artifact_boxes,
            "artifact_source_box_percent": safe_percent(artifact_boxes, total_boxes),
            "artifact_source_class_counts": {
                f"{class_id} {class_names[class_id]}": int(artifact_box_counts[class_id])
                for class_id in sorted(ARTIFACT_SOURCE_CLASS_IDS)
            },
            "non_artifact_source_class_counts": {
                f"{class_id} {class_names[class_id]}": int(non_artifact_box_counts[class_id])
                for class_id in sorted(ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS)
            },
            "missing_label_files": sum(row["label_exists"] in {False, "False", "false", "0"} for row in split_rows),
            "empty_label_files": sum(row["label_empty"] in {True, "True", "true", "1"} for row in split_rows),
            "top_10_classes_by_gt_box_count": [
                {"class": f"{class_id} {class_names[class_id]}", "count": int(count)}
                for class_id, count in sorted(class_box_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
            "top_artifact_source_classes_by_gt_box_count": [
                {"class": f"{class_id} {class_names[class_id]}", "count": int(count)}
                for class_id, count in sorted(artifact_box_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
            ],
        }
    return summary


def class_mapping_json(class_names):
    return {
        "artifact_source_label_definition": {
            "0": "artifact absent / negligible based on absence of radiopaque artifact-source classes",
            "1": "artifact present based on presence of at least one radiopaque artifact-source class",
        },
        "important_note": "This is an artifact-source-based stratification, not exhaustive manual artifact ground truth.",
        "artifact_source_class_ids": sorted(ARTIFACT_SOURCE_CLASS_IDS),
        "artifact_absent_or_negligible_class_ids": sorted(ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS),
        "class_names": {str(k): v for k, v in class_names.items()},
    }


def sanitize_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "run"


def finite_or_nan(value):
    try:
        value = float(value)
    except Exception:
        return float("nan")
    if math.isfinite(value):
        return value
    return float("nan")


def read_subset_image_count(data_yaml):
    data = load_yaml(data_yaml)
    test_value = data.get("test")
    if not test_value:
        raise ValueError(f"ERROR: {data_yaml} does not contain key 'test'.")
    test_path = Path(test_value)
    if test_path.is_file():
        with test_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if test_path.is_dir():
        return len(collect_images(test_path))
    raise FileNotFoundError(f"ERROR: test path in {data_yaml} does not exist: {test_path}")


def markdown_table(rows, columns):
    def cell(value):
        if value is None:
            return ""
        text = str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body) + "\n"


def format_number(value, digits=4):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"

