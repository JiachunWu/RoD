import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (
    METRIC_COLUMNS,
    ensure_dir,
    finite_or_nan,
    finish_logger,
    read_subset_image_count,
    sanitize_name,
    setup_logger,
    str_to_bool,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on artifact-source stratified YOLO subsets.")
    parser.add_argument("--models", nargs="+", required=True, help="Model checkpoint paths.")
    parser.add_argument("--model-names", nargs="+", required=True, help="Display names matching --models.")
    parser.add_argument("--data-yamls", nargs="+", required=True, help="Subset data YAML paths.")
    parser.add_argument("--subset-names", nargs="+", required=True, help="Subset names matching --data-yamls.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--agnostic-nms", type=str_to_bool, default=False)
    parser.add_argument("--augment", type=str_to_bool, default=False)
    parser.add_argument("--rect", type=str_to_bool, default=True)
    parser.add_argument("--half", type=str_to_bool, default=False)
    parser.add_argument("--save-json", type=str_to_bool, default=False)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--run-tag", default="fair")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def get_attr(obj, *names, default=float("nan")):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def scalar(value):
    try:
        if hasattr(value, "item"):
            value = value.item()
        return finite_or_nan(value)
    except Exception:
        return float("nan")


def extract_ap75(metrics):
    box = getattr(metrics, "box", None)
    if box is None:
        return float("nan"), "AP75 unavailable: metrics.box missing"
    if hasattr(box, "map75"):
        value = scalar(box.map75)
        if math.isfinite(value):
            return value, ""
    if hasattr(box, "all_ap"):
        try:
            all_ap = np.asarray(box.all_ap, dtype=float)
            if all_ap.ndim == 2 and all_ap.shape[1] >= 6:
                values = all_ap[:, 5]
                values = values[np.isfinite(values)]
                values = values[values >= 0]
                if values.size:
                    return float(np.mean(values)), ""
            if all_ap.ndim == 1 and all_ap.shape[0] >= 6:
                value = float(all_ap[5])
                if math.isfinite(value):
                    return value, ""
        except Exception as exc:
            return float("nan"), f"AP75 unavailable: failed to parse metrics.box.all_ap ({exc})"
    return float("nan"), "AP75 unavailable: metrics.box.all_ap missing or empty"


def empty_metric_row(model_name, model_path, subset_name, data_yaml, num_images, args, runtime, notes):
    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "subset_name": subset_name,
        "data_yaml": str(data_yaml),
        "num_images": num_images,
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "ap": float("nan"),
        "ap50": float("nan"),
        "ap75": float("nan"),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "agnostic_nms": args.agnostic_nms,
        "augment": args.augment,
        "rect": args.rect,
        "half": args.half,
        "run_tag": args.run_tag,
        "runtime_seconds": runtime,
        "notes": notes,
    }


def evaluate_one(model_path, model_name, data_yaml, subset_name, num_images, args, logger):
    from ultralytics import YOLO

    start = time.time()
    notes = []
    run_name = f"{sanitize_name(model_name)}_{sanitize_name(subset_name)}"
    project_dir = Path(args.out_dir) / "val_runs"
    logger.info("Evaluating model=%s subset=%s data=%s", model_name, subset_name, data_yaml)
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        augment=args.augment,
        rect=args.rect,
        half=args.half,
        workers=args.workers,
        save_json=args.save_json,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    runtime = time.time() - start
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("metrics.box missing from Ultralytics result.")
    precision = scalar(get_attr(box, "mp", default=float("nan")))
    recall = scalar(get_attr(box, "mr", default=float("nan")))
    ap = scalar(get_attr(box, "map", default=float("nan")))
    ap50 = scalar(get_attr(box, "map50", default=float("nan")))
    ap75, ap75_note = extract_ap75(metrics)
    if ap75_note:
        notes.append(ap75_note)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    for metric_name, value in {"precision": precision, "recall": recall, "f1": f1, "ap": ap, "ap50": ap50}.items():
        if not math.isfinite(value):
            notes.append(f"{metric_name} is not finite")

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "subset_name": subset_name,
        "data_yaml": str(data_yaml),
        "num_images": num_images,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
        "ap50": ap50,
        "ap75": ap75,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "agnostic_nms": args.agnostic_nms,
        "augment": args.augment,
        "rect": args.rect,
        "half": args.half,
        "run_tag": args.run_tag,
        "runtime_seconds": runtime,
        "notes": "; ".join(notes),
    }


def main():
    args = parse_args()
    logger = setup_logger(args.out_dir, "eval_artifact_subsets.log", args)
    rows = []
    try:
        if len(args.models) != len(args.model_names):
            raise ValueError("ERROR: number of model names does not match number of model paths.")
        if len(args.data_yamls) != len(args.subset_names):
            raise ValueError("ERROR: number of subset names does not match number of data YAML paths.")

        model_paths = [Path(path).resolve() for path in args.models]
        data_yamls = [Path(path).resolve() for path in args.data_yamls]
        for path in model_paths:
            if not path.exists():
                raise FileNotFoundError(f"ERROR: model path does not exist: {path}")
        for path in data_yamls:
            if not path.exists():
                raise FileNotFoundError(f"ERROR: data yaml does not exist: {path}")

        subset_counts = {str(path): read_subset_image_count(path) for path in data_yamls}
        logger.info("Subset image counts: %s", subset_counts)

        for model_path, model_name in zip(model_paths, args.model_names):
            for data_yaml, subset_name in zip(data_yamls, args.subset_names):
                num_images = subset_counts[str(data_yaml)]
                start = time.time()
                try:
                    row = evaluate_one(model_path, model_name, data_yaml, subset_name, num_images, args, logger)
                except Exception as exc:
                    runtime = time.time() - start
                    logger.exception("Evaluation failed for model=%s subset=%s", model_name, subset_name)
                    row = empty_metric_row(model_name, model_path, subset_name, data_yaml, num_images, args, runtime, f"ERROR: {exc}")
                rows.append(row)
                write_csv(Path(args.out_dir) / "metrics" / "per_run_metrics.csv", rows, METRIC_COLUMNS)
                write_json(Path(args.out_dir) / "metrics" / "per_run_metrics.json", rows)

        write_csv(Path(args.out_dir) / "metrics" / "combined_subset_metrics.csv", rows, METRIC_COLUMNS)
        write_json(Path(args.out_dir) / "metrics" / "per_run_metrics.json", rows)
        logger.info("Evaluation rows written: %d", len(rows))
    finally:
        finish_logger(logger)


if __name__ == "__main__":
    main()
