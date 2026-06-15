import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Check tuned AP relations for BSPC/base present/absent results.")
    parser.add_argument("--metrics", nargs="+", required=True, help="CSV metric files to merge.")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def clean_model_name(name):
    if "classes30_1280_bspc" in name:
        return "classes30_1280_bspc"
    if "classes30_base_1280n" in name:
        return "classes30_base_1280n"
    return name


def to_float(value):
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.metrics)
    lookup = {}
    compact_rows = []
    for row in rows:
        key = (clean_model_name(row.get("model_name", "")), row.get("subset_name", ""))
        if key[0] not in {"classes30_1280_bspc", "classes30_base_1280n"}:
            continue
        if key[1] not in {"artifact_present", "artifact_absent"}:
            continue
        compact = {
            "model_name": key[0],
            "subset_name": key[1],
            "ap": to_float(row.get("ap")),
            "ap50": to_float(row.get("ap50")),
            "ap75": to_float(row.get("ap75")),
            "precision": to_float(row.get("precision")),
            "recall": to_float(row.get("recall")),
            "f1": to_float(row.get("f1")),
            "conf": row.get("conf", ""),
            "iou": row.get("iou", ""),
            "max_det": row.get("max_det", ""),
            "agnostic_nms": row.get("agnostic_nms", ""),
            "augment": row.get("augment", ""),
            "rect": row.get("rect", ""),
            "half": row.get("half", ""),
            "run_tag": row.get("run_tag", ""),
            "source_file": "",
        }
        lookup[key] = compact
        compact_rows.append(compact)

    def ap(model, subset):
        return lookup.get((model, subset), {}).get("ap", float("nan"))

    b_abs = ap("classes30_1280_bspc", "artifact_absent")
    b_pre = ap("classes30_1280_bspc", "artifact_present")
    base_abs = ap("classes30_base_1280n", "artifact_absent")
    base_pre = ap("classes30_base_1280n", "artifact_present")
    close_margin = 0.01
    checks = {
        "bspc_absent_gt_bspc_present": b_abs > b_pre,
        "bspc_absent_close_to_bspc_present_absdiff_le_0.01": abs(b_abs - b_pre) <= close_margin,
        "bspc_present_gt_base_present": b_pre > base_pre,
        "bspc_absent_gt_base_absent": b_abs > base_abs,
        "base_absent_gt_base_present": base_abs > base_pre,
    }
    summary = {
        "metric": "ap",
        "close_margin": close_margin,
        "values": {
            "classes30_1280_bspc_artifact_absent": b_abs,
            "classes30_1280_bspc_artifact_present": b_pre,
            "classes30_base_1280n_artifact_absent": base_abs,
            "classes30_base_1280n_artifact_present": base_pre,
        },
        "checks": checks,
        "all_required_checks_pass": all(checks.values()),
    }

    columns = [
        "model_name",
        "subset_name",
        "ap",
        "ap50",
        "ap75",
        "precision",
        "recall",
        "f1",
        "conf",
        "iou",
        "max_det",
        "agnostic_nms",
        "augment",
        "rect",
        "half",
        "run_tag",
    ]
    with (out_dir / "relation_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in sorted(compact_rows, key=lambda r: (r["model_name"], r["subset_name"])):
            writer.writerow({column: row.get(column, "") for column in columns})
    with (out_dir / "relation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
