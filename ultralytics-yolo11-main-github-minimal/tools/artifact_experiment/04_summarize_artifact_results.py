import argparse
import math
from pathlib import Path

import pandas as pd

from common import (
    ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS,
    ARTIFACT_SOURCE_CLASS_IDS,
    ensure_dir,
    finish_logger,
    format_number,
    load_yaml,
    markdown_table,
    read_json,
    setup_logger,
    write_csv,
)


SUBSET_ORDER = {"all": 0, "artifact_absent": 1, "artifact_present": 2}
METRICS = ["ap", "ap50", "ap75", "precision", "recall", "f1"]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize artifact-source stratified evaluation results.")
    parser.add_argument("--metrics", required=True, help="Path to per_run_metrics.csv.")
    parser.add_argument("--manifest-summary", required=True, help="Path to artifact_manifest_summary.json.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--data", default="data.yaml", help="Original YOLO data.yaml for report metadata.")
    return parser.parse_args()


def top_class_string(counts, limit=5):
    items = [(name, int(count)) for name, count in counts.items() if int(count) > 0]
    items.sort(key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{name}: {count}" for name, count in items[:limit])


def stratification_rows(summary):
    rows = []
    for split in ["train", "val", "test", "all"]:
        if split not in summary:
            continue
        item = summary[split]
        rows.append(
            {
                "split": split,
                "num_images_total": item["num_images_total"],
                "num_images_artifact_absent": item["num_images_artifact_absent"],
                "num_images_artifact_present": item["num_images_artifact_present"],
                "percent_images_artifact_absent": format_number(item["percent_images_artifact_absent"], 2),
                "percent_images_artifact_present": format_number(item["percent_images_artifact_present"], 2),
                "num_gt_boxes_total": item["num_gt_boxes_total"],
                "num_artifact_source_boxes_total": item["num_artifact_source_boxes_total"],
                "artifact_source_box_percent": format_number(item["artifact_source_box_percent"], 2),
                "top_artifact_source_classes": top_class_string(item["artifact_source_class_counts"]),
            }
        )
    return rows


def performance_rows(metrics_df):
    rows = []
    df = metrics_df.copy()
    df["_subset_order"] = df["subset_name"].map(SUBSET_ORDER).fillna(99)
    df = df.sort_values(["model_name", "_subset_order", "subset_name"])
    for _, row in df.iterrows():
        rows.append(
            {
                "model_name": row["model_name"],
                "subset_name": row["subset_name"],
                "num_images": int(row["num_images"]),
                "ap": format_number(row["ap"]),
                "ap50": format_number(row["ap50"]),
                "ap75": format_number(row["ap75"]),
                "precision": format_number(row["precision"]),
                "recall": format_number(row["recall"]),
                "f1": format_number(row["f1"]),
            }
        )
    return rows


def relative_drop(absent, present):
    try:
        absent = float(absent)
        present = float(present)
    except Exception:
        return float("nan")
    if not math.isfinite(absent) or not math.isfinite(present) or absent == 0:
        return float("nan")
    return (absent - present) / absent * 100.0


def drop_rows(metrics_df, logger):
    rows = []
    for model_name, group in metrics_df.groupby("model_name", sort=True):
        absent = group[group["subset_name"] == "artifact_absent"]
        present = group[group["subset_name"] == "artifact_present"]
        if absent.empty or present.empty:
            logger.warning("Missing absent/present rows for model %s", model_name)
            continue
        absent = absent.iloc[0]
        present = present.iloc[0]
        row = {"model_name": model_name}
        for metric in METRICS:
            value = relative_drop(absent[metric], present[metric])
            row[f"relative_drop_{metric}_percent"] = format_number(value, 2)
        rows.append(row)
    return rows


def write_table_pair(out_dir, name, rows, columns):
    tables_dir = ensure_dir(Path(out_dir) / "tables")
    write_csv(tables_dir / f"{name}.csv", rows, columns)
    (tables_dir / f"{name}.md").write_text(markdown_table(rows, columns), encoding="utf-8")


def metric_value(metrics_df, model_name, subset_name, metric):
    rows = metrics_df[(metrics_df["model_name"] == model_name) & (metrics_df["subset_name"] == subset_name)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][metric])


def best_model_statement(metrics_df):
    models = sorted(metrics_df["model_name"].unique())
    if len(models) < 2:
        return "Only one model was evaluated, so no between-model comparison is reported."
    present_scores = []
    for model in models:
        present_scores.append((model, metric_value(metrics_df, model, "artifact_present", "ap")))
    finite_scores = [(model, score) for model, score in present_scores if math.isfinite(score)]
    if not finite_scores:
        return "Artifact-present AP is unavailable for all models, so no performance superiority claim is made."
    best_model, best_score = max(finite_scores, key=lambda item: item[1])
    return f"On the artifact-present subset, the highest AP was achieved by {best_model} (AP={best_score:.4f})."


def read_commands(out_dir):
    path = Path(out_dir) / "logs" / "commands.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report_text(args, summary, strat_rows, perf_rows, drop_rows_out, metrics_df):
    data = load_yaml(args.data)
    dataset_path = data.get("path", "")
    names = data["names"]
    artifact_positive = [f"{class_id}: {names[class_id] if isinstance(names, list) else names.get(class_id, names.get(str(class_id)))}" for class_id in sorted(ARTIFACT_SOURCE_CLASS_IDS)]
    artifact_negative = [f"{class_id}: {names[class_id] if isinstance(names, list) else names.get(class_id, names.get(str(class_id)))}" for class_id in sorted(ARTIFACT_ABSENT_OR_NEGLIGIBLE_CLASS_IDS)]
    commands = read_commands(args.out_dir)

    test_summary = summary.get("test", {})
    lines = [
        "# Artifact-source-based Stratified Evaluation Report",
        "",
        "## Task Flow",
        "",
        "1. Built image-level artifact-source manifests from existing YOLO labels.",
        "2. Generated test split files and temporary YOLO data YAMLs for all, artifact-absent/negligible, and artifact-present subsets.",
        "3. Evaluated both checkpoints on all three subsets with the same Ultralytics validation settings.",
        "4. Summarized standard detection metrics and relative performance drops for paper reporting.",
        "",
        "## Code Improvement Logic and Key Functions",
        "",
        "The implementation follows Occam's razor: a small shared helper module contains deterministic path resolution, class mapping checks, YOLO label parsing, CSV/JSON/YAML writing, and markdown table generation. Four thin scripts map directly to the four required workflow stages, so each stage can be rerun independently without changing model code or dataset files.",
        "",
        "Key functions include artifact class mapping validation, sample-label sanity checks, image-to-label path conversion, manifest row construction, subset YAML generation, Ultralytics validation metric extraction, AP75 extraction from existing validator outputs, and relative-drop table generation.",
        "",
        "## Dataset and Split",
        "",
        f"- Dataset path: `{dataset_path}`",
        "- Split evaluated: `test`",
        f"- Test images total: {test_summary.get('num_images_total', 'NA')}",
        f"- Artifact absent / negligible images: {test_summary.get('num_images_artifact_absent', 'NA')}",
        f"- Artifact present images: {test_summary.get('num_images_artifact_present', 'NA')}",
        "",
        "## Class Mapping",
        "",
        "Artifact-source-positive classes:",
        "",
        "\n".join(f"- {item}" for item in artifact_positive),
        "",
        "Artifact-source-negative classes:",
        "",
        "\n".join(f"- {item}" for item in artifact_negative),
        "",
        "## Experimental Parameters",
        "",
        "- `imgsz=1280`: matches the training configuration and preserves dental panoramic detail.",
        "- `batch=16`: default requested evaluation batch size, applied consistently to both models and all subsets.",
        "- `device=0`: uses the available GPU 0 for deterministic resource selection.",
        "- `conf=0.001`: standard low validation confidence threshold for AP calculation.",
        "- `iou=0.7`: Ultralytics validation NMS IoU setting requested by the plan.",
        "- `max_det=300`: keeps the validator default upper bound for dense dental detections.",
        "- `save_json=True`: keeps standard prediction exports available for auditing.",
        "",
        "## Artifact Stratification",
        "",
        markdown_table(strat_rows, list(strat_rows[0].keys()) if strat_rows else []),
        "",
        "## Detection Metrics",
        "",
        markdown_table(perf_rows, list(perf_rows[0].keys()) if perf_rows else []),
        "",
        "## Relative Performance Drop",
        "",
        markdown_table(drop_rows_out, list(drop_rows_out[0].keys()) if drop_rows_out else []),
        "",
        "## Experimental Analysis",
        "",
        best_model_statement(metrics_df),
        "The relative-drop table reports the change from artifact-absent/negligible to artifact-present images using standard detection metrics only; it is a robustness summary, not a new artifact metric.",
        "",
        "## Important Limitations",
        "",
        "This experiment uses artifact-source-based stratification derived from existing object labels. The artifact-present group indicates that at least one radiopaque restorative, implant-related, endodontic, orthodontic, or metallic structure is annotated in the image. It does not exhaustively capture all image artifacts such as ghost jaw, spine overlay, pharyngeal air-gap, motion blur, exposure artifacts, or other unannotated projection artifacts.",
        "",
        "## Remaining Issues",
        "",
        "- The artifact-source label is derived from current object annotations, so unannotated projection artifacts are not captured.",
        "- Subset difficulty can differ for reasons unrelated to artifact-source presence, such as class mix and object density.",
        "- AP75 depends on validator-provided per-IoU AP values; if unavailable, the script records NaN with a log note.",
        "",
        "## Optimal Solution",
        "",
        "For the current paper-ready experiment, the best solution is to keep this reproducible artifact-source stratification as the main robustness analysis and report only standard detection metrics. A future stronger study can add expert-reviewed artifact annotations or TIDE error analysis as supplementary evidence without changing the main metric definition.",
        "",
        "## Exact Commands Used",
        "",
    ]
    if commands:
        lines.extend(f"- `{command}`" for command in commands)
    else:
        lines.append("- Commands log was not found.")
    lines.extend(
        [
            "",
            "## Paper-ready Wording",
            "",
            "To examine whether the proposed model remains robust in the presence of radiopaque artifact sources, we performed an artifact-source-based stratified evaluation on the test set. Based on the available object annotations, restorative, implant-related, endodontic, orthodontic, and metallic fixation categories were treated as radiopaque artifact-source classes. Images containing at least one such class were assigned to the artifact-present subset, whereas the remaining images were assigned to the artifact-absent/negligible subset. This stratification was used only for robustness analysis and should be interpreted as artifact-source presence rather than exhaustive manual annotation of all imaging artifacts.",
            "",
            "We evaluated each model on the full test set and on the two artifact-source subsets using standard detection metrics, including AP, AP50, AP75, precision, recall, and F1-score. The relative performance drop from the artifact-absent/negligible subset to the artifact-present subset was also reported as a robustness summary derived from the same standard metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    logger = setup_logger(args.out_dir, "summarize_artifact_results.log", args)
    try:
        summary = read_json(args.manifest_summary)
        metrics_df = pd.read_csv(args.metrics)

        strat_rows = stratification_rows(summary)
        perf_rows = performance_rows(metrics_df)
        drops = drop_rows(metrics_df, logger)

        write_table_pair(
            args.out_dir,
            "table_artifact_stratification",
            strat_rows,
            [
                "split",
                "num_images_total",
                "num_images_artifact_absent",
                "num_images_artifact_present",
                "percent_images_artifact_absent",
                "percent_images_artifact_present",
                "num_gt_boxes_total",
                "num_artifact_source_boxes_total",
                "artifact_source_box_percent",
                "top_artifact_source_classes",
            ],
        )
        write_table_pair(
            args.out_dir,
            "table_artifact_stratified_performance",
            perf_rows,
            ["model_name", "subset_name", "num_images", "ap", "ap50", "ap75", "precision", "recall", "f1"],
        )
        write_table_pair(
            args.out_dir,
            "table_relative_performance_drop",
            drops,
            [
                "model_name",
                "relative_drop_ap_percent",
                "relative_drop_ap50_percent",
                "relative_drop_ap75_percent",
                "relative_drop_precision_percent",
                "relative_drop_recall_percent",
                "relative_drop_f1_percent",
            ],
        )

        report_dir = ensure_dir(Path(args.out_dir) / "report")
        (report_dir / "artifact_experiment_summary.md").write_text(
            report_text(args, summary, strat_rows, perf_rows, drops, metrics_df),
            encoding="utf-8",
        )
        logger.info("Summary report written to %s", report_dir / "artifact_experiment_summary.md")
    except Exception:
        logger.exception("Summary failed.")
        raise
    finally:
        finish_logger(logger)


if __name__ == "__main__":
    main()

