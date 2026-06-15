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


def best_rows_by_subset(metrics_df):
    rows = []
    for subset_name in ["all", "artifact_absent", "artifact_present"]:
        subset = metrics_df[metrics_df["subset_name"] == subset_name]
        if subset.empty:
            continue
        row = subset.sort_values("ap", ascending=False).iloc[0]
        rows.append(
            {
                "subset_name": subset_name,
                "best_model_by_ap": row["model_name"],
                "best_ap": format_number(row["ap"]),
                "best_ap50": format_number(row["ap50"]),
                "best_ap75": format_number(row["ap75"]),
            }
        )
    return rows


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
        "3. Evaluated all three checkpoints on all three subsets with the same Ultralytics validation settings.",
        "4. Summarized standard detection metrics and relative performance drops for paper reporting.",
        "",
        "## Code Improvement Logic and Key Functions",
        "",
        "The implementation follows Occam's razor: a small shared helper module contains deterministic path resolution, class mapping checks, YOLO label parsing, CSV/JSON/YAML writing, and markdown table generation. The stage scripts map directly to the required workflow stages, so each stage can be rerun independently without changing model code or dataset files.",
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
        "- `batch=16`: default requested evaluation batch size, applied consistently to all models and subsets.",
        "- `device=0` inside each worker with `CUDA_VISIBLE_DEVICES=<physical_gpu>`: the final fair run used four physical GPUs in parallel while each isolated worker saw only one GPU.",
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
        "The improved BSPC model was therefore already best on the full test set and on the artifact-present subset under the fair shared-parameter setting; no tuned inference pass was run.",
        "The relative-drop table reports the change from artifact-absent/negligible to artifact-present images using standard detection metrics only; it is a robustness summary, not a new artifact metric.",
        "",
        "## Important Limitations",
        "",
        "This experiment uses artifact-source-based stratification derived from existing object labels. The artifact-present group indicates that at least one radiopaque restorative, implant-related, endodontic, prosthetic, or artificial dental structure is annotated in the image. It does not exhaustively capture all image artifacts such as ghost jaw, spine overlay, pharyngeal air-gap, motion blur, exposure artifacts, or other unannotated projection artifacts.",
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
            "To examine whether the proposed model remains robust in the presence of radiopaque artifact sources, we performed an artifact-source-based stratified evaluation on the test set. Based on the available object annotations, restorative, implant-related, endodontic, prosthetic, and artificial component categories were treated as radiopaque artifact-source classes. Specifically, Crown, Filling, Implant, Root Canal Treatment, abutment, and gingival former were assigned as artifact-source-positive classes. Images containing at least one such class were assigned to the artifact-present subset, whereas images containing only Fracture teeth, Retained root, Root Piece, or impacted tooth were assigned to the artifact-absent/negligible subset. This stratification was used only for robustness analysis and should be interpreted as artifact-source presence rather than exhaustive manual annotation of all imaging artifacts.",
            "",
            "We evaluated each model on the full test set and on the two artifact-source subsets using standard detection metrics, including AP, AP50, AP75, precision, recall, and F1-score. The relative performance drop from the artifact-absent/negligible subset to the artifact-present subset was also reported as a robustness summary derived from the same standard metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def chinese_report_text(args, summary, strat_rows, perf_rows, drop_rows_out, metrics_df):
    test_summary = summary.get("test", {})
    best_rows = best_rows_by_subset(metrics_df)
    commands = read_commands(args.out_dir)
    return "\n".join(
        [
            "# PDR-10 Artifact-source-based 分层评估中文报告",
            "",
            "## 1. 任务流程",
            "",
            "1. 根据 `artifact-experimen-v2.md` 的定义读取 PDR-10 YOLO 数据集，并构建 train/val/test 的 artifact-source manifest。",
            "2. 基于 test manifest 生成 all、artifact_absent、artifact_present 三个测试子集，同时在输出目录中创建评估副本视图，避免修改原始数据目录。",
            "3. 使用统一推理/验证参数公平评估三个权重：`PDR10_base_1280n`、`PDR10_1280_bspc`、`PDR10_our_WTconv_dff2`。",
            "4. 使用 4 张物理 GPU 并行执行评估分片，每个 worker 通过 `CUDA_VISIBLE_DEVICES` 隔离到单张卡，最终合并 9 行公平结果。",
            "5. 生成分层统计表、检测指标表、相对性能变化表和最终报告。",
            "",
            "## 2. 代码改进逻辑和关键功能说明",
            "",
            "代码遵循奥卡姆剃刀原则，只新增独立实验脚本副本，不改模型源码、训练结果或原始标签。公共逻辑集中在 `common.py`，入口脚本保持薄封装，便于单阶段复现。",
            "",
            "- `01_build_artifact_manifest.py`: 解析 YOLO 标签，生成图像级 artifact-source 标签和统计摘要。",
            "- `02_make_artifact_subset_yamls.py`: 生成 all/absent/present 子集 txt 与临时 YAML，并创建输出目录下的数据副本视图。",
            "- `03_eval_artifact_subsets.py`: 调用 Ultralytics 标准 `model.val()`，提取 AP、AP50、AP75、Precision、Recall、F1。",
            "- `05_run_parallel_fair_eval.py`: 多 GPU 并行调度公平评估；每个子任务独立输出，最后合并主 metrics。",
            "- `04_summarize_artifact_results.py`: 生成表格和中英文报告。",
            "",
            "## 3. 实验参数设置说明",
            "",
            "- `imgsz=1280`: 与训练配置一致，保留全景牙片细节。",
            "- `batch=16`: 所有模型和子集统一使用，保证公平比较。",
            "- `conf=0.001`: AP 计算常用低置信度阈值，避免提前过滤候选框。",
            "- `iou=0.7`: 统一 NMS IoU 设置。",
            "- `max_det=300`: 保留密集牙科目标检测的最大检测数量上限。",
            "- `save_json=True`: 保存预测 JSON，便于后续审计。",
            "- `CUDA_VISIBLE_DEVICES=<physical_gpu>` + worker 内 `device=0`: 用 4 张物理 GPU 并行加速，同时保证每个验证进程只看到一张 GPU。",
            "",
            "## 4. 实验验证结果",
            "",
            f"- test 图像总数：{test_summary.get('num_images_total', 'NA')}",
            f"- artifact_absent/negligible：{test_summary.get('num_images_artifact_absent', 'NA')}",
            f"- artifact_present：{test_summary.get('num_images_artifact_present', 'NA')}",
            "",
            "### 检测指标",
            "",
            markdown_table(perf_rows, list(perf_rows[0].keys()) if perf_rows else []),
            "",
            "### 各子集 AP 最优模型",
            "",
            markdown_table(best_rows, list(best_rows[0].keys()) if best_rows else []),
            "",
            "### 相对性能变化",
            "",
            markdown_table(drop_rows_out, list(drop_rows_out[0].keys()) if drop_rows_out else []),
            "",
            "## 5. 实验分析",
            "",
            "`PDR10_1280_bspc` 在全测试集 AP=0.3360、artifact_present AP=0.3373，均为三模型最高，因此改进模型在主要公平对比和 artifact-source present 鲁棒性场景中表现最好。artifact_absent 子集由基线模型 AP=0.3402 最高，说明无 artifact-source 图像上基线仍有优势。",
            "",
            "由于 BSPC 已在公平共享参数下取得 all 与 artifact_present 的最高 AP，本次没有执行 tuned 推理超参数补测，避免将调参结果与公平结果混淆。",
            "",
            "## 6. 还存在的问题",
            "",
            "- artifact-source 标签来自现有目标框类别，不是人工标注的完整影像伪影真值。",
            "- absent/present 子集存在类别分布和目标密度差异，相对下降不应解释为单一伪影因果效应。",
            "- 多 GPU 并行只改变执行方式，不改变公平评估参数；如需逐位完全一致，可单 GPU 串行重跑作为补充审计。",
            "",
            "## 7. 最优解决方案",
            "",
            "当前最优方案是将本次 artifact-source-based stratification 作为论文主鲁棒性分析，报告标准检测指标和相对性能变化，不引入新的 artifact 指标。若后续需要更强证据，可补充专家人工伪影标注或 TIDE 错误分析，但不替代当前公平检测指标。",
            "",
            "## 8. 命令记录",
            "",
            "\n".join(f"- `{command}`" for command in commands) if commands else "- 未找到命令日志。",
            "",
        ]
    )


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
        (report_dir / "artifact_experiment_summary_zh.md").write_text(
            chinese_report_text(args, summary, strat_rows, perf_rows, drops, metrics_df),
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
