import argparse
import csv
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import ensure_dir, finish_logger, read_csv, sanitize_name, setup_logger, str_to_bool, write_csv, write_json


TUNED_COLUMNS = [
    "rank",
    "model_name",
    "model_path",
    "subset_name",
    "data_yaml",
    "num_images",
    "ap",
    "ap50",
    "ap75",
    "precision",
    "recall",
    "f1",
    "imgsz",
    "batch",
    "conf",
    "iou",
    "max_det",
    "agnostic_nms",
    "augment",
    "rect",
    "half",
    "run_tag",
    "device",
    "runtime_seconds",
    "objective",
    "fair_base_ap",
    "fair_base_f1",
    "fair_bspc_ap",
    "fair_bspc_f1",
    "delta_ap_vs_fair_base",
    "delta_f1_vs_fair_base",
    "delta_ap_vs_fair_bspc",
    "delta_f1_vs_fair_bspc",
    "notes",
    "parallel_gpu",
    "parallel_run_dir",
    "parallel_runtime_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Tune BSPC inference parameters without overwriting fair metrics.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="classes30_1280_bspc")
    parser.add_argument("--data-yamls", nargs="+", required=True)
    parser.add_argument("--subset-names", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--imgsz-values", nargs="+", type=int, default=[1280, 1536])
    parser.add_argument("--iou-values", nargs="+", type=float, default=[0.6, 0.65, 0.7, 0.75])
    parser.add_argument("--max-det-values", nargs="+", type=int, default=[300, 500])
    parser.add_argument("--agnostic-nms-values", nargs="+", type=str_to_bool, default=[False])
    parser.add_argument("--augment-values", nargs="+", type=str_to_bool, default=[False, True])
    parser.add_argument("--conf-values", nargs="+", type=float, default=[0.001])
    parser.add_argument("--rect-values", nargs="+", type=str_to_bool, default=[True])
    parser.add_argument("--half-values", nargs="+", type=str_to_bool, default=[False])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fair-metrics", required=True)
    parser.add_argument("--fair-base-name", default="classes30_base_1280n")
    parser.add_argument("--fair-bspc-name", default="classes30_1280_bspc")
    parser.add_argument("--objective-subset", default="all", choices=["all", "artifact_present", "artifact_absent"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-tag-prefix", default="tuned_bspc")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if len(args.data_yamls) != len(args.subset_names):
        raise ValueError("ERROR: --data-yamls and --subset-names must have the same length.")
    if args.objective_subset not in args.subset_names:
        raise ValueError(f"ERROR: objective subset {args.objective_subset!r} not found in --subset-names.")
    for path in [args.model, args.fair_metrics] + list(args.data_yamls):
        if not Path(path).exists():
            raise FileNotFoundError(f"ERROR: path does not exist: {path}")


def build_candidates(args):
    candidates = []
    index = 0
    for imgsz in args.imgsz_values:
        for iou in args.iou_values:
            for max_det in args.max_det_values:
                for augment in args.augment_values:
                    for agnostic_nms in args.agnostic_nms_values:
                        for conf in args.conf_values:
                            for rect in args.rect_values:
                                for half in args.half_values:
                                    index += 1
                                    candidates.append(
                                        {
                                            "candidate_id": f"c{index:03d}",
                                            "imgsz": imgsz,
                                            "iou": iou,
                                            "max_det": max_det,
                                            "agnostic_nms": bool(agnostic_nms),
                                            "augment": bool(augment),
                                            "conf": conf,
                                            "rect": bool(rect),
                                            "half": bool(half),
                                        }
                                    )
    return candidates


def build_tasks(args):
    tasks = []
    for candidate in build_candidates(args):
        for data_yaml, subset_name in zip(args.data_yamls, args.subset_names):
            task = dict(candidate)
            task["data_yaml"] = Path(data_yaml).resolve()
            task["subset_name"] = subset_name
            tasks.append(task)
    return tasks


def run_tag(args, task):
    return (
        f"{args.run_tag_prefix}_{task['candidate_id']}_img{task['imgsz']}_iou{task['iou']}"
        f"_md{task['max_det']}_agn{int(task['agnostic_nms'])}_aug{int(task['augment'])}_conf{task['conf']}"
    )


def task_name(args, task):
    return f"{sanitize_name(run_tag(args, task))}_{sanitize_name(task['subset_name'])}"


def is_valid_metric_row(row):
    if row.get("notes"):
        return False
    for column in ("precision", "recall", "f1", "ap", "ap50", "ap75"):
        try:
            value = float(row.get(column, "nan"))
        except Exception:
            return False
        if not math.isfinite(value):
            return False
    return True


def run_task(args, task, gpu):
    name = task_name(args, task)
    run_out = Path(args.out_dir).resolve() / "runs" / name
    ensure_dir(run_out)
    metrics_path = run_out / "metrics" / "per_run_metrics.csv"
    if args.resume and metrics_path.exists():
        rows = read_csv(metrics_path)
        if len(rows) == 1 and is_valid_metric_row(rows[0]):
            row = rows[0]
            row["parallel_gpu"] = "reused"
            row["parallel_run_dir"] = str(run_out)
            row["parallel_runtime_seconds"] = "0.000"
            return row

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().with_name("03_eval_artifact_subsets.py")),
        "--models",
        str(Path(args.model).resolve()),
        "--model-names",
        args.model_name,
        "--data-yamls",
        str(task["data_yaml"]),
        "--subset-names",
        task["subset_name"],
        "--imgsz",
        str(task["imgsz"]),
        "--batch",
        str(args.batch),
        "--device",
        "0",
        "--conf",
        str(task["conf"]),
        "--iou",
        str(task["iou"]),
        "--max-det",
        str(task["max_det"]),
        "--agnostic-nms",
        str(task["agnostic_nms"]),
        "--augment",
        str(task["augment"]),
        "--rect",
        str(task["rect"]),
        "--half",
        str(task["half"]),
        "--save-json",
        "False",
        "--workers",
        "0",
        "--run-tag",
        run_tag(args, task),
        "--out-dir",
        str(run_out),
    ]
    log_path = run_out / "worker_stdout_stderr.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)
    runtime = time.time() - start
    if completed.returncode != 0 and not metrics_path.exists():
        raise RuntimeError(f"{name} failed on GPU {gpu}; see {log_path}")
    rows = read_csv(metrics_path)
    if len(rows) != 1:
        raise RuntimeError(f"{name} produced {len(rows)} metric rows, expected 1.")
    row = rows[0]
    if completed.returncode != 0 and is_valid_metric_row(row):
        row["notes"] = "valid metrics; worker returned non-zero during shutdown"
    elif completed.returncode != 0:
        raise RuntimeError(f"{name} failed on GPU {gpu}; see {log_path}")
    row["parallel_gpu"] = str(gpu)
    row["parallel_run_dir"] = str(run_out)
    row["parallel_runtime_seconds"] = f"{runtime:.3f}"
    return row


def split_tasks_by_gpu(tasks, gpus):
    groups = {gpu: [] for gpu in gpus}
    for index, task in enumerate(tasks):
        groups[gpus[index % len(gpus)]].append(task)
    return groups


def gpu_worker(args, gpu, tasks, logger):
    rows = []
    for task in tasks:
        logger.info("Running %s on GPU %s", task_name(args, task), gpu)
        rows.append(run_task(args, task, gpu))
        logger.info("Completed %s on GPU %s", task_name(args, task), gpu)
    return rows


def metric_lookup(fair_rows, model_name, subset_name, metric):
    for row in fair_rows:
        if row.get("model_name") == model_name and row.get("subset_name") == subset_name:
            try:
                return float(row.get(metric, "nan"))
            except Exception:
                return float("nan")
    return float("nan")


def enrich_and_rank(rows, args):
    fair_rows = read_csv(args.fair_metrics)
    enriched = []
    for row in rows:
        subset = row["subset_name"]
        ap = float(row["ap"])
        f1 = float(row["f1"])
        fair_base_ap = metric_lookup(fair_rows, args.fair_base_name, subset, "ap")
        fair_base_f1 = metric_lookup(fair_rows, args.fair_base_name, subset, "f1")
        fair_bspc_ap = metric_lookup(fair_rows, args.fair_bspc_name, subset, "ap")
        fair_bspc_f1 = metric_lookup(fair_rows, args.fair_bspc_name, subset, "f1")
        out = dict(row)
        out["fair_base_ap"] = f"{fair_base_ap:.6f}" if math.isfinite(fair_base_ap) else ""
        out["fair_base_f1"] = f"{fair_base_f1:.6f}" if math.isfinite(fair_base_f1) else ""
        out["fair_bspc_ap"] = f"{fair_bspc_ap:.6f}" if math.isfinite(fair_bspc_ap) else ""
        out["fair_bspc_f1"] = f"{fair_bspc_f1:.6f}" if math.isfinite(fair_bspc_f1) else ""
        out["delta_ap_vs_fair_base"] = f"{ap - fair_base_ap:.6f}" if math.isfinite(fair_base_ap) else ""
        out["delta_f1_vs_fair_base"] = f"{f1 - fair_base_f1:.6f}" if math.isfinite(fair_base_f1) else ""
        out["delta_ap_vs_fair_bspc"] = f"{ap - fair_bspc_ap:.6f}" if math.isfinite(fair_bspc_ap) else ""
        out["delta_f1_vs_fair_bspc"] = f"{f1 - fair_bspc_f1:.6f}" if math.isfinite(fair_bspc_f1) else ""
        objective = ap if subset == args.objective_subset else float("nan")
        out["objective"] = f"{objective:.6f}" if math.isfinite(objective) else ""
        enriched.append(out)

    objective_rows = [row for row in enriched if row["subset_name"] == args.objective_subset]
    objective_rows.sort(key=lambda row: (float(row["ap"]), float(row["f1"]), float(row["ap50"])), reverse=True)
    rank_by_tag = {row["run_tag"]: index + 1 for index, row in enumerate(objective_rows)}
    for row in enriched:
        row["rank"] = rank_by_tag.get(row["run_tag"], "")
    enriched.sort(key=lambda row: (int(row["rank"]) if row["rank"] else 999999, row["run_tag"], row["subset_name"]))
    return enriched


def write_outputs(args, rows):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "metrics")
    ranked = enrich_and_rank(rows, args)
    write_csv(out_dir / "metrics" / "tuned_search_metrics.csv", ranked, TUNED_COLUMNS)
    write_json(out_dir / "metrics" / "tuned_search_metrics.json", ranked)
    best = [row for row in ranked if row.get("rank") == 1 or row.get("rank") == "1"]
    write_csv(out_dir / "metrics" / "best_tuned_candidate_rows.csv", best, TUNED_COLUMNS)
    write_json(out_dir / "metrics" / "best_tuned_candidate_rows.json", best)


def main():
    args = parse_args()
    logger = setup_logger(args.out_dir, "tune_bspc_inference.log", args)
    rows = []
    try:
        validate_args(args)
        tasks = build_tasks(args)
        logger.info("Tuned search tasks: %d", len(tasks))
        logger.info("GPUs: %s", args.gpus)
        groups = split_tasks_by_gpu(tasks, args.gpus)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = {}
            for gpu, gpu_tasks in groups.items():
                logger.info("Submitting %d tasks to GPU %s", len(gpu_tasks), gpu)
                futures[executor.submit(gpu_worker, args, gpu, gpu_tasks, logger)] = gpu
            for future in as_completed(futures):
                gpu = futures[future]
                gpu_rows = future.result()
                rows.extend(gpu_rows)
                logger.info("GPU %s completed %d tasks", gpu, len(gpu_rows))
                write_outputs(args, rows)
        write_outputs(args, rows)
        logger.info("Tuned search rows written: %d", len(rows))
    except Exception:
        logger.exception("Tuned search failed.")
        raise
    finally:
        finish_logger(logger)


if __name__ == "__main__":
    main()
