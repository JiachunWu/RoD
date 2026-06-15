import argparse
import csv
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import METRIC_COLUMNS, ensure_dir, finish_logger, read_csv, sanitize_name, setup_logger, str_to_bool, write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run one-model one-subset evaluations in parallel across GPUs.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--model-names", nargs="+", required=True)
    parser.add_argument("--data-yamls", nargs="+", required=True)
    parser.add_argument("--subset-names", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
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
    parser.add_argument("--resume", action="store_true", help="Reuse completed one-row worker metrics when present.")
    return parser.parse_args()


def validate_args(args):
    if len(args.models) != len(args.model_names):
        raise ValueError("ERROR: number of model names does not match number of model paths.")
    if len(args.data_yamls) != len(args.subset_names):
        raise ValueError("ERROR: number of subset names does not match number of data YAML paths.")
    if not args.gpus:
        raise ValueError("ERROR: at least one GPU id is required.")
    for path in list(args.models) + list(args.data_yamls):
        if not Path(path).exists():
            raise FileNotFoundError(f"ERROR: path does not exist: {path}")


def build_tasks(args):
    tasks = []
    for model_path, model_name in zip(args.models, args.model_names):
        for data_yaml, subset_name in zip(args.data_yamls, args.subset_names):
            tasks.append(
                {
                    "model_path": Path(model_path).resolve(),
                    "model_name": model_name,
                    "data_yaml": Path(data_yaml).resolve(),
                    "subset_name": subset_name,
                }
            )
    return tasks


def run_task(task, gpu, args):
    task_name = f"{sanitize_name(task['model_name'])}_{sanitize_name(task['subset_name'])}"
    run_out = Path(args.out_dir).resolve() / "parallel_fair_runs" / task_name
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
        str(task["model_path"]),
        "--model-names",
        task["model_name"],
        "--data-yamls",
        str(task["data_yaml"]),
        "--subset-names",
        task["subset_name"],
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        "0",
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        str(args.max_det),
        "--agnostic-nms",
        str(args.agnostic_nms),
        "--augment",
        str(args.augment),
        "--rect",
        str(args.rect),
        "--half",
        str(args.half),
        "--save-json",
        str(args.save_json),
        "--workers",
        str(args.workers),
        "--run-tag",
        args.run_tag,
        "--out-dir",
        str(run_out),
    ]
    start = time.time()
    log_path = run_out / "worker_stdout_stderr.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)
    runtime = time.time() - start
    if completed.returncode != 0:
        raise RuntimeError(f"{task_name} failed on GPU {gpu}; see {log_path}")
    rows = read_csv(metrics_path)
    if len(rows) != 1:
        raise RuntimeError(f"{task_name} produced {len(rows)} metric rows, expected 1.")
    row = rows[0]
    row["parallel_gpu"] = str(gpu)
    row["parallel_run_dir"] = str(run_out)
    row["parallel_runtime_seconds"] = f"{runtime:.3f}"
    return row


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


def write_merged_metrics(out_dir, rows):
    metrics_dir = ensure_dir(Path(out_dir) / "metrics")
    ordered = sorted(rows, key=lambda row: (row["model_name"], row["subset_name"]))
    columns = list(METRIC_COLUMNS)
    metric_rows = [{column: row.get(column, "") for column in columns} for row in ordered]
    write_csv(metrics_dir / "per_run_metrics.csv", metric_rows, columns)
    write_csv(metrics_dir / "combined_subset_metrics.csv", metric_rows, columns)
    write_json(metrics_dir / "per_run_metrics.json", metric_rows)
    write_parallel_manifest(metrics_dir / "parallel_fair_run_manifest.csv", rows)


def write_parallel_manifest(path, rows):
    columns = list(METRIC_COLUMNS) + ["parallel_gpu", "parallel_run_dir", "parallel_runtime_seconds"]
    path = Path(path)
    ensure_dir(path.parent)
    ordered = sorted(rows, key=lambda row: (row["model_name"], row["subset_name"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)


def gpu_worker(gpu, tasks, args, logger):
    rows = []
    for task in tasks:
        logger.info("Running %s/%s on GPU %s", task["model_name"], task["subset_name"], gpu)
        rows.append(run_task(task, gpu, args))
        logger.info("Completed %s/%s on GPU %s", task["model_name"], task["subset_name"], gpu)
    return rows


def split_tasks_by_gpu(tasks, gpus):
    task_groups = {gpu: [] for gpu in gpus}
    for index, task in enumerate(tasks):
        task_groups[gpus[index % len(gpus)]].append(task)
    return task_groups


def main():
    args = parse_args()
    logger = setup_logger(args.out_dir, "parallel_fair_eval.log", args)
    rows = []
    try:
        validate_args(args)
        tasks = build_tasks(args)
        logger.info("Parallel fair eval tasks: %d", len(tasks))
        logger.info("GPUs: %s", args.gpus)

        task_groups = split_tasks_by_gpu(tasks, args.gpus)
        future_to_gpu = {}
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            for gpu, gpu_tasks in task_groups.items():
                logger.info("Submitting %d tasks to GPU %s", len(gpu_tasks), gpu)
                future = executor.submit(gpu_worker, gpu, gpu_tasks, args, logger)
                future_to_gpu[future] = gpu
            for future in as_completed(future_to_gpu):
                gpu = future_to_gpu[future]
                gpu_rows = future.result()
                rows.extend(gpu_rows)
                logger.info("GPU %s completed %d tasks", gpu, len(gpu_rows))
                write_merged_metrics(args.out_dir, rows)

        expected = len(args.models) * len(args.data_yamls)
        if len(rows) != expected:
            raise RuntimeError(f"ERROR: expected {expected} metric rows, got {len(rows)}.")
        write_merged_metrics(args.out_dir, rows)
        logger.info("Merged fair metrics rows written: %d", len(rows))
    except Exception:
        logger.exception("Parallel fair evaluation failed.")
        raise
    finally:
        finish_logger(logger)


if __name__ == "__main__":
    main()
