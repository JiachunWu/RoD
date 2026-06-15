import argparse
import os
import warnings

import numpy as np
from prettytable import PrettyTable

from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a trained dental YOLO model.")
    parser.add_argument("--model", required=True, help="Model weight path.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="custom_val")
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def get_weight_size(path):
    stats = os.stat(path)
    return f"{stats.st_size / 1024 / 1024:.1f}"


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.model)
    result = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        save_json=args.save_json,
        project=args.project,
        name=args.name,
    )

    if model.task == "detect":
        length = result.box.p.size
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed["preprocess"]
        inference_time_per_image = result.speed["inference"]
        postprocess_time_per_image = result.speed["postprocess"]
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        _, n_p, _, flops = model_info(model.model)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = [
            "GFLOPs",
            "Parameters",
            "Preprocess/Image",
            "Inference/Image",
            "Postprocess/Image",
            "FPS(All)",
            "FPS(Inference)",
            "Model File Size",
        ]
        model_info_table.add_row(
            [
                f"{flops:.1f}",
                f"{n_p:,}",
                f"{preprocess_time_per_image / 1000:.6f}s",
                f"{inference_time_per_image / 1000:.6f}s",
                f"{postprocess_time_per_image / 1000:.6f}s",
                f"{1000 / all_time_per_image:.2f}",
                f"{1000 / inference_time_per_image:.2f}",
                f"{get_weight_size(args.model)}MB",
            ]
        )
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrics"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx in range(length):
            model_metrice_table.add_row(
                [
                    model_names[idx],
                    f"{result.box.p[idx]:.4f}",
                    f"{result.box.r[idx]:.4f}",
                    f"{result.box.f1[idx]:.4f}",
                    f"{result.box.ap50[idx]:.4f}",
                    f"{result.box.all_ap[idx, 5]:.4f}",
                    f"{result.box.ap[idx]:.4f}",
                ]
            )
        model_metrice_table.add_row(
            [
                "all(mean)",
                f"{result.results_dict['metrics/precision(B)']:.4f}",
                f"{result.results_dict['metrics/recall(B)']:.4f}",
                f"{np.mean(result.box.f1[:length]):.4f}",
                f"{result.results_dict['metrics/mAP50(B)']:.4f}",
                f"{np.mean(result.box.all_ap[:length, 5]):.4f}",
                f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
            ]
        )
        print(model_metrice_table)

        with open(result.save_dir / "paper_data.txt", "w+", errors="ignore", encoding="utf-8") as f:
            f.write(str(model_info_table))
            f.write("\n")
            f.write(str(model_metrice_table))
