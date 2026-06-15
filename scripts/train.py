import argparse
import warnings

from ultralytics import YOLO


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom YOLO12 dental model.")
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/12/our_WTconv_DYT.yaml",
        help="Model YAML path.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset YAML path.",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="PDR10_our_WTconv_DYT")
    parser.add_argument("--cache", action="store_true", help="Enable dataset cache.")
    parser.add_argument("--device", default=None, help="GPU device id(s), e.g. 0 or 0,1.")
    parser.add_argument("--close-mosaic", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "cache": args.cache,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "close_mosaic": args.close_mosaic,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)
