import argparse
import shutil
from pathlib import Path

from common import ensure_dir, finish_logger, load_yaml, normalize_names, read_csv, resolve_dataset_root, setup_logger, write_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Create artifact-source subset image lists and YOLO data YAMLs.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--manifest", required=True, help="Path to artifact manifest CSV for the selected split.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    return parser.parse_args()


def write_image_list(path, image_paths):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for image_path in image_paths:
            f.write(f"{image_path}\n")


def prepare_eval_view(rows, out_dir, split, logger):
    view_root = Path(out_dir).resolve() / "dataset_view"
    image_dir = ensure_dir(view_root / "images" / split)
    label_dir = ensure_dir(view_root / "labels" / split)
    seen_names = {}
    eval_rows = []

    for row in rows:
        src_image = Path(row["image_path"])
        if src_image.name in seen_names and seen_names[src_image.name] != str(src_image):
            raise ValueError(f"ERROR: duplicate image filename in split view: {src_image.name}")
        seen_names[src_image.name] = str(src_image)

        dst_image = image_dir / src_image.name
        if dst_image.exists() or dst_image.is_symlink():
            if not dst_image.is_symlink() or dst_image.resolve() != src_image.resolve():
                raise ValueError(f"ERROR: evaluation view image path already exists and is not the expected symlink: {dst_image}")
        else:
            dst_image.symlink_to(src_image)

        dst_label = label_dir / f"{src_image.stem}.txt"
        src_label = Path(row["label_path"])
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")

        eval_row = dict(row)
        eval_row["eval_image_path"] = str(dst_image)
        eval_rows.append(eval_row)

    logger.info("Evaluation view root: %s", view_root)
    logger.info("Evaluation view images prepared: %d", len(eval_rows))
    return eval_rows


def make_subset_yaml(original_data, dataset_root, split_txt_path, split):
    subset_data = {
        "path": str(dataset_root),
        "train": original_data.get("train"),
        "val": original_data.get("val"),
        "test": str(Path(split_txt_path).resolve()),
        "names": original_data["names"],
    }
    if split != "test":
        subset_data[split] = str(Path(split_txt_path).resolve())
    return subset_data


def main():
    args = parse_args()
    logger = setup_logger(args.out_dir, "make_artifact_subset_yamls.log", args)
    try:
        data_yaml = Path(args.data).resolve()
        data = load_yaml(data_yaml)
        if "names" not in data:
            raise ValueError("ERROR: data.yaml does not contain key 'names'.")
        normalize_names(data["names"])
        dataset_root = resolve_dataset_root(data_yaml, data)
        logger.info("Resolved dataset root: %s", dataset_root)

        rows = [row for row in read_csv(args.manifest) if row["split"] == args.split]
        if not rows:
            raise ValueError(f"ERROR: manifest has no rows for split {args.split!r}.")
        rows = sorted(rows, key=lambda row: row["image_path"])
        rows = prepare_eval_view(rows, args.out_dir, args.split, logger)

        all_images = [row["eval_image_path"] for row in rows]
        absent_images = [row["eval_image_path"] for row in rows if int(row["artifact_source_label"]) == 0]
        present_images = [row["eval_image_path"] for row in rows if int(row["artifact_source_label"]) == 1]
        if not absent_images:
            raise ValueError("ERROR: artifact_absent subset is empty.")
        if not present_images:
            raise ValueError("ERROR: artifact_present subset is empty.")

        split_dir = ensure_dir(Path(args.out_dir) / "splits")
        yaml_dir = ensure_dir(Path(args.out_dir) / "data_yaml")

        split_files = {
            "all": split_dir / f"{args.split}_all.txt",
            "artifact_absent": split_dir / f"{args.split}_artifact_absent.txt",
            "artifact_present": split_dir / f"{args.split}_artifact_present.txt",
        }
        write_image_list(split_files["all"], all_images)
        write_image_list(split_files["artifact_absent"], absent_images)
        write_image_list(split_files["artifact_present"], present_images)

        for subset_name, split_file in split_files.items():
            write_yaml(yaml_dir / f"{args.split}_{subset_name}.yaml", make_subset_yaml(data, dataset_root, split_file, args.split))

        logger.info("num_%s_all=%d", args.split, len(all_images))
        logger.info("num_%s_artifact_absent=%d", args.split, len(absent_images))
        logger.info("num_%s_artifact_present=%d", args.split, len(present_images))
    except Exception:
        logger.exception("Subset YAML build failed.")
        raise
    finally:
        finish_logger(logger)


if __name__ == "__main__":
    main()
