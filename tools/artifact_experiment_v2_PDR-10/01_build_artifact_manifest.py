import argparse
from pathlib import Path

from common import (
    ARTIFACT_SOURCE_CLASS_IDS,
    MANIFEST_COLUMNS,
    build_summary,
    class_mapping_json,
    collect_images,
    ensure_dir,
    finish_logger,
    load_yaml,
    make_manifest_row,
    normalize_names,
    resolve_dataset_root,
    resolve_split_path,
    run_sample_label_tests,
    setup_logger,
    str_to_bool,
    validate_artifact_mapping,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build artifact-source-based image manifest from YOLO labels.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml or data(1).yaml.")
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--strict-labels", type=str_to_bool, default=False)
    return parser.parse_args()


def log_split_summary(logger, split, rows, summary):
    item = summary[split]
    logger.info(
        "%s: images=%d, artifact_absent=%d (%.2f%%), artifact_present=%d (%.2f%%)",
        split,
        item["num_images_total"],
        item["num_images_artifact_absent"],
        item["percent_images_artifact_absent"],
        item["num_images_artifact_present"],
        item["percent_images_artifact_present"],
    )
    logger.info("%s: missing_label_files=%d, empty_label_files=%d", split, item["missing_label_files"], item["empty_label_files"])
    logger.info("%s: top 10 classes by GT box count=%s", split, item["top_10_classes_by_gt_box_count"])
    logger.info("%s: top artifact-source classes by GT box count=%s", split, item["top_artifact_source_classes_by_gt_box_count"])


def main():
    args = parse_args()
    logger = setup_logger(args.out_dir, "build_artifact_manifest.log", args)
    try:
        validate_artifact_mapping()
        run_sample_label_tests()
        logger.info("Artifact mapping and sample-label tests passed.")

        data_yaml = Path(args.data).resolve()
        data = load_yaml(data_yaml)
        if "names" not in data:
            raise ValueError("ERROR: data.yaml does not contain key 'names'.")
        class_names = normalize_names(data["names"])
        dataset_root = resolve_dataset_root(data_yaml, data)
        logger.info("Resolved dataset root: %s", dataset_root)
        logger.info("Artifact-source-positive class ids: %s", sorted(ARTIFACT_SOURCE_CLASS_IDS))

        all_rows = []
        manifest_dir = ensure_dir(Path(args.out_dir) / "manifests")
        for split in args.splits:
            split_path = resolve_split_path(data_yaml, data, split)
            logger.info("Resolved %s image path: %s", split, split_path)
            images = collect_images(split_path)
            logger.info("%s: processing %d images", split, len(images))
            rows = [make_manifest_row(split, image_path, class_names, strict_labels=args.strict_labels) for image_path in images]
            write_csv(manifest_dir / f"artifact_manifest_{split}.csv", rows, MANIFEST_COLUMNS)
            all_rows.extend(rows)

        all_rows = sorted(all_rows, key=lambda row: (row["split"], row["image_path"]))
        write_csv(manifest_dir / "artifact_manifest_all.csv", all_rows, MANIFEST_COLUMNS)

        stats_dir = ensure_dir(Path(args.out_dir) / "stats")
        write_json(stats_dir / "artifact_class_mapping.json", class_mapping_json(class_names))
        summary = build_summary(all_rows, class_names)
        write_json(stats_dir / "artifact_manifest_summary.json", summary)

        for split in args.splits:
            log_split_summary(logger, split, [row for row in all_rows if row["split"] == split], summary)
        log_split_summary(logger, "all", all_rows, summary)

        warnings = [row for row in all_rows if row["warning"]]
        if warnings:
            logger.warning("Warnings found in manifest rows: %d", len(warnings))
        logger.info("Processed images total: %d", len(all_rows))
    except Exception:
        logger.exception("Manifest build failed.")
        raise
    finally:
        finish_logger(logger)


if __name__ == "__main__":
    main()
