"""Forenklet utdrag av ChessCams LeYOLO-trening.

Bruker ferdig merkede YOLO-datasett og eksisterende LeYOLO-vekter.
Treningsinnstillingene og vektingen av datakilder følger real-v2-kjøringen.
Nedlasting, rådatakonvertering og Modal-oppsett er utelatt.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

CLASSES = [
    "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rook",
    "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rook",
]
# Navn på klargjort datasett, gjentakelser i trening og ved modellvalg.
DATASETS = [("upstream", 1, 1), ("custom", 15, 10),
            ("external-v1", 4, 10), ("real-v2", 4, 10)]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def prepare_split(source: Path, destination: Path, split: str, repeat: int) -> Path:
    """Gi små datakilder større vekt uten å endre hvilke bilder de inneholder."""
    images = sorted(p for p in (source / "images" / split).rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
    if not images:
        raise ValueError(f"Ingen bilder: {source / 'images' / split}")
    output_images = destination / "images" / split
    output_labels = destination / "labels" / split
    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)
    for number, image in enumerate(images):
        relative = image.relative_to(source / "images" / split).with_suffix(".txt")
        label = source / "labels" / split / relative
        if not label.is_file():
            raise FileNotFoundError(f"Mangler merking: {label}")
        for repetition in range(repeat):
            name = f"{number:06d}_{repetition:02d}"
            for original, target in [(image, output_images / f"{name}{image.suffix.lower()}"),
                                     (label, output_labels / f"{name}.txt")]:
                try:
                    os.link(original, target)
                except OSError:
                    shutil.copyfile(original, target)
    return output_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Mappe med klargjorte datasett")
    parser.add_argument("--weights", type=Path, required=True, help="Startmodell i .pt-format")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cpu", help="cpu, mps eller CUDA-enhet 0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Vis oppsettet uten å trene")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error("epochs og batch må være positive")

    settings = dict(
        epochs=args.epochs, batch=args.batch, device=args.device, imgsz=480,
        optimizer="AdamW", lr0=0.00005, lrf=0.1, weight_decay=0.0005,
        warmup_epochs=0.0, patience=6, seed=20260910, deterministic=True,
        hsv_h=0.01, hsv_s=0.35, hsv_v=0.25, degrees=3.0,
        translate=0.05, scale=0.25, perspective=0.0005,
        flipud=0.0, fliplr=0.5, mosaic=0.0, mixup=0.0,
        workers=0, amp=args.device.isdigit(), cache=False,
        project=str(args.output.resolve()), name="pieces", plots=True,
    )
    if args.dry_run:
        print(json.dumps({"settings": settings, "datasets": DATASETS}, indent=2))
        return

    import ultralytics
    import yaml
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.1.4":
        raise ValueError("Installer LeYOLO-versjonen i requirements.txt i et eget miljø.")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    with tempfile.TemporaryDirectory(prefix="chesscam-training-") as temporary:
        temporary = Path(temporary)
        train_sources, val_sources = [], []
        for name, train_repeat, val_repeat in DATASETS:
            source = args.data.resolve() / name
            for split, repeat, target in [("train", train_repeat, train_sources),
                                          ("val", val_repeat, val_sources)]:
                target.append(str(prepare_split(source, temporary / name, split, repeat)))
        dataset_yaml = temporary / "combined.yaml"
        dataset_yaml.write_text(yaml.safe_dump({
            "train": train_sources, "val": val_sources, "names": CLASSES,
        }, sort_keys=False))

        # Biblioteket beregner tap og gradienter, oppdaterer vektene og lagrer beste modell.
        model = YOLO(str(args.weights))
        model.train(data=str(dataset_yaml), **settings)
        output = Path(model.trainer.save_dir)
        best = YOLO(str(output / "weights/best.pt"))

        # Evaluer hver kilde separat, uten gjentakelsene som brukes ved modellvalg.
        validation = {}
        for name, _, _ in DATASETS:
            val_images = str(args.data.resolve() / name / "images/val")
            val_yaml = temporary / f"{name}.yaml"
            val_yaml.write_text(yaml.safe_dump({
                "train": val_images, "val": val_images, "names": CLASSES,
            }, sort_keys=False))
            metrics = best.val(data=str(val_yaml), imgsz=480, batch=args.batch,
                               device=args.device, workers=0, plots=False,
                               project=str(output), name=f"{name}-val")
            validation[name] = {key: float(value) for key, value in metrics.results_dict.items()}
        (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(f"Beste modell: {output / 'weights/best.pt'}")


if __name__ == "__main__":
    main()
