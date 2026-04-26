import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image

from dataset import UrbanDataset
from transforms import get_transforms
from model import get_model
from train import train_model
from evaluate import evaluate_model
from corruption import corrupt_pil

DATA_ROOT = "../images"
SPLITS = "../splits"

strategies = ["none", "rotation", "crop", "color", "cutout"]
corruptions = ["clean", "blur", "dark", "rotate", "noise"]

def run_experiments():
    results = []
    detailed_results = {}

    for strategy in strategies:
        print(f"\n=== Training: {strategy} ===")

        train_ds = UrbanDataset(f"{SPLITS}/train.csv", DATA_ROOT, get_transforms(strategy))
        val_ds = UrbanDataset(f"{SPLITS}/val.csv", DATA_ROOT, get_transforms("none"))
        test_ds = UrbanDataset(f"{SPLITS}/test.csv", DATA_ROOT, get_transforms("none"))

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        test_loader = DataLoader(test_ds, batch_size=32)

        model = get_model(5)
        model = train_model(model, train_loader, val_loader)
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), f"saved_models/{strategy}.pth")

        row = {"augmentation": strategy}
        detailed_results[strategy] = {}

        # ---- CLEAN ----
        clean_metrics = evaluate_model(
            model,
            test_loader,
            save_failures=True,
            failure_path=f"failure_cases/{strategy}_clean.csv"
        )
        row["clean"] = clean_metrics["accuracy"]
        detailed_results[strategy]["clean"] = clean_metrics

        # ---- CORRUPTIONS ----
        for corr in corruptions[1:]:
            print(f"Evaluating {corr}...")

            corrupted_imgs = []
            corrupted_labels = []
            corrupted_paths = []

            for _, label, path in test_ds:
                pil_img = Image.open(path).convert("RGB")
                corrupted = corrupt_pil(pil_img, corr)
                corrupted = get_transforms("none")(corrupted)

                corrupted_imgs.append(corrupted)
                corrupted_labels.append(label)
                corrupted_paths.append(path)

            # create loader manually
            dataset = list(zip(corrupted_imgs, corrupted_labels, corrupted_paths))
            loader = DataLoader(dataset, batch_size=32)

            metrics = evaluate_model(
                model,
                loader,
                save_failures=True,
                failure_path=f"failure_cases/{strategy}_{corr}.csv"
            )

            row[corr] = metrics["accuracy"]
            detailed_results[strategy][corr] = metrics

        results.append(row)

    # ---- SAVE RESULTS ----
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv("results/results.tsv", sep="\t", index=False)

    import json
    with open("results/summary.json", "w") as f:
        json.dump(detailed_results, f, indent=4)

    print("\nFinal Results:")
    print(df)