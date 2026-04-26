import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_model(model, loader, save_failures=False, failure_path=None):
    model.eval()

    all_preds = []
    all_labels = []
    failures = []

    for imgs, labels, paths in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        preds = outputs.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # ---- FAILURE CASES ----
        if save_failures:
            for i in range(len(paths)):
                if preds[i] != labels[i]:
                    failures.append({
                        "image": paths[i],
                        "true": int(labels[i].cpu().item()),
                        "pred": int(preds[i].cpu().item())
                    })

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    # save failures
    if save_failures and failure_path:
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_path, index=False)

    return {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }