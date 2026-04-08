"""
Baseline 2 training script
Admission-level concatenation baseline
"""

import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

from data_loader_baseline2 import get_dataloaders
from model_baseline2 import get_model
from evaluate import evaluate


CONFIG = {
    "mimic_dir": "C:\\Users\\tlsdn\\OneDrive\\Documents\\바탕 화면\\LLM",
    "output_dir": "./checkpoints_baseline2",
    "top_k_codes": 50,
    "batch_size": 4,      # concat는 길어져서 batch 더 작게 잡는 게 안전
    "max_epochs": 3,
    "lr_bert": 2e-5,
    "lr_head": 1e-3,
    "warmup_ratio": 0.1,
    "grad_clip": 1.0,
    "threshold": 0.3,
    "seed": 42,
}


def train():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Device: {device}")

    train_loader, val_loader, test_loader, mlb, code_list = get_dataloaders(
        mimic_dir=Path(CONFIG["mimic_dir"]),
        top_k=CONFIG["top_k_codes"],
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
    )

    num_classes = len(code_list)
    model = get_model(num_classes, device)

    optimizer = optim.AdamW(
        [
            {"params": model.bert.parameters(), "lr": CONFIG["lr_bert"]},
            {"params": model.classifier.parameters(), "lr": CONFIG["lr_head"]},
        ],
        weight_decay=1e-2,
    )

    total_steps = len(train_loader) * CONFIG["max_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    history = []

    print(f"\n▶ Training Baseline 2 for {CONFIG['max_epochs']} epochs | {num_classes} ICD classes\n")

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast():
                loss, _ = model(input_ids, attention_mask, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 20 == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader, device, threshold=CONFIG["threshold"])
        elapsed = time.time() - t0

        print(
            f"\nEpoch {epoch}/{CONFIG['max_epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Micro-F1: {val_metrics['micro_f1']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Time: {elapsed:.1f}s\n"
        )

        history.append({"epoch": epoch, "loss": avg_loss, **val_metrics})

        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  ✓ Best model saved (val micro-F1: {best_val_f1:.4f})\n")

    print("▶ Evaluating on test set...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device, threshold=CONFIG["threshold"])

    print("\n══ TEST RESULTS (Baseline 2 — Admission Concatenation) ══")
    print(f"  Micro F1   : {test_metrics['micro_f1']:.4f}")
    print(f"  Macro F1   : {test_metrics['macro_f1']:.4f}")
    print(f"  Precision@5: {test_metrics['p_at_5']:.4f}")
    print(f"  Precision@8: {test_metrics['p_at_8']:.4f}")
    print("═══════════════════════════════════════════════════════════\n")

    results = {"config": CONFIG, "test_metrics": test_metrics, "history": history}
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"▶ Results saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    train()