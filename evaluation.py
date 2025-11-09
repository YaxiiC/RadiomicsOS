"""Evaluation utilities for the Osteosarcoma project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import TumorDataset
from train import (
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
    MC_EVAL_SEEDS,
    TEST_PATIENT,
    fuse_probs_to_three_classes,
    set_mc_dropout,
    compute_overall_and_perclass,
    print_metrics_block,
)


def evaluate_model(model, artifacts):
    df = artifacts["df"]
    classes = artifacts["classes"]
    rad_df = artifacts["rad_df"]
    feat_names = artifacts["feat_names"]
    rad_stats = artifacts["rad_stats"]
    test_df = artifacts["test_df"]

    if "best_ckpt" in artifacts:
        print(f"[Eval] Using best checkpoint: {artifacts['best_ckpt']}")

    print("\n[Sanity] Test patients:", sorted(test_df["patient"].unique().tolist()))
    assert set(test_df["patient"]) == {TEST_PATIENT}
    print("[Sanity] Per-class counts (TEST):", test_df["label"].value_counts().to_dict())
    print(f"[Evaluation] TEST images (all {TEST_PATIENT}): {len(test_df)}")

    loader = DataLoader(
        TumorDataset(test_df, rad_df, feat_names, rad_stats, train=False),
        batch_size=16,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    model.eval()
    seen_test = 0
    with torch.no_grad():
        for x_img, x_rad, y in loader:
            x_img = x_img.to(DEVICE, non_blocking=True)
            x_rad = x_rad.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits_a, logits_b = model(x_img, x_rad)
            prob = fuse_probs_to_three_classes(logits_a, logits_b)
            y_true += y.cpu().tolist()
            y_pred += prob.argmax(1).cpu().tolist()
            y_prob += prob.cpu().tolist()
            seen_test += y.size(0)

    print(f"[Evaluation] test images processed: {seen_test}/{len(test_df)}")

    y_prob_arr = np.array(y_prob)
    overall, perclass = compute_overall_and_perclass(y_true, y_pred, y_prob_arr, classes)
    print_metrics_block(f"TEST EVALUATION ({TEST_PATIENT})", overall, perclass)

    return overall, perclass


def evaluate_model_5x(
    model,
    artifacts,
    *,
    seeds_eval=MC_EVAL_SEEDS,
    patient_name: str | None = None,
):
    classes = artifacts["classes"]
    rad_df = artifacts["rad_df"]
    feat_names = artifacts["feat_names"]
    rad_stats = artifacts["rad_stats"]
    test_df = artifacts["test_df"]

    pname = patient_name if patient_name is not None else TEST_PATIENT

    print("\n[Sanity] Test patients:", sorted(test_df["patient"].unique().tolist()))
    assert set(test_df["patient"]) == {pname}
    print("[Sanity] Per-class counts (TEST):", test_df["label"].value_counts().to_dict())
    print(f"[Evaluation] TEST images (all {pname}): {len(test_df)}")

    base_loader = DataLoader(
        TumorDataset(test_df, rad_df, feat_names, rad_stats, train=False),
        batch_size=16,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    results = []
    perclass_all = []

    for s in seeds_eval:
        from train import set_seed  # local import avoids circularity at module import

        set_seed(s)
        set_mc_dropout(model, enable=True)

        y_true: list[int] = []
        y_pred: list[int] = []
        y_prob: list[list[float]] = []
        seen_test = 0
        with torch.no_grad():
            for x_img, x_rad, y in base_loader:
                x_img = x_img.to(DEVICE, non_blocking=True)
                x_rad = x_rad.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                logits_a, logits_b = model(x_img, x_rad)
                prob = fuse_probs_to_three_classes(logits_a, logits_b)
                y_true += y.cpu().tolist()
                y_pred += prob.argmax(1).cpu().tolist()
                y_prob += prob.cpu().tolist()
                seen_test += y.size(0)

        print(
            f"[Evaluation seed={s} (MC dropout)] test images: {seen_test}/{len(test_df)}"
        )

        y_prob_arr = np.array(y_prob)
        overall, perclass = compute_overall_and_perclass(y_true, y_pred, y_prob_arr, classes)
        print_metrics_block(
            f"TEST EVALUATION (seed={s}, MC-Dropout, patient={pname})",
            overall,
            perclass,
        )

        results.append(overall)
        perclass_all.append(perclass)

    set_mc_dropout(model, enable=False)

    dfres = pd.DataFrame(results, index=[f"eval_seed_{s}" for s in seeds_eval])
    print("\n=== Overall Summary (Mean ± Std over MC-Dropout passes) ===")
    for col in dfres.columns:
        print(f"{col:>16}: {dfres[col].mean():.4f} ± {dfres[col].std(ddof=1):.4f}")

    metrics = [
        "precision",
        "recall",
        "specificity",
        "sens_at_spec90",
        "spec_at_sens90",
        "f1",
        "auc",
    ]
    rows = []
    for idx, cls in enumerate(classes):
        agg = {m: [perclass_all[k][idx][m] for k in range(len(seeds_eval))] for m in metrics}
        rows.append(
            {
                "Class": cls,
                **{f"{m}_mean": float(np.mean(agg[m])) for m in metrics},
                **{f"{m}_std": float(np.std(agg[m], ddof=1)) for m in metrics},
            }
        )

    dfpc = pd.DataFrame(rows)
    hdr = (
        f"{'Class':<22} "
        f"{'Prec':>10} {'±':>3} "
        f"{'Rec':>10} {'±':>3} "
        f"{'Spec':>10} {'±':>3} "
        f"{'Sens@Spec90':>12} {'±':>3} "
        f"{'Spec@Sens90':>12} {'±':>3} "
        f"{'F1':>10} {'±':>3} "
        f"{'AUC':>10} {'±':>3}"
    )
    print(hdr)
    print("-" * len(hdr))
    for _, row in dfpc.iterrows():
        print(
            f"{row['Class']:<22} "
            f"{row['precision_mean']:>10.3f} ±{row['precision_std']:<5.3f} "
            f"{row['recall_mean']:>10.3f} ±{row['recall_std']:<5.3f} "
            f"{row['specificity_mean']:>10.3f} ±{row['specificity_std']:<5.3f} "
            f"{row['sens_at_spec90_mean']:>12.3f} ±{row['sens_at_spec90_std']:<5.3f} "
            f"{row['spec_at_sens90_mean']:>12.3f} ±{row['spec_at_sens90_std']:<5.3f} "
            f"{row['f1_mean']:>10.3f} ±{row['f1_std']:<5.3f} "
            f"{row['auc_mean']:>10.3f} ±{row['auc_std']:<5.3f}"
        )

    return dfres, dfpc


if __name__ == "__main__":
    print("This module provides evaluation utilities and is not meant to be run standalone.")
