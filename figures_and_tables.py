#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    ndcg_score,
)

# --- Configuration ---
MODEL_DIR = Path("/home/naeppsa1/release/pubchem-jobs/models")
DATA_DIR = Path("/home/naeppsa1/release/frequent-hitters/data/processed/")
OUTPUT_DIR = Path("./analysis_results")
SEEDS = [1, 2, 3, 4, 5]
THRESHOLDS_PCT = ["50", "60", "70", "80", "90", "95"]
ASSAY_MODE = "biochemical"  # set to "cellular" for cellular assays
ASSAY_SHORT_NAMES = {"biochemical": "bio", "cellular": "cell"}
ASSAY_SHORT = ASSAY_SHORT_NAMES.get(ASSAY_MODE)
if ASSAY_SHORT is None:
    raise ValueError(f"Unsupported assay mode: {ASSAY_MODE}")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting Style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# --- Metric Functions ---

def precision_at_recall_90(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Max precision where recall >= 0.90."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # Filter for recall >= 0.90
    mask = recall >= 0.90
    if not np.any(mask):
        return 0.0
    return float(np.max(precision[mask]))


def calculate_ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int = None) -> float:
    """
    Normalized Discounted Cumulative Gain.
    y_true: Continuous relevance scores (Hit Rates).
    y_score: Predicted scores.
    k: Cutoff (e.g., look at top k results). If None, uses all.
    """
    # sklearn ndcg_score expects shape (n_samples, n_labels)
    # We treat the entire test set as one "query" to rank.
    return ndcg_score([y_true], [y_score], k=k)

def continuous_lift(y_true: np.ndarray, y_score: np.ndarray, top_frac: float = 0.01) -> float:
    """
    Continuous version of Enrichment Factor.
    Returns: (Mean Hit Rate of Top X%) / (Global Mean Hit Rate)
    """
    n = len(y_true)
    k = max(1, int(np.ceil(n * top_frac)))
    
    # Sort indices by predicted score descending
    order = np.argsort(y_score)[::-1][:k]
    
    # Mean hit rate of the top k predicted compounds
    mean_top = y_true[order].mean()
    global_mean = y_true.mean()
    
    if global_mean == 0:
        return np.nan
    return float(mean_top / global_mean)

# --- Data Loading Helpers ---

def load_thresholds(mode: str) -> Dict:
    """Loads the percentile threshold values from JSON."""
    path = DATA_DIR / mode / f"{mode}_thresholds.json"
    with open(path) as f:
        return json.load(f)["percentiles_by_seed"]

def prevalence_weighted_average(lf: pl.LazyFrame, metadata: pl.DataFrame) -> pl.LazyFrame:
    """
    Computes simple mean and prevalence-weighted mean for multitask predictions.
    """
    # All assay columns are those that are not 'smiles'
    assay_cols = [c for c in lf.collect_schema().names() if c != "smiles"]

    # 1. Create a Struct mapping Assay_ID -> N_Screens
    weights_map = {
        row["assay_id"]: row["n_screens"] 
        for row in metadata.select(["assay_id", "n_screens"]).iter_rows(named=True)
        if row["assay_id"] in assay_cols 
    }
    
    # Filter assay_cols to those we actually have metadata for
    valid_assay_cols = [c for c in assay_cols if c in weights_map]
    
    if not valid_assay_cols:
        return lf.select(["smiles"]).with_columns(
            mean=pl.lit(None), 
            weighted_mean=pl.lit(None)
        )

    # 2. Build the Weighted Sum Expression
    total_weight = sum(weights_map[aid] for aid in valid_assay_cols)
    
    weighted_sum_expr = pl.sum_horizontal([
        pl.col(aid) * weights_map[aid] for aid in valid_assay_cols
    ])

    return lf.with_columns(
        mean=pl.mean_horizontal(valid_assay_cols),
        weighted_mean=weighted_sum_expr / total_weight
    ).select(["smiles", "mean", "weighted_mean"])

def process_seed(seed: int, thresholds_map: Dict, mode: str, assay_short: str):
    """
    Loads GT and all Model Predictions for a single seed.
    """
    print(f"Processing Seed {seed}...")
    
    # 1. Load Ground Truth
    gt_lf = pl.scan_parquet(DATA_DIR / mode / f"{mode}_multilabel.parquet")
    gt_df = (
        gt_lf
        .filter(pl.col(f"split{seed}") == "test")
        .select(["smiles", f"score_seed{seed}"])
        .rename({f"score_seed{seed}": "gt_score"})
        .collect(engine="streaming")
    )
    
    # 2. Metadata for weighting
    metadata = pl.read_parquet(DATA_DIR / mode / "assay_metadata.parquet")
    
    preds_map = {}

    # --- Load Regression ---
    reg_path = MODEL_DIR / f"reg_{assay_short}_seed{seed}" / "model_0" / "test_predictions.csv"
    if reg_path.exists():
        # User confirmed: Reg column is "score_seed{seed}"
        reg_df = pl.read_csv(reg_path)
        col_name = f"score_seed{seed}"
        if col_name in reg_df.columns:
            preds_map["Regression"] = reg_df.select([
                pl.col("smiles"), 
                pl.col(col_name).alias("score")
            ])
        else:
            print(f"Warning: Column {col_name} not found in Regression output.")

    # --- Load Multitask ---
    mt_path = MODEL_DIR / f"mt_{assay_short}_seed{seed}" / "model_0" / "test_predictions.csv"
    if mt_path.exists():
        mt_lf = pl.scan_csv(mt_path)
        mt_agg = prevalence_weighted_average(mt_lf, metadata).collect()
        
        preds_map["MT_Mean"] = mt_agg.select(["smiles", pl.col("mean").alias("score")])
        preds_map["MT_Weighted"] = mt_agg.select(["smiles", pl.col("weighted_mean").alias("score")])

    # --- Load Threshold Classifiers ---
    for th in THRESHOLDS_PCT:
        thr_path = MODEL_DIR / f"thr_{assay_short}_seed{seed}_p50_{th}" / "model_0" / "test_predictions.csv"
        if thr_path.exists():
            t_df = pl.read_csv(thr_path)
            if "target" in t_df.columns:
                preds_map[f"Thr_{th}"] = t_df.select([
                    pl.col("smiles"), 
                    pl.col("target").alias("score")
                ])

    return gt_df, preds_map

# --- Main Analysis Loop ---

def main():
    thresholds_data = load_thresholds(ASSAY_MODE)
    
    # Accumulators
    ranking_results = []
    lift_curve_data = []
    classification_results = []

    for seed in SEEDS:
        gt_df, preds_map = process_seed(seed, thresholds_data, ASSAY_MODE, ASSAY_SHORT)
        
        # Get strict lower cutoff (p50) for this seed
        p50_val = thresholds_data[str(seed)]["50"]
        
        for model_name, pred_df in preds_map.items():
            # Join GT and Predictions
            joined = gt_df.join(pred_df, on="smiles", how="inner")
            
            y_true_cont = joined["gt_score"].to_numpy()
            y_score = joined["score"].to_numpy()
            
            # --- 1. Ranking Metrics (NDCG & Lift) ---
            
            # NDCG (Global)
            ndcg_all = calculate_ndcg(y_true_cont, y_score, k=None)
            spearman_rho = stats.spearmanr(y_true_cont, y_score, nan_policy="omit").statistic
            
            ranking_results.append({
                "seed": seed,
                "model": model_name,
                "ndcg": ndcg_all,
                "spearman_rho": spearman_rho,
            })
            
            # Lift Curve (0.5% to 10%)
            fractions = [0.005, 0.01, 0.02, 0.05, 0.10]
            for frac in fractions:
                val = continuous_lift(y_true_cont, y_score, frac)
                lift_curve_data.append({
                    "seed": seed,
                    "model": model_name,
                    "top_fraction": frac * 100,
                    "lift": val
                })
            
            # --- 2. Classification Metrics (Per Threshold "Donut") ---
            
            for th_key in THRESHOLDS_PCT:
                # Upper cutoff for this threshold task
                p_upper_val = thresholds_data[str(seed)][th_key]
                
                # Filter Logic: Exclude (p50, p_upper)
                # Keep: score <= p50 OR score >= p_upper
                mask_negative = joined["gt_score"] <= p50_val
                mask_positive = joined["gt_score"] >= p_upper_val
                
                eval_df = joined.filter(
                    (pl.col("gt_score") <= p50_val) | (pl.col("gt_score") >= p_upper_val)
                )
                
                if len(eval_df) < 10:
                    continue 
                
                # Binary Target: 1 if >= upper, 0 if <= lower
                y_bin = (eval_df["gt_score"] >= p_upper_val).cast(pl.Int32).to_numpy()
                y_pred_bin = eval_df["score"].to_numpy()
                
                try:
                    roc = roc_auc_score(y_bin, y_pred_bin)
                    pr = average_precision_score(y_bin, y_pred_bin)
                except ValueError:
                    roc, pr = np.nan, np.nan
                
                precision_90_recall = precision_at_recall_90(y_bin, y_pred_bin)
                
                classification_results.append({
                    "seed": seed,
                    "model": model_name,
                    "eval_threshold": int(th_key),
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "precision_at_90_recall": precision_90_recall,
                })

    # --- Save Results ---
    
    rank_df = pl.DataFrame(ranking_results)
    lift_df = pl.DataFrame(lift_curve_data)
    cls_df = pl.DataFrame(classification_results)
    
    rank_df.write_csv(OUTPUT_DIR / "ranking_metrics_ndcg.csv")
    lift_df.write_csv(OUTPUT_DIR / "ranking_metrics_lift_curve.csv")
    cls_df.write_csv(OUTPUT_DIR / "classification_metrics_all_seeds.csv")
    rank_df.select(["seed", "model", "spearman_rho"]).write_csv(OUTPUT_DIR / "ranking_metrics_spearman.csv")
    cls_df.select(["seed", "model", "eval_threshold", "pr_auc"]).write_csv(OUTPUT_DIR / "classification_metrics_pr_auc.csv")
    
    # --- Plotting ---
    
    plot_results(rank_df, lift_df, cls_df)

def plot_results(rank_df: pl.DataFrame, lift_df: pl.DataFrame, cls_df: pl.DataFrame):
    """Generates the requested paper figures."""
    
    # Helper for sorting models
    # Order: Reg, MT_Mean, MT_Weighted, Thr_50, ... Thr_95
    model_order = ["Regression", "MT_Mean", "MT_Weighted"] + [f"Thr_{t}" for t in THRESHOLDS_PCT]
    
    # --- 1. NDCG Bar Plot ---
    plt.figure(figsize=(10, 6))
    valid_order = [m for m in model_order if m in set(rank_df["model"].to_list())]
    
    sns.barplot(
        data=rank_df, x="model", y="ndcg", 
        order=valid_order,
        capsize=.1, errorbar="sd", palette="viridis"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Ranking Quality (NDCG)")
    plt.ylabel("NDCG Score")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ranking_ndcg.png")
    plt.close()

    # --- 1b. Spearman's Rho Bar Plot ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=rank_df, x="model", y="spearman_rho",
        order=valid_order,
        capsize=.1, errorbar="sd", palette="magma"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Ranking Quality (Spearman's Rho)")
    plt.ylabel("Spearman's Rho")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ranking_spearman.png")
    plt.close()

    # --- 2. Lift Curve (Top End Performance) ---
    plt.figure(figsize=(8, 6))
    
    # Focus on key models to avoid clutter
    key_models = ["Regression", "MT_Weighted", "Thr_95", "Thr_50"]
    plot_data = lift_df.filter(pl.col("model").is_in(key_models))
    
    sns.lineplot(
        data=plot_data, x="top_fraction", y="lift", hue="model", style="model",
        markers=True, dashes=False, err_style="bars", errorbar="sd"
    )
    plt.title("Lift Curve: Enrichment at Top K%")
    plt.xlabel("Top K% of Predictions")
    plt.ylabel("Lift (Fold Enrichment)")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lift_curve.png")
    plt.close()

    # --- 3. Classification Performance vs Difficulty ---
    plt.figure(figsize=(8, 6))
    
    # A. Standard Models (Lines)
    main_models = ["Regression", "MT_Weighted"]
    main_subset = cls_df.filter(pl.col("model").is_in(main_models)).with_columns(
        pl.col("model").alias("plot_group")
    )
    
    # B. Native Classifiers (Diagonal)
    native_rows = []
    for th in THRESHOLDS_PCT:
        model_name = f"Thr_{th}"
        subset = cls_df.filter(
            (pl.col("model") == model_name) & (pl.col("eval_threshold") == int(th))
        )
        if subset.height > 0:
            native_rows.append(
                subset.with_columns(pl.lit("Native Threshold Classifiers").alias("plot_group"))
            )
    
    native_df = pl.concat(native_rows) if native_rows else pl.DataFrame()
    
    # Combine
    plot_data = pl.concat([main_subset, native_df]) if native_rows else main_subset
    
    sns.lineplot(
        data=plot_data, x="eval_threshold", y="roc_auc", hue="plot_group", style="plot_group",
        markers=True, dashes=False, err_style="bars", errorbar="sd"
    )
    plt.title("Performance vs. Frequent Hitter Definition")
    plt.ylabel("ROC-AUC")
    plt.xlabel("Frequent Hitter Threshold (Percentile)")
    plt.legend(title="Model Type")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classification_performance_curve.png")
    plt.close()

    # --- 3b. PR-AUC Performance vs Difficulty ---
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=plot_data, x="eval_threshold", y="pr_auc", hue="plot_group", style="plot_group",
        markers=True, dashes=False, err_style="bars", errorbar="sd"
    )
    plt.title("PR-AUC vs. Frequent Hitter Definition")
    plt.ylabel("PR-AUC")
    plt.xlabel("Frequent Hitter Threshold (Percentile)")
    plt.legend(title="Model Type")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classification_performance_curve_pr_auc.png")
    plt.close()

if __name__ == "__main__":
    main()
