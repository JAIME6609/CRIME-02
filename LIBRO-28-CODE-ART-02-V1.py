#!/usr/bin/env python3
"""
Responsible AI Marketing Risk Observatory
=========================================

This complete, reproducible study accompanies the chapter
"Responsible AI for Crime-Risk Prediction in Digital Marketing".

The program creates a privacy-preserving synthetic panel of aggregated
zone-week observations, trains temporally validated risk models, calibrates the
best model, explains its behavior, audits group fairness, quantifies predictive
uncertainty with split conformal prediction, and converts model outputs into
non-exclusionary customer-protection actions.

Important use limitation
------------------------
The generated risk score is a probabilistic decision-support signal. It must
not be used to identify individuals, deny service, raise prices, exclude
neighborhoods, or trigger punitive action. Its only intended operational uses
are protective communication, fraud warnings, secure-channel verification,
and human review.

All outputs are written under results/5.1, results/5.2, and results/5.3 so that
they can be inserted directly into the three empirical subsections of the
chapter.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


# ---------------------------------------------------------------------------
# 0. Dependency verification and local installation
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES: Mapping[str, str] = {
    "numpy": "numpy>=1.26",
    "pandas": "pandas>=2.1",
    "polars": "polars>=0.20",
    "sklearn": "scikit-learn>=1.4",
    "xgboost": "xgboost>=2.0",
    "shap": "shap>=0.45",
    "fairlearn": "fairlearn>=0.10",
    "matplotlib": "matplotlib>=3.8",
    "seaborn": "seaborn>=0.13",
}


def ensure_dependencies() -> None:
    """Install missing scientific packages into a project-local directory.

    A local target avoids modifying the global Python installation and works in
    Anaconda, Spyder, Jupyter, and ordinary command-line environments. Set the
    environment variable RAI_SKIP_INSTALL=1 to disable automatic installation.
    """

    local_target = Path(__file__).resolve().parent / ".rai_marketing_packages"
    if local_target.exists():
        sys.path.insert(0, str(local_target))

    missing = [spec for module, spec in REQUIRED_PACKAGES.items()
               if importlib.util.find_spec(module) is None]
    if not missing:
        return

    if os.environ.get("RAI_SKIP_INSTALL", "0") == "1":
        raise ModuleNotFoundError(
            "Missing packages: " + ", ".join(missing) +
            ". Remove RAI_SKIP_INSTALL or install them manually."
        )

    local_target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--target", str(local_target), *missing,
    ]
    print("Installing missing scientific dependencies locally:")
    print(" ".join(command))
    subprocess.check_call(command)
    sys.path.insert(0, str(local_target))
    importlib.invalidate_caches()


ensure_dependencies()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import shap
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate
from matplotlib.ticker import PercentFormatter
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# 1. Reproducible configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StudyConfig:
    master_seed: int = 2026
    n_zones: int = 36
    n_weeks: int = 156
    train_weeks: int = 108
    calibration_weeks: int = 24
    test_weeks: int = 24
    high_risk_quantile: float = 0.72
    conformal_alpha: float = 0.10
    bootstrap_repetitions: int = 500
    output_root: str = "results"


CONFIG = StudyConfig()
FEATURES: List[str] = [
    "engagement_volume_log",
    "campaign_exposure",
    "late_night_share",
    "fraud_signal_rate",
    "failed_payment_rate",
    "verified_channel_share",
    "economic_stress_index",
    "public_event_intensity",
    "lag_1_incidents",
    "lag_4_incidents_mean",
    "rolling_8_incidents_mean",
    "week_sin",
    "week_cos",
    "trend",
]


def setup_output_directories(root: Path) -> Dict[str, Path]:
    folders = {section: root / section for section in ("5.1", "5.2", "5.3")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save human-readable CSV and publication-ready Excel-free text output."""
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Synthetic privacy-preserving panel generation
# ---------------------------------------------------------------------------

def generate_synthetic_panel(config: StudyConfig) -> pd.DataFrame:
    """Generate aggregated zone-week observations from a documented mechanism.

    The data-generating process contains temporal seasonality, persistent zone
    heterogeneity, campaign intensity, platform fraud signals, protective
    verification, and autoregressive incident dynamics. A synthetic audit group
    is retained only for fairness evaluation and is excluded from FEATURES.
    """

    rng = np.random.default_rng(config.master_seed)
    weeks = pd.date_range("2023-01-02", periods=config.n_weeks, freq="W-MON")
    zones = np.arange(config.n_zones)

    audit_group = np.where(zones % 3 == 0, "Higher vulnerability",
                           np.where(zones % 3 == 1, "Intermediate vulnerability",
                                    "Lower vulnerability"))
    group_shift = {
        "Higher vulnerability": 0.18,
        "Intermediate vulnerability": 0.04,
        "Lower vulnerability": -0.08,
    }
    zone_effect = rng.normal(0.0, 0.32, config.n_zones) + np.array(
        [group_shift[g] for g in audit_group]
    )
    economic_base = np.clip(rng.beta(2.4, 2.8, config.n_zones), 0.05, 0.95)

    rows: List[dict] = []
    history: Dict[int, List[int]] = {int(z): [0] * 8 for z in zones}

    for t, week in enumerate(weeks):
        seasonal = math.sin(2.0 * math.pi * t / 52.0)
        seasonal_cos = math.cos(2.0 * math.pi * t / 52.0)
        trend = t / max(config.n_weeks - 1, 1)

        for z in zones:
            group = audit_group[z]
            econ = np.clip(
                economic_base[z] + 0.10 * seasonal + rng.normal(0, 0.045),
                0.0, 1.0,
            )
            engagement = rng.lognormal(
                mean=7.7 + 0.22 * seasonal_cos + 0.08 * (z % 5), sigma=0.28
            )
            campaign = np.clip(rng.beta(2.3 + 0.5 * seasonal_cos, 2.4), 0, 1)
            late_night = np.clip(rng.beta(2.0 + 0.8 * campaign, 7.0), 0, 1)
            public_event = np.clip(
                0.20 + 0.45 * max(seasonal, 0) + rng.normal(0, 0.12), 0, 1
            )
            fraud_signal = np.clip(
                rng.beta(1.5 + 3.0 * campaign + 1.2 * late_night, 12.0), 0, 1
            )
            failed_payment = np.clip(
                0.015 + 0.22 * fraud_signal + 0.05 * econ + rng.normal(0, 0.018),
                0, 0.40,
            )
            verified_share = np.clip(
                0.78 - 0.16 * campaign + rng.normal(0, 0.055), 0.35, 0.98
            )

            lag1 = history[int(z)][-1]
            lag4 = float(np.mean(history[int(z)][-4:]))
            lag8 = float(np.mean(history[int(z)][-8:]))

            log_lambda = (
                0.48 + zone_effect[z]
                + 0.22 * np.log1p(engagement / 1000.0)
                + 2.20 * fraud_signal
                + 1.30 * failed_payment
                + 0.78 * late_night
                + 0.62 * econ
                + 0.38 * public_event
                - 0.98 * verified_share
                + 0.16 * np.log1p(lag1)
                + 0.12 * np.log1p(lag4)
                + 0.12 * seasonal
                + 0.10 * trend
            )
            expected_incidents = float(np.clip(np.exp(log_lambda), 0.10, 18.0))
            incidents = int(rng.poisson(expected_incidents))
            history[int(z)].append(incidents)

            rows.append({
                "week": week,
                "week_index": t,
                "zone_id": f"Z{z + 1:02d}",
                "audit_group": group,
                "engagement_volume": engagement,
                "engagement_volume_log": np.log1p(engagement),
                "campaign_exposure": campaign,
                "late_night_share": late_night,
                "fraud_signal_rate": fraud_signal,
                "failed_payment_rate": failed_payment,
                "verified_channel_share": verified_share,
                "economic_stress_index": econ,
                "public_event_intensity": public_event,
                "lag_1_incidents": lag1,
                "lag_4_incidents_mean": lag4,
                "rolling_8_incidents_mean": lag8,
                "week_sin": seasonal,
                "week_cos": seasonal_cos,
                "trend": trend,
                "expected_incidents": expected_incidents,
                "incidents": incidents,
            })

    df = pd.DataFrame(rows).sort_values(["week_index", "zone_id"]).reset_index(drop=True)
    threshold = float(df.loc[df["week_index"] < config.train_weeks, "incidents"].quantile(
        config.high_risk_quantile
    ))
    df["high_risk"] = (df["incidents"] >= max(1.0, threshold)).astype(int)
    return df


def temporal_split(df: pd.DataFrame, config: StudyConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["week_index"] < config.train_weeks].copy()
    calibration = df[
        (df["week_index"] >= config.train_weeks)
        & (df["week_index"] < config.train_weeks + config.calibration_weeks)
    ].copy()
    test = df[df["week_index"] >= config.train_weeks + config.calibration_weeks].copy()
    assert len(train) + len(calibration) + len(test) == len(df)
    return train, calibration, test


# ---------------------------------------------------------------------------
# 3. Models, temporal validation, and probability calibration
# ---------------------------------------------------------------------------

def build_models(seed: int, positive_weight: float) -> Dict[str, object]:
    return {
        "Logistic regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2500, class_weight="balanced", random_state=seed
            )),
        ]),
        "Random forest": RandomForestClassifier(
            n_estimators=650,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=700,
            learning_rate=0.035,
            max_depth=4,
            min_child_weight=5,
            subsample=0.82,
            colsample_bytree=0.84,
            reg_alpha=0.08,
            reg_lambda=1.4,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=positive_weight,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(p, edges, right=True) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = assignments == b
        if np.any(mask):
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def metric_row(model_name: str, y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "Model": model_name,
        "ROC AUC": roc_auc_score(y, p),
        "PR AUC": average_precision_score(y, p),
        "Brier score": brier_score_loss(y, p),
        "Log loss": log_loss(y, p, labels=[0, 1]),
        "ECE": expected_calibration_error(y, p),
        "Accuracy": accuracy_score(y, pred),
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Threshold": threshold,
    }


def optimize_protective_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """Choose a threshold using an explicit customer-protection utility.

    True positives produce an estimated protection benefit of +5.0 units;
    false positives incur communication/review burden of -1.2; false negatives
    incur an avoidable-harm cost of -4.0; true negatives have zero incremental
    value. The utility is illustrative and must be locally validated.
    """

    records = []
    for threshold in np.linspace(0.10, 0.90, 81):
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        utility = 5.0 * tp - 1.2 * fp - 4.0 * fn
        records.append({
            "Threshold": threshold,
            "True positives": tp,
            "False positives": fp,
            "False negatives": fn,
            "True negatives": tn,
            "Protection utility": utility,
            "Utility per observation": utility / len(y),
        })
    table = pd.DataFrame(records)
    best = table.sort_values(
        ["Protection utility", "Threshold"], ascending=[False, True]
    ).iloc[0]
    return float(best["Threshold"]), table


def bootstrap_auc_interval(y: np.ndarray, p: np.ndarray, repetitions: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y)
    for _ in range(repetitions):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size == 2:
            scores.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.quantile(scores, [0.025, 0.975]).tolist())


# ---------------------------------------------------------------------------
# 4. Explainability, fairness, and conformal uncertainty
# ---------------------------------------------------------------------------

def compute_shap_importance(model: XGBClassifier, x_test: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    sample = x_test.sample(min(900, len(x_test)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[-1]
    importance = pd.DataFrame({
        "Feature": sample.columns,
        "Mean absolute SHAP value": np.abs(values).mean(axis=0),
        "Mean signed SHAP value": values.mean(axis=0),
    }).sort_values("Mean absolute SHAP value", ascending=False)
    return importance, np.asarray(values), sample


def audit_fairness(y: np.ndarray, pred: np.ndarray, p: np.ndarray, groups: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = {
        "Selection rate": selection_rate,
        "True-positive rate": recall_score,
        "False-positive rate": false_positive_rate,
        "Precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "Accuracy": accuracy_score,
    }
    frame = MetricFrame(metrics=metrics, y_true=y, y_pred=pred, sensitive_features=groups)
    by_group = frame.by_group.reset_index().rename(columns={"audit_group": "Audit group"})
    calibration_rows = []
    for group in sorted(groups.unique()):
        mask = groups.to_numpy() == group
        calibration_rows.append({
            "Audit group": group,
            "Sample size": int(mask.sum()),
            "Observed prevalence": float(y[mask].mean()),
            "Mean predicted risk": float(p[mask].mean()),
            "Brier score": float(brier_score_loss(y[mask], p[mask])),
            "Calibration gap": float(abs(y[mask].mean() - p[mask].mean())),
        })
    return by_group, pd.DataFrame(calibration_rows)


def split_conformal_sets(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    probabilities_cal = np.column_stack([1.0 - p_cal, p_cal])
    calibration_scores = 1.0 - probabilities_cal[np.arange(len(y_cal)), y_cal]
    n = len(calibration_scores)
    quantile_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    qhat = float(np.quantile(calibration_scores, quantile_level, method="higher"))

    probabilities_test = np.column_stack([1.0 - p_test, p_test])
    prediction_sets = probabilities_test >= (1.0 - qhat)
    contains_truth = prediction_sets[np.arange(len(y_test)), y_test]
    sizes = prediction_sets.sum(axis=1)

    summary = pd.DataFrame([{
        "Nominal coverage": 1.0 - alpha,
        "Empirical coverage": contains_truth.mean(),
        "Mean set size": sizes.mean(),
        "Singleton rate": (sizes == 1).mean(),
        "Ambiguous-set rate": (sizes == 2).mean(),
        "Empty-set rate": (sizes == 0).mean(),
        "Conformal quantile": qhat,
    }])
    details = pd.DataFrame({
        "Observed class": y_test,
        "Predicted risk": p_test,
        "Include low-risk class": prediction_sets[:, 0],
        "Include high-risk class": prediction_sets[:, 1],
        "Prediction-set size": sizes,
        "Contains observed class": contains_truth,
    })
    return summary, details, qhat


# ---------------------------------------------------------------------------
# 5. Publication figures and tables
# ---------------------------------------------------------------------------

PURPLE = "#6A1B9A"
TEAL = "#00796B"
ORANGE = "#EF6C00"
BLUE = "#1565C0"


def configure_visual_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def figure_prevalence(df: pd.DataFrame, path: Path) -> None:
    weekly = df.groupby("week", as_index=False).agg(
        prevalence=("high_risk", "mean"),
        incidents=("incidents", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(8.2, 4.4))
    ax1.plot(weekly["week"], weekly["prevalence"], color=PURPLE, lw=1.8,
             label="High-risk prevalence")
    ax1.set_ylabel("High-risk prevalence", color=PURPLE)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2 = ax1.twinx()
    ax2.plot(weekly["week"], weekly["incidents"], color=TEAL, alpha=0.72,
             lw=1.4, label="Mean incidents")
    ax2.set_ylabel("Mean incidents per zone-week", color=TEAL)
    ax1.set_xlabel("Week")
    ax1.set_title("Temporal prevalence and incident intensity in the synthetic panel")
    fig.tight_layout()
    save_figure(fig, path)


def figure_roc_pr(y: np.ndarray, probabilities: Dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    colors = [PURPLE, TEAL, ORANGE, BLUE]
    for (name, p), color in zip(probabilities.items(), colors):
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, lw=1.8, color=color,
                     label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        axes[1].plot(recall, precision, lw=1.8, color=color,
                     label=f"{name} (AP={average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="0.55", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate",
                title="Receiver operating characteristic")
    axes[1].axhline(y.mean(), ls="--", color="0.55", lw=1)
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision–recall curve")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save_figure(fig, path)


def figure_calibration(y: np.ndarray, raw_p: np.ndarray, calibrated_p: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    for name, p, color in [
        ("Raw XGBoost", raw_p, ORANGE),
        ("Isotonic-calibrated XGBoost", calibrated_p, PURPLE),
    ]:
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", lw=1.8, color=color, label=name)
    ax.plot([0, 1], [0, 1], "--", color="0.45", label="Perfect calibration")
    ax.set(xlabel="Mean predicted risk", ylabel="Observed frequency",
           title="Out-of-time probability calibration")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, path)


def figure_shap_bar(importance: pd.DataFrame, path: Path) -> None:
    top = importance.head(12).sort_values("Mean absolute SHAP value")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.barh(top["Feature"], top["Mean absolute SHAP value"], color=PURPLE, alpha=0.86)
    ax.set(xlabel="Mean |SHAP value|", ylabel="",
           title="Global feature contribution to predicted crime risk")
    fig.tight_layout()
    save_figure(fig, path)


def figure_shap_distribution(values: np.ndarray, sample: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8.0, 5.3))
    shap.summary_plot(values, sample, max_display=12, show=False, plot_size=None)
    plt.title("Direction and magnitude of local feature contributions", pad=12)
    fig = plt.gcf()
    fig.tight_layout()
    save_figure(fig, path)


def figure_group_metrics(group_metrics: pd.DataFrame, path: Path) -> None:
    long = group_metrics.melt(
        id_vars=["Audit group"],
        value_vars=["Selection rate", "True-positive rate", "False-positive rate", "Precision"],
        var_name="Metric", value_name="Value",
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.barplot(data=long, x="Metric", y="Value", hue="Audit group", ax=ax,
                palette=[PURPLE, TEAL, ORANGE])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(xlabel="", ylabel="Rate", title="Fairness audit across synthetic vulnerability strata")
    ax.legend(title="Audit group", frameon=False, fontsize=7.5)
    fig.tight_layout()
    save_figure(fig, path)


def figure_uncertainty_action(test: pd.DataFrame, p: np.ndarray, details: pd.DataFrame,
                              threshold: float, path: Path) -> None:
    plot_df = test[["fraud_signal_rate", "verified_channel_share"]].reset_index(drop=True).copy()
    plot_df["Predicted risk"] = p
    plot_df["Uncertainty state"] = np.where(
        details["Prediction-set size"].to_numpy() == 1, "Singleton", "Ambiguous / abstain"
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.scatterplot(
        data=plot_df, x="fraud_signal_rate", y="Predicted risk",
        hue="Uncertainty state", style="Uncertainty state", size="verified_channel_share",
        sizes=(25, 120), alpha=0.68, palette=[PURPLE, ORANGE], ax=ax,
    )
    ax.axhline(threshold, color="black", ls="--", lw=1.1,
               label=f"Protective-action threshold ({threshold:.2f})")
    ax.set(title="Risk, uncertainty, and the human-review boundary",
           xlabel="Aggregated fraud-signal rate", ylabel="Calibrated high-risk probability")
    ax.legend(frameon=False, fontsize=7.2, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, path)


def figure_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    ax.axis("off")
    boxes = [
        (0.02, 0.58, "Aggregated\nzone-week data", TEAL),
        (0.22, 0.58, "Temporal validation\nand calibration", PURPLE),
        (0.43, 0.58, "Explainability, fairness,\nand uncertainty", BLUE),
        (0.66, 0.58, "Human governance\ngate", ORANGE),
        (0.82, 0.18, "Protective marketing\nactions only", TEAL),
    ]
    for x, y, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.16, 0.20, facecolor=color,
                                   alpha=0.12, edgecolor=color, lw=1.7))
        ax.text(x + 0.08, y + 0.10, label, ha="center", va="center", fontsize=9)
    arrows = [((0.18, 0.68), (0.22, 0.68)), ((0.38, 0.68), (0.43, 0.68)),
              ((0.59, 0.68), (0.66, 0.68)), ((0.74, 0.58), (0.86, 0.38))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="0.25"))
    ax.text(0.52, 0.28,
            "Forbidden path: individual profiling, service denial, price discrimination,\n"
            "neighborhood exclusion, or punitive enforcement",
            ha="center", va="center", fontsize=9, color="#B71C1C",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFEBEE", edgecolor="#B71C1C"))
    ax.set_title("Responsible AI Marketing Risk Observatory: control architecture", pad=12)
    fig.tight_layout()
    save_figure(fig, path)


# ---------------------------------------------------------------------------
# 6. End-to-end study orchestration
# ---------------------------------------------------------------------------

def run_study(config: StudyConfig = CONFIG) -> dict:
    configure_visual_style()
    root = Path(__file__).resolve().parent / config.output_root
    folders = setup_output_directories(root)

    print("1/7 Generating privacy-preserving synthetic zone-week data...")
    df = generate_synthetic_panel(config)
    train, calibration, test = temporal_split(df, config)

    data_profile = pd.DataFrame([
        {"Partition": "Training", "Rows": len(train), "Weeks": train["week_index"].nunique(),
         "Positive prevalence": train["high_risk"].mean()},
        {"Partition": "Calibration", "Rows": len(calibration), "Weeks": calibration["week_index"].nunique(),
         "Positive prevalence": calibration["high_risk"].mean()},
        {"Partition": "Out-of-time test", "Rows": len(test), "Weeks": test["week_index"].nunique(),
         "Positive prevalence": test["high_risk"].mean()},
    ])
    save_table(data_profile, folders["5.1"] / "Table_01_Data_Profile.csv")
    df.to_csv(folders["5.1"] / "Synthetic_Zone_Week_Dataset.csv", index=False,
              encoding="utf-8-sig", float_format="%.6f")
    figure_prevalence(df, folders["5.1"] / "Figure_01_Temporal_Risk_Prevalence.png")

    x_train, y_train = train[FEATURES], train["high_risk"].to_numpy()
    x_cal, y_cal = calibration[FEATURES], calibration["high_risk"].to_numpy()
    x_test, y_test = test[FEATURES], test["high_risk"].to_numpy()
    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    print("2/7 Training temporally ordered model candidates...")
    models = build_models(config.master_seed, positive_weight)
    raw_test_probabilities: Dict[str, np.ndarray] = {}
    calibration_probabilities: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        calibration_probabilities[name] = model.predict_proba(x_cal)[:, 1]
        raw_test_probabilities[name] = model.predict_proba(x_test)[:, 1]

    # Select the candidate on calibration PR AUC, never on the test partition.
    best_name = max(
        models,
        key=lambda name: average_precision_score(y_cal, calibration_probabilities[name]),
    )
    best_model = models[best_name]
    raw_cal = calibration_probabilities[best_name]
    raw_test = raw_test_probabilities[best_name]

    print(f"3/7 Calibrating {best_name} and selecting a protection threshold...")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(raw_cal, y_cal)
    calibrated_cal = calibrator.predict(raw_cal)
    calibrated_test = calibrator.predict(raw_test)
    threshold, threshold_table = optimize_protective_threshold(y_cal, calibrated_cal)
    save_table(threshold_table, folders["5.1"] / "Table_03_Threshold_Utility.csv")

    all_test_probabilities = dict(raw_test_probabilities)
    all_test_probabilities[f"{best_name} calibrated"] = calibrated_test
    performance = pd.DataFrame([
        metric_row(name, y_test, p, threshold if "calibrated" in name else 0.5)
        for name, p in all_test_probabilities.items()
    ])
    low, high = bootstrap_auc_interval(
        y_test, calibrated_test, config.bootstrap_repetitions, config.master_seed + 11
    )
    performance["ROC AUC 95% CI lower"] = np.nan
    performance["ROC AUC 95% CI upper"] = np.nan
    mask = performance["Model"] == f"{best_name} calibrated"
    performance.loc[mask, "ROC AUC 95% CI lower"] = low
    performance.loc[mask, "ROC AUC 95% CI upper"] = high
    save_table(performance, folders["5.1"] / "Table_02_Model_Performance.csv")
    figure_roc_pr(y_test, all_test_probabilities,
                  folders["5.1"] / "Figure_02_ROC_and_PR_Curves.png")
    figure_calibration(y_test, raw_test, calibrated_test,
                       folders["5.1"] / "Figure_03_Probability_Calibration.png")

    print("4/7 Computing SHAP explanations...")
    if best_name == "XGBoost":
        shap_importance, shap_values, shap_sample = compute_shap_importance(
            best_model, x_test, config.master_seed
        )
    else:
        # This path is used only if another candidate wins on calibration data.
        perm = permutation_importance(best_model, x_test, y_test, n_repeats=30,
                                      random_state=config.master_seed, n_jobs=-1,
                                      scoring="average_precision")
        shap_importance = pd.DataFrame({
            "Feature": FEATURES,
            "Mean absolute SHAP value": np.abs(perm.importances_mean),
            "Mean signed SHAP value": perm.importances_mean,
        }).sort_values("Mean absolute SHAP value", ascending=False)
        # Refit an explainable XGBoost companion solely for SHAP diagnostics.
        companion = build_models(config.master_seed, positive_weight)["XGBoost"]
        companion.fit(x_train, y_train)
        shap_importance, shap_values, shap_sample = compute_shap_importance(
            companion, x_test, config.master_seed
        )
    save_table(shap_importance, folders["5.2"] / "Table_04_SHAP_Feature_Importance.csv")
    figure_shap_bar(shap_importance, folders["5.2"] / "Figure_04_Global_SHAP_Importance.png")
    figure_shap_distribution(shap_values, shap_sample,
                             folders["5.2"] / "Figure_05_SHAP_Distribution.png")

    print("5/7 Auditing group fairness and calibration...")
    test_pred = (calibrated_test >= threshold).astype(int)
    group_metrics, group_calibration = audit_fairness(
        y_test, test_pred, calibrated_test, test["audit_group"].reset_index(drop=True)
    )
    save_table(group_metrics, folders["5.2"] / "Table_05_Fairness_Metrics.csv")
    save_table(group_calibration, folders["5.2"] / "Table_06_Group_Calibration.csv")
    figure_group_metrics(group_metrics,
                         folders["5.2"] / "Figure_06_Group_Fairness_Audit.png")

    print("6/7 Constructing conformal uncertainty sets and governance outputs...")
    conformal_summary, conformal_details, qhat = split_conformal_sets(
        y_cal, calibrated_cal, y_test, calibrated_test, config.conformal_alpha
    )
    save_table(conformal_summary, folders["5.3"] / "Table_07_Conformal_Performance.csv")
    conformal_details.to_csv(
        folders["5.3"] / "Conformal_Prediction_Details.csv", index=False,
        encoding="utf-8-sig", float_format="%.6f"
    )

    action_matrix = pd.DataFrame([
        {"Risk state": "Low risk; singleton conformal set", "Permitted action": "Standard secure communication",
         "Human review": "Routine sampling", "Prohibited action": "No exclusion or reduced service"},
        {"Risk state": "Elevated risk; singleton conformal set", "Permitted action": "Fraud warning, verified link, safer scheduling",
         "Human review": "Required for material campaign change", "Prohibited action": "No individual profiling or punitive escalation"},
        {"Risk state": "Ambiguous conformal set", "Permitted action": "Abstain from automation; request analyst review",
         "Human review": "Mandatory", "Prohibited action": "No automated high-impact decision"},
        {"Risk state": "Drift or fairness guardrail breached", "Permitted action": "Pause model-assisted action and investigate",
         "Human review": "Mandatory governance review", "Prohibited action": "No continued unattended deployment"},
    ])
    save_table(action_matrix, folders["5.3"] / "Table_08_Responsible_Action_Matrix.csv")

    disparity = float(group_metrics["True-positive rate"].max() - group_metrics["True-positive rate"].min())
    governance_scorecard = pd.DataFrame([
        {"Control": "Out-of-time ROC AUC", "Observed value": float(roc_auc_score(y_test, calibrated_test)),
         "Guardrail": ">= 0.70", "Status": "Pass" if roc_auc_score(y_test, calibrated_test) >= 0.70 else "Review"},
        {"Control": "Brier score", "Observed value": float(brier_score_loss(y_test, calibrated_test)),
         "Guardrail": "<= 0.20", "Status": "Pass" if brier_score_loss(y_test, calibrated_test) <= 0.20 else "Review"},
        {"Control": "Conformal coverage", "Observed value": float(conformal_summary.loc[0, "Empirical coverage"]),
         "Guardrail": f">= {1-config.conformal_alpha-0.03:.2f}",
         "Status": "Pass" if conformal_summary.loc[0, "Empirical coverage"] >= 1-config.conformal_alpha-0.03 else "Review"},
        {"Control": "TPR disparity", "Observed value": disparity,
         "Guardrail": "<= 0.15", "Status": "Pass" if disparity <= 0.15 else "Review"},
        {"Control": "Permitted action scope", "Observed value": 1.0,
         "Guardrail": "Protective actions only", "Status": "Pass"},
    ])
    save_table(governance_scorecard, folders["5.3"] / "Table_09_Governance_Scorecard.csv")
    figure_uncertainty_action(
        test.reset_index(drop=True), calibrated_test, conformal_details, threshold,
        folders["5.3"] / "Figure_07_Uncertainty_and_Human_Review.png"
    )
    figure_architecture(folders["5.3"] / "Figure_08_Responsible_AI_Architecture.png")

    print("7/7 Writing reproducibility manifest...")
    summary = {
        "chapter_title": "Responsible AI for Crime-Risk Prediction in Digital Marketing",
        "configuration": asdict(config),
        "feature_set": FEATURES,
        "best_model_selected_on_calibration_partition": best_name,
        "protective_action_threshold": threshold,
        "conformal_quantile": qhat,
        "test_roc_auc": float(roc_auc_score(y_test, calibrated_test)),
        "test_pr_auc": float(average_precision_score(y_test, calibrated_test)),
        "test_brier_score": float(brier_score_loss(y_test, calibrated_test)),
        "test_ece": float(expected_calibration_error(y_test, calibrated_test)),
        "conformal_empirical_coverage": float(conformal_summary.loc[0, "Empirical coverage"]),
        "true_positive_rate_disparity": disparity,
        "synthetic_data_statement": (
            "All records are synthetic, aggregated zone-week observations created with "
            "master_seed=2026. They do not describe real people, places, crimes, or campaigns."
        ),
        "permitted_use": "Customer-protective marketing decision support with human oversight.",
        "prohibited_use": (
            "Individual profiling, service denial, price discrimination, neighborhood exclusion, "
            "or punitive law-enforcement action."
        ),
    }
    with open(root / "reproducibility_manifest.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("Study completed successfully.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run_study()
