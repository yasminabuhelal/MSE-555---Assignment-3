from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================================
# SETTINGS
# ============================================================================

CHOSEN_K = 3
SESSION_LIMIT = 12
PALETTE = ["tab:blue", "tab:red", "tab:green"]
OUT_DIR = Path("output/q3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# intake features used for classification
INTAKE_COLS = ["age_years", "complexity_score", "gender", "referral_reason"]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_historical_data() -> pd.DataFrame:
    # join intake features with cluster assignments from Q2
    features = pd.read_csv("data/client_features.csv")
    features["client_id"] = features["client_id"].astype(str).str.strip()

    clusters = pd.read_csv("output/q2/t_star_assignments.csv")
    clusters["client_id"] = clusters["client_id"].astype(str).str.strip()

    return features.merge(clusters[["client_id", "cluster", "t_star"]], on="client_id", how="inner")


# ============================================================================
# 3A — EDA: INTAKE FEATURES BY TRAJECTORY TYPE
# ============================================================================

def run_3a(df: pd.DataFrame) -> None:
    print("\n--- 3a: Exploring Intake Features by Trajectory Type ---")

    cluster_ids = sorted(df["cluster"].unique())
    x = np.arange(len(cluster_ids))
    labels = [f"Cluster {c}" for c in cluster_ids]

    # box plots for continuous features
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, ylabel, title in [
        (ax1, "age_years",        "Age (years)",      "Age at Intake by Cluster"),
        (ax2, "complexity_score", "Complexity Score", "Complexity Score by Cluster"),
    ]:
        data = [df[df["cluster"] == c][col].values for c in cluster_ids]
        bp = ax.boxplot(data, positions=cluster_ids, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[i])
            patch.set_alpha(0.7)
        ax.set_xticks(cluster_ids)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Cluster", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Continuous Intake Features by Trajectory Cluster", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_continuous_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: eda_continuous_features.png")

    # grouped bar charts for categorical features
    ref_reasons = sorted(df["referral_reason"].unique())
    ref_props = pd.crosstab(df["cluster"], df["referral_reason"], normalize="index")
    ref_props = ref_props.reindex(columns=ref_reasons, fill_value=0)

    genders = sorted(df["gender"].unique())
    gender_props = pd.crosstab(df["cluster"], df["gender"], normalize="index")
    gender_props = gender_props.reindex(columns=genders, fill_value=0)

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

    cat_colours = ["tab:green", "tab:orange", "tab:purple", "tab:pink"]
    n_reasons = len(ref_reasons)
    bar_w = 0.7 / n_reasons
    for j, reason in enumerate(ref_reasons):
        offset = (j - n_reasons / 2 + 0.5) * bar_w
        ax3.bar(x + offset, ref_props[reason].values, bar_w,
                label=reason, color=cat_colours[j], edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_xlabel("Cluster", fontsize=10)
    ax3.set_ylabel("Proportion", fontsize=10)
    ax3.set_title("Referral Reason by Cluster", fontsize=11)
    ax3.legend(title="Referral Reason", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    g_colours = ["tab:pink", "tab:cyan"]
    g_w = 0.35
    for j, g in enumerate(genders):
        offset = (j - len(genders) / 2 + 0.5) * g_w
        ax4.bar(x + offset, gender_props[g].values, g_w,
                label=g, color=g_colours[j], edgecolor="white")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.set_xlabel("Cluster", fontsize=10)
    ax4.set_ylabel("Proportion", fontsize=10)
    ax4.set_title("Gender by Cluster", fontsize=11)
    ax4.legend(title="Gender", fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    fig2.suptitle("Categorical Intake Features by Trajectory Cluster", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "eda_categorical_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: eda_categorical_features.png")

    # print continuous feature summary
    print(f"\n{'=' * 62}")
    print(f"  {'CONTINUOUS FEATURES BY CLUSTER':^56}")
    print("=" * 62)
    summary = (
        df.groupby("cluster")[["age_years", "complexity_score"]]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [f"{feat} {stat}" for feat, stat in summary.columns]
    summary.index = [f"Cluster {c}" for c in summary.index]

    age_cols  = [c for c in summary.columns if c.startswith("age_years")]
    cplx_cols = [c for c in summary.columns if c.startswith("complexity")]

    for feat_name, cols in [("Age at Intake", age_cols), ("Complexity Score", cplx_cols)]:
        print(f"\n  {feat_name}")
        print(f"  {'':18}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'-'*58}")
        for idx in summary.index:
            v = summary.loc[idx, cols]
            print(f"  {idx:18}  {v.iloc[0]:8.3f}  {v.iloc[1]:8.3f}  {v.iloc[2]:8.3f}  {v.iloc[3]:8.3f}")

    # print referral reason proportions
    rr = ref_props.copy()
    rr.index = [f"Cluster {c}" for c in rr.index]
    cw = 14
    tw = 2 + 18 + (cw + 2) * len(rr.columns) + 2
    print(f"\n{'=' * tw}")
    print(f"  {'REFERRAL REASON PROPORTIONS BY CLUSTER':^{tw - 4}}")
    print(f"{'=' * tw}")
    print(f"\n  {'':18}" + "".join(f"  {col:>{cw}}" for col in rr.columns))
    print(f"  {'-'*(18 + (cw + 2) * len(rr.columns))}")
    for idx, row in rr.iterrows():
        print(f"  {idx:18}" + "".join(f"  {v:>{cw}.1%}" for v in row))

    # print gender proportions
    gp = gender_props.copy()
    gp.index = [f"Cluster {c}" for c in gp.index]
    cw2 = 10
    tw2 = 2 + 18 + (cw2 + 2) * len(gp.columns) + 2
    print(f"\n{'=' * tw2}")
    print(f"  {'GENDER PROPORTIONS BY CLUSTER':^{tw2 - 4}}")
    print(f"{'=' * tw2}")
    print(f"\n  {'':18}" + "".join(f"  {col:>{cw2}}" for col in gp.columns))
    print(f"  {'-'*(18 + (cw2 + 2) * len(gp.columns))}")
    for idx, row in gp.iterrows():
        print(f"  {idx:18}" + "".join(f"  {v:>{cw2}.1%}" for v in row))
    print()


# ============================================================================
# 3B — CLASSIFICATION: PREDICT TRAJECTORY GROUP FROM INTAKE
# ============================================================================

def build_preprocessor() -> ColumnTransformer:
    # pass numeric features through, one-hot encode categoricals
    return ColumnTransformer(transformers=[
        ("num", "passthrough", ["age_years", "complexity_score"]),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["gender", "referral_reason"]),
    ])


def run_3b(df: pd.DataFrame) -> Pipeline:
    print("\n--- 3b: Training Classifiers for Trajectory Group Prediction ---")

    X = df[INTAKE_COLS]
    y = df["cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()

    candidates = {
        "Logistic Regression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
    }

    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels=[1, 2, 3])
        fitted[name] = {"pipe": pipe, "acc": acc, "cm": cm, "preds": preds}

        # save confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Cluster 1", "Cluster 2", "Cluster 3"]).plot(
            ax=ax, colorbar=True, cmap="Blues"
        )
        ax.set_title(f"Confusion Matrix — {name}", fontsize=12)
        fig.tight_layout()
        fname = "confusion_matrix_logreg.png" if "Logistic" in name else "confusion_matrix_rf.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

    # print accuracy comparison
    class_names = ["Cluster 1", "Cluster 2", "Cluster 3"]
    cw, rw = 11, 18
    print(f"\n{'=' * 62}")
    print(f"  {'MODEL ACCURACY COMPARISON':^58}")
    print(f"{'=' * 62}")
    print(f"\n  {'Model':<28}  {'Train N':>8}  {'Test N':>7}  {'Accuracy':>9}")
    print(f"  {'-'*58}")
    for name, res in fitted.items():
        print(f"  {name:<28}  {len(X_train):>8}  {len(X_test):>7}  {res['acc']:>8.1%}")

    # print per-model classification reports
    tw = 2 + rw + (cw + 2) * 4 + 2
    for name, res in fitted.items():
        rd = classification_report(
            y_test, res["preds"], labels=[1, 2, 3],
            target_names=class_names, output_dict=True
        )
        print(f"\n{'=' * tw}")
        print(f"  {'CLASSIFICATION REPORT — ' + name.upper():^{tw - 4}}")
        print(f"{'=' * tw}")
        print(f"\n  {'':>{rw}}  {'Precision':>{cw}}  {'Recall':>{cw}}  {'F1-Score':>{cw}}  {'Support':>{cw}}")
        print(f"  {'-'*(rw + (cw + 2) * 4)}")
        for cls in class_names:
            d = rd[cls]
            print(f"  {cls:>{rw}}  {d['precision']:>{cw}.3f}  {d['recall']:>{cw}.3f}  {d['f1-score']:>{cw}.3f}  {int(d['support']):>{cw}}")
        print(f"  {'-'*(rw + (cw + 2) * 4)}")
        d = rd["weighted avg"]
        print(f"  {'Weighted Avg':>{rw}}  {d['precision']:>{cw}.3f}  {d['recall']:>{cw}.3f}  {d['f1-score']:>{cw}.3f}  {int(d['support']):>{cw}}")
    print()

    return fitted["Logistic Regression"]["pipe"]


# ============================================================================
# 3C — WAITLIST CAPACITY ESTIMATION
# ============================================================================

def run_3c(model: Pipeline) -> None:
    print("\n--- 3c: Waitlist Capacity Estimation ---")

    waitlist = pd.read_csv("data/waitlist.csv")
    waitlist["client_id"] = waitlist["client_id"].astype(str).str.strip()
    waitlist["predicted_cluster"] = model.predict(waitlist[INTAKE_COLS])

    # recompute Q* and F(Q*) from t_star assignments
    t_star_df = pd.read_csv("output/q2/t_star_assignments.csv")
    cluster_ids = sorted(t_star_df["cluster"].unique())

    q_stars: dict[int, int] = {}
    f_at_qstar: dict[int, float] = {}

    for c in cluster_ids:
        t_vals = t_star_df[t_star_df["cluster"] == c]["t_star"].values
        best_q, best_s, best_f = 1, -1.0, 0.0
        for Q in range(1, SESSION_LIMIT + 1):
            F = float(np.mean(t_vals <= Q))
            s = F * (SESSION_LIMIT - Q)
            if s > best_s:
                best_s, best_q, best_f = s, Q, F
        q_stars[c] = best_q
        f_at_qstar[c] = best_f

    # E[sessions delivered] = F*Q + (1-F)*T_max
    e_delivered = {
        c: f_at_qstar[c] * q_stars[c] + (1 - f_at_qstar[c]) * SESSION_LIMIT
        for c in cluster_ids
    }

    waitlist["q_star"] = waitlist["predicted_cluster"].map(q_stars)
    waitlist["e_delivered"] = waitlist["predicted_cluster"].map(e_delivered)

    n_wait = len(waitlist)
    baseline = n_wait * SESSION_LIMIT

    # print capacity breakdown
    cw, rw = 14, 11
    tw = 2 + rw + (cw + 2) * 5 + 2
    print(f"\n{'=' * tw}")
    print(f"  {'WAITLIST CAPACITY BREAKDOWN BY CLUSTER':^{tw - 4}}")
    print(f"{'=' * tw}")
    print(f"\n  {'':>{rw}}  {'N predicted':>{cw}}  {'Q*':>{cw}}  {'F(Q*)':>{cw}}  {'E[sessions]':>{cw}}  {'Total sessions':>{cw}}")
    print(f"  {'-'*(rw + (cw + 2) * 5)}")

    grand_total = 0.0
    for c in cluster_ids:
        n_c = int((waitlist["predicted_cluster"] == c).sum())
        q = q_stars[c]
        f = f_at_qstar[c]
        e = e_delivered[c]
        total_c = n_c * e
        grand_total += total_c
        print(f"  {f'Cluster {c}':>{rw}}  {n_c:>{cw}}  {q:>{cw}}  {f:>{cw}.3f}  {e:>{cw}.2f}  {total_c:>{cw}.1f}")

    print(f"  {'-'*(rw + (cw + 2) * 5)}")
    print(f"  {'Total':>{rw}}  {n_wait:>{cw}}  {'—':>{cw}}  {'—':>{cw}}  {'—':>{cw}}  {grand_total:>{cw}.1f}")

    saved = baseline - grand_total
    pct = saved / baseline * 100

    print(f"\n{'=' * 54}")
    print(f"  {'SUMMARY':^50}")
    print(f"{'=' * 54}")
    print(f"\n  {'Metric':<36}  {'Value':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline (all clients × 12 sessions)':<36}  {baseline:>12}")
    print(f"  {'Q* policy total sessions':<36}  {round(grand_total):>12}")
    print(f"  {'Sessions saved':<36}  {round(saved):>12}")
    print(f"  {'% capacity saved':<36}  {pct:>11.1f}%")
    print()

    # save predictions
    waitlist[["client_id", "predicted_cluster", "q_star", "e_delivered"]].to_csv(
        OUT_DIR / "waitlist_predictions.csv", index=False
    )
    print(f"Saved: {OUT_DIR / 'waitlist_predictions.csv'}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df = load_historical_data()

    # 3a: explore how intake features differ across trajectory clusters
    run_3a(df)

    # 3b: train classifiers to predict cluster from intake features
    best_model = run_3b(df)

    # 3c: apply best model to waitlist and estimate session demand
    run_3c(best_model)