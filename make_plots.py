import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

H14_PATH = "user_features_churn_full_h14.csv"
H30_PATH = "user_features_churn_full_h30.csv"

OUT_CHURN_PNG = "churn_distribution_h14_h30.png"
OUT_IMPORTANCE_PNG_H14 = "feature_importance_rf_h14.png"
OUT_IMPORTANCE_PNG_H30 = "feature_importance_rf_h30.png"
OUT_RECENCY_PNG = "recency_vs_churn_h14_h30.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 누수/메타 컬럼은 모델 입력에서 제외
DROP_COLS = ["churn", "sessions_in_horizon", "horizon_days", "feature_window_days", "cutoff_date"]


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def churn_distribution_plot(df14: pd.DataFrame, df30: pd.DataFrame):
    r14 = float(df14["churn"].mean())
    r30 = float(df30["churn"].mean())

    plt.figure()
    plt.bar(["H=14", "H=30"], [r14, r30])
    plt.title("Churn rate by horizon")
    plt.ylabel("Churn rate")
    plt.ylim(0, 1)
    plt.savefig(OUT_CHURN_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", OUT_CHURN_PNG, "| rates:", r14, r30)


def train_rf_and_plot_importance(df: pd.DataFrame, tag: str, out_png: str, top_k: int = 15):
    y = df["churn"].astype(int)

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # 간단 학습(importance 뽑기 목적)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        min_samples_leaf=5,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    names = [X.columns[i] for i in idx]
    vals = importances[idx]

    # 가로 막대 그래프(읽기 쉬움)
    plt.figure(figsize=(8, 6))
    plt.barh(list(reversed(names)), list(reversed(vals)))
    plt.title(f"Top {top_k} Feature Importance (RF) - {tag}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", out_png)


def recency_vs_churn_plot(df14: pd.DataFrame, df30: pd.DataFrame):
    # recency_days가 없으면 종료
    if "recency_days" not in df14.columns or "recency_days" not in df30.columns:
        print("[Skip] recency_days not found.")
        return

    def binned_churn(df: pd.DataFrame, max_day: int = 30):
        tmp = df.copy()
        tmp["recency_days"] = pd.to_numeric(tmp["recency_days"], errors="coerce").fillna(0).astype(int)
        tmp["recency_days"] = tmp["recency_days"].clip(lower=0, upper=max_day)
        grp = tmp.groupby("recency_days")["churn"].mean()
        return grp

    g14 = binned_churn(df14)
    g30 = binned_churn(df30)

    plt.figure()
    plt.plot(g14.index, g14.values, marker="o", label="H=14")
    plt.plot(g30.index, g30.values, marker="o", label="H=30")
    plt.title("Churn rate by recency_days")
    plt.xlabel("recency_days (binned)")
    plt.ylabel("Churn rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(OUT_RECENCY_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", OUT_RECENCY_PNG)


def main():
    df14 = load_df(H14_PATH)
    df30 = load_df(H30_PATH)

    # 1) churn 분포
    churn_distribution_plot(df14, df30)

    # 2) feature importance (RF)
    train_rf_and_plot_importance(df14, "HORIZON=14", OUT_IMPORTANCE_PNG_H14)
    train_rf_and_plot_importance(df30, "HORIZON=30", OUT_IMPORTANCE_PNG_H30)

    # 3) recency vs churn
    recency_vs_churn_plot(df14, df30)

    print("Done.")


if __name__ == "__main__":
    main()