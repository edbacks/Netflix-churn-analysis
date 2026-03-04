import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

H14_PATH = "user_features_churn_full_h14.csv"
H30_PATH = "user_features_churn_full_h30.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# 누수/메타 컬럼 제거 리스트
DROP_COLS = ["churn", "sessions_in_horizon", "horizon_days", "feature_window_days", "cutoff_date"]


def run_one(path: str, tag: str):
    df = pd.read_csv(path)

    # label
    y = df["churn"].astype(int)

    # features: 숫자형만 + 누수/메타 제거
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)

    print(f"\n===== {tag} =====")
    print("rows(users):", len(df), "features:", X.shape[1])
    print("churn rate:", float(y.mean()))
    print("churn counts:\n", y.value_counts())

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        min_samples_leaf=5,
        n_jobs=-1
    )

    # CV AUC
    lr_cv_auc = float(cross_val_score(lr, X_train, y_train, cv=cv, scoring="roc_auc").mean())
    rf_cv_auc = float(cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc").mean())

    # fit
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # test metrics
    lr_prob = lr.predict_proba(X_test)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]

    lr_auc = float(roc_auc_score(y_test, lr_prob))
    rf_auc = float(roc_auc_score(y_test, rf_prob))

    # PR-AUC (불균형 데이터에 유용)
    lr_prauc = float(average_precision_score(y_test, lr_prob))
    rf_prauc = float(average_precision_score(y_test, rf_prob))

    print("\n[CV ROC-AUC] LR:", lr_cv_auc, "| RF:", rf_cv_auc)
    print("[Test ROC-AUC] LR:", lr_auc, "| RF:", rf_auc)
    print("[Test PR-AUC ] LR:", lr_prauc, "| RF:", rf_prauc)

    print("\n[Classification report - RF]")
    print(classification_report(y_test, rf.predict(X_test), digits=4))

    # Top features
    feat_names = X.columns.tolist()

    lr_coef = lr.named_steps["clf"].coef_[0]
    lr_top = sorted(zip(feat_names, lr_coef), key=lambda x: abs(x[1]), reverse=True)[:10]

    rf_top = sorted(zip(feat_names, rf.feature_importances_), key=lambda x: x[1], reverse=True)[:10]

    print("\n[Top 10 - Logistic (|coef|)]")
    for n, v in lr_top:
        print(f"{n:30s} coef={v:.6f}")

    print("\n[Top 10 - RandomForest]")
    for n, v in rf_top:
        print(f"{n:30s} imp={v:.6f}")


def main():
    run_one(H14_PATH, "HORIZON=14")
    run_one(H30_PATH, "HORIZON=30")


if __name__ == "__main__":
    main()
    