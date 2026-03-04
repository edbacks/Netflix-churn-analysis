import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report, confusion_matrix
)

# =========================
# 설정 (여기만 필요하면 바꾸세요)
# =========================
CSV_PATH = "watch_joined.csv"

HORIZON_DAYS = 14          # "이탈 판단" 기간 (14일/30일)
FEATURE_WINDOW_DAYS = 30   # 피처 생성에 쓸 과거 기간(30일)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# churn 정의:
# cutoff_date 이후 HORIZON_DAYS 안에 "시청 1번이라도" 있으면 churn=0(유지),
# 없으면 churn=1(이탈)
# cutoff_date는 데이터의 마지막 날짜에서 HORIZON_DAYS를 뺀 날짜로 자동 설정


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def main():
    print("=== churn_model_v2 시작 ===")
    print("작업 폴더:", os.getcwd())

    # -----------------------
    # 1) Load + 타입 정리
    # -----------------------
    df = pd.read_csv(CSV_PATH)

    required = ["user_id", "session_id", "watch_date"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {c}")

    df["watch_date"] = pd.to_datetime(df["watch_date"], errors="coerce")
    df = df.dropna(subset=["user_id", "session_id", "watch_date"])

    # 중복 세션 제거(안전)
    df = df.drop_duplicates(subset=["session_id"], keep="last")

    # 숫자형(있으면 사용)
    if "watch_duration_minutes" in df.columns:
        df["watch_duration_minutes"] = pd.to_numeric(df["watch_duration_minutes"], errors="coerce")
    else:
        df["watch_duration_minutes"] = np.nan

    if "watch_ratio" in df.columns:
        df["watch_ratio"] = pd.to_numeric(df["watch_ratio"], errors="coerce")
        df.loc[df["watch_ratio"] < 0, "watch_ratio"] = np.nan
    else:
        df["watch_ratio"] = np.nan

    if "completion_rate" in df.columns:
        df["completion_rate"] = pd.to_numeric(df["completion_rate"], errors="coerce")
    else:
        df["completion_rate"] = np.nan

    # -----------------------
    # 2) cutoff_date 자동 결정
    # -----------------------
    max_date = df["watch_date"].max()
    cutoff_date = max_date - pd.to_timedelta(HORIZON_DAYS, unit="D")

    print(f"데이터 기간: {df['watch_date'].min().date()} ~ {max_date.date()}")
    print("cutoff_date:", cutoff_date.date(), "| horizon_days:", HORIZON_DAYS, "| feature_window:", FEATURE_WINDOW_DAYS)

    # label window: (cutoff_date, cutoff_date + HORIZON_DAYS]
    label_end = cutoff_date + pd.to_timedelta(HORIZON_DAYS, unit="D")
    in_label = (df["watch_date"] > cutoff_date) & (df["watch_date"] <= label_end)

    # feature window: (cutoff_date - FEATURE_WINDOW_DAYS, cutoff_date]
    feat_start = cutoff_date - pd.to_timedelta(FEATURE_WINDOW_DAYS, unit="D")
    in_feat = (df["watch_date"] > feat_start) & (df["watch_date"] <= cutoff_date)

    feat_df = df.loc[in_feat].copy()
    label_df = df.loc[in_label].copy()

    # -----------------------
    # 3) 유저 단위 피처 생성 (1 user = 1 row)
    # -----------------------
    g = feat_df.groupby("user_id")

    user_feat = pd.DataFrame({
        # Frequency
        "sessions_30d": g["session_id"].nunique(),

        # Volume
        "watch_minutes_30d": g["watch_duration_minutes"].sum(min_count=1),

        # Engagement
        "avg_watch_ratio_30d": g["watch_ratio"].mean(),
        "median_watch_ratio_30d": g["watch_ratio"].median(),
        "avg_completion_rate_30d": g["completion_rate"].mean(),

        # Preference / Diversity
        "unique_title_30d": g["title"].nunique() if "title" in feat_df.columns else 0,
        "unique_genre_30d": g["genre_primary"].nunique() if "genre_primary" in feat_df.columns else 0,
    })

    # Recency: cutoff_date 기준 마지막 시청(피처 윈도우 내)과의 차이
    last_watch_in_feat = g["watch_date"].max().rename("last_watch_in_feat")
    user_feat = user_feat.join(last_watch_in_feat, on="user_id")
    user_feat["recency_days"] = (cutoff_date - user_feat["last_watch_in_feat"]).dt.days

    # 장르 엔트로피(가능하면)
    if "genre_primary" in feat_df.columns:
        genre_counts = (
            feat_df.groupby(["user_id", "genre_primary"])["session_id"]
            .nunique()
            .reset_index(name="cnt")
        )
        ent_map = {}
        for uid, sub in genre_counts.groupby("user_id"):
            ent_map[uid] = entropy_from_counts(sub["cnt"].to_numpy())
        user_feat["genre_entropy_30d"] = pd.Series(ent_map)
    else:
        user_feat["genre_entropy_30d"] = 0.0

    # 결측/무한 처리
    numeric_cols = [
        "sessions_30d", "watch_minutes_30d",
        "avg_watch_ratio_30d", "median_watch_ratio_30d",
        "avg_completion_rate_30d",
        "unique_title_30d", "unique_genre_30d",
        "recency_days", "genre_entropy_30d"
    ]
    for c in numeric_cols:
        user_feat[c] = pd.to_numeric(user_feat[c], errors="coerce")
    user_feat = user_feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    # -----------------------
    # 4) 라벨 생성: churn (이탈=1)
    # cutoff 이후 horizon 기간에 시청 세션이 0이면 churn=1
    # -----------------------
    label_sessions = label_df.groupby("user_id")["session_id"].nunique().rename("sessions_in_horizon")
    user_feat = user_feat.join(label_sessions, on="user_id")
    user_feat["sessions_in_horizon"] = user_feat["sessions_in_horizon"].fillna(0)

    # churn=1 (이탈): horizon에 세션 없음
    user_feat["churn"] = (user_feat["sessions_in_horizon"] == 0).astype(int)

    # label 분포 확인 (중요)
    print("\n[churn 분포]")
    print(user_feat["churn"].value_counts(dropna=False))

    # 만약 한 클래스만 있으면, horizon을 늘리거나(14->30) cutoff를 바꾸는 게 필요
    if user_feat["churn"].nunique() < 2:
        raise ValueError(
            "churn 라벨이 한 클래스만 존재합니다. "
            "HORIZON_DAYS를 30으로 바꾸거나, 데이터 기간/정의를 조정해야 합니다."
        )

    # -----------------------
    # 5) Train/Test split (Stratify)
    # -----------------------
    X = user_feat[numeric_cols].copy()
    y = user_feat["churn"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n[y_train 분포]")
    print(y_train.value_counts())
    print("\n[y_test 분포]")
    print(y_test.value_counts())

    # -----------------------
    # 6) Baseline (휴리스틱)
    # 예: recency_days가 horizon보다 크면 churn=1
    # -----------------------
    baseline_pred = (X_test["recency_days"] > HORIZON_DAYS).astype(int)
    print("\n[Baseline]")
    print("F1:", f1_score(y_test, baseline_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, baseline_pred))

    # -----------------------
    # 7) Models + Stratified CV
    # -----------------------
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

    lr_cv_auc = cross_val_score(lr, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    rf_cv_auc = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    print("\n[CV AUC]")
    print("Logistic Regression CV AUC:", lr_cv_auc)
    print("Random Forest    CV AUC:", rf_cv_auc)

    # fit + test
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print("\n[Test AUC]")
    print("Logistic Regression Test AUC:", lr_auc)
    print("Random Forest    Test AUC:", rf_auc)

    print("\n[Classification Report: Logistic Regression]")
    print(classification_report(y_test, lr.predict(X_test), digits=4))

    print("\n[Classification Report: Random Forest]")
    print(classification_report(y_test, rf.predict(X_test), digits=4))

    # -----------------------
    # 8) 중요 피처 확인
    # -----------------------
    feat_names = numeric_cols

    lr_coef = lr.named_steps["clf"].coef_[0]
    lr_rank = sorted(zip(feat_names, lr_coef), key=lambda x: abs(x[1]), reverse=True)[:10]
    print("\n[Top predictors - Logistic Regression (|coef|)]")
    for name, val in lr_rank:
        print(f"{name:25s} coef={val:.6f}")

    rf_rank = sorted(zip(feat_names, rf.feature_importances_), key=lambda x: x[1], reverse=True)[:10]
    print("\n[Top predictors - Random Forest]")
    for name, val in rf_rank:
        print(f"{name:25s} importance={val:.6f}")

    # -----------------------
    # 9) 저장 (반드시 생성되게)
    # -----------------------
    user_feat_out = user_feat.reset_index(drop=False).rename(columns={"index": "user_id"})
    out_path = "user_features_churn.csv"
    user_feat_out.to_csv(out_path, index=False)

    print("\n저장 완료:", out_path)
    print("유저 수:", len(user_feat_out), "| churn mean:", float(user_feat_out["churn"].mean()))
    print("=== churn_model_v2 끝 ===")


if __name__ == "__main__":
    main()