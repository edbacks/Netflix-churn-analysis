import argparse
import os
import numpy as np
import pandas as pd

# =========================
# 설정(원하면 수정)
# =========================
WATCH_PATH_DEFAULT = "watch_joined.csv"

# feature는 7/14/30일을 만들기 때문에 최소 30일치 window가 필요
FEATURE_MAX_WINDOW_DAYS = 30

TOP_GENRES = 10
TOP_DEVICES = 5


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def top_share_pivot(df, user_col: str, cat_col: str, top_k: int, prefix: str) -> pd.DataFrame:
    """유저별 카테고리 비율(세션 기준) pivot. 전체 상위 top_k 카테고리만 컬럼화."""
    tmp = df[[user_col, cat_col, "session_id"]].dropna()
    if tmp.empty:
        return pd.DataFrame()

    cnt = tmp.groupby([user_col, cat_col])["session_id"].nunique().rename("n").reset_index()

    top_cats = (
        cnt.groupby(cat_col)["n"].sum()
        .sort_values(ascending=False)
        .head(top_k).index.tolist()
    )

    user_total = cnt.groupby(user_col)["n"].sum().rename("total")
    cnt = cnt.join(user_total, on=user_col)
    cnt["share"] = cnt["n"] / cnt["total"]

    cnt = cnt[cnt[cat_col].isin(top_cats)]
    pivot = cnt.pivot(index=user_col, columns=cat_col, values="share").fillna(0.0)

    pivot.columns = [f"{prefix}{str(c)}" for c in pivot.columns]
    return pivot


def build_user_features(feat_watch: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    cutoff_date 이전 로그(feat_watch)로 유저 단위 피처 생성.
    (7/14/30일 Frequency/Volume + Recency + Consistency + 장르/디바이스 share)
    """
    df = feat_watch.copy()

    # 기본 정리
    df["watch_date"] = pd.to_datetime(df["watch_date"], errors="coerce")
    df = df.dropna(subset=["user_id", "session_id", "watch_date"])

    # 숫자형
    df["watch_duration_minutes"] = pd.to_numeric(df.get("watch_duration_minutes"), errors="coerce")
    df["watch_ratio"] = pd.to_numeric(df.get("watch_ratio"), errors="coerce")
    df.loc[df["watch_ratio"] < 0, "watch_ratio"] = np.nan
    df["completion_rate"] = pd.to_numeric(df.get("completion_rate"), errors="coerce")

    users = df["user_id"].unique()
    out = pd.DataFrame(index=users)
    out.index.name = "user_id"

    # Recency: cutoff 기준 마지막 시청일
    last_watch = df.groupby("user_id")["watch_date"].max()
    out["recency_days"] = (cutoff_date - last_watch).dt.days

    # 윈도우 슬라이스 함수
    def window_df(days: int) -> pd.DataFrame:
        start = cutoff_date - pd.to_timedelta(days, unit="D")
        return df[(df["watch_date"] > start) & (df["watch_date"] <= cutoff_date)]

    # Frequency/Volume/Engagement: 7/14/30
    for d in [7, 14, 30]:
        w = window_df(d)
        g = w.groupby("user_id")
        out[f"sessions_{d}d"] = g["session_id"].nunique()
        out[f"watch_minutes_{d}d"] = g["watch_duration_minutes"].sum(min_count=1)
        out[f"avg_watch_ratio_{d}d"] = g["watch_ratio"].mean()
        out[f"avg_completion_rate_{d}d"] = g["completion_rate"].mean()

    # 결측 → 0
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # =========================
    # Consistency (규칙성) - "시간대"는 없음(날짜만 있으므로)
    # 1) 요일 분포 엔트로피(30일)
    # 2) 주말 비율(30일)
    # 3) 시청일 간격 평균/표준편차(30일)
    # =========================
    w30 = window_df(30).copy()
    if not w30.empty:
        w30["dow"] = w30["watch_date"].dt.dayofweek  # Mon=0..Sun=6
        # 세션 기준 요일 분포
        dow_cnt = w30.groupby(["user_id", "dow"])["session_id"].nunique().rename("n").reset_index()

        ent_map = {}
        weekend_map = {}

        for uid, sub in dow_cnt.groupby("user_id"):
            counts = sub["n"].to_numpy()
            ent_map[uid] = entropy_from_counts(counts)

            weekend_n = sub.loc[sub["dow"].isin([5, 6]), "n"].sum()
            total_n = sub["n"].sum()
            weekend_map[uid] = float(weekend_n / total_n) if total_n > 0 else 0.0

        out["dow_entropy_30d"] = pd.Series(ent_map).fillna(0.0)
        out["weekend_share_30d"] = pd.Series(weekend_map).fillna(0.0)

        # 시청 "일자" 간격 규칙성(같은 날 여러 세션은 1일로 압축)
        days_per_user = (
            w30.assign(day=w30["watch_date"].dt.date)[["user_id", "day"]]
            .drop_duplicates()
            .sort_values(["user_id", "day"])
        )

        gap_mean = {}
        gap_std = {}
        for uid, sub in days_per_user.groupby("user_id"):
            dts = pd.to_datetime(sub["day"])
            if len(dts) <= 1:
                gap_mean[uid] = 0.0
                gap_std[uid] = 0.0
            else:
                gaps = dts.diff().dt.days.dropna()
                gap_mean[uid] = float(gaps.mean())
                gap_std[uid] = float(gaps.std(ddof=0))

        out["gap_mean_days_30d"] = pd.Series(gap_mean).fillna(0.0)
        out["gap_std_days_30d"] = pd.Series(gap_std).fillna(0.0)
    else:
        out["dow_entropy_30d"] = 0.0
        out["weekend_share_30d"] = 0.0
        out["gap_mean_days_30d"] = 0.0
        out["gap_std_days_30d"] = 0.0

    # =========================
    # 장르/디바이스 피처 (30일 window 기준)
    # =========================
    # genre share: 상위 TOP_GENRES만 pivot + 기타(other)
    if "genre_primary" in w30.columns:
        genre_share = top_share_pivot(w30, "user_id", "genre_primary", TOP_GENRES, "genre_share_")
        out = out.join(genre_share, how="left")
        share_cols = [c for c in out.columns if c.startswith("genre_share_")]
        if share_cols:
            out["genre_share_other"] = 1.0 - out[share_cols].sum(axis=1)
            out.loc[out["genre_share_other"] < 0, "genre_share_other"] = 0.0
        else:
            out["genre_share_other"] = 0.0
    else:
        out["genre_share_other"] = 0.0

    # device share: 상위 TOP_DEVICES만 pivot (+ 합 체크용)
    if "device_type" in w30.columns:
        device_share = top_share_pivot(w30, "user_id", "device_type", TOP_DEVICES, "device_share_")
        out = out.join(device_share, how="left")
        dcols = [c for c in out.columns if c.startswith("device_share_")]
        if dcols:
            out["device_share_sum_top"] = out[dcols].sum(axis=1)
        else:
            out["device_share_sum_top"] = 0.0
    else:
        out["device_share_sum_top"] = 0.0

    # 최종 결측/무한 정리
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def build_churn_dataset(watch: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    horizon_days(14/30 등)에 따라
    - cutoff_date = max_date - horizon_days
    - cutoff 이전 30일로 feature 생성
    - cutoff 이후 horizon_days 동안 활동 없으면 churn=1
    """
    max_date = watch["watch_date"].max()
    cutoff_date = max_date - pd.to_timedelta(horizon_days, unit="D")

    feat_start = cutoff_date - pd.to_timedelta(FEATURE_MAX_WINDOW_DAYS, unit="D")
    label_end = cutoff_date + pd.to_timedelta(horizon_days, unit="D")

    in_feat = (watch["watch_date"] > feat_start) & (watch["watch_date"] <= cutoff_date)
    in_label = (watch["watch_date"] > cutoff_date) & (watch["watch_date"] <= label_end)

    feat_watch = watch.loc[in_feat].copy()
    label_watch = watch.loc[in_label].copy()

    user_feat = build_user_features(feat_watch, cutoff_date)

    # 라벨: horizon 동안 세션 수
    sessions_in_h = label_watch.groupby("user_id")["session_id"].nunique().rename("sessions_in_horizon")
    user_feat = user_feat.join(sessions_in_h, how="left")
    user_feat["sessions_in_horizon"] = user_feat["sessions_in_horizon"].fillna(0)

    user_feat["churn"] = (user_feat["sessions_in_horizon"] == 0).astype(int)

    # 메타 컬럼(디버깅용)
    user_feat["horizon_days"] = horizon_days
    user_feat["cutoff_date"] = str(cutoff_date.date())
    user_feat["feature_window_days"] = FEATURE_MAX_WINDOW_DAYS

    # user_id 컬럼으로 내보내기
    out = user_feat.reset_index()

    # 간단 검증(라벨 무결성)
    bad = out[((out["sessions_in_horizon"] == 0) & (out["churn"] != 1)) |
              ((out["sessions_in_horizon"] > 0) & (out["churn"] != 0))]
    if len(bad) > 0:
        raise ValueError(f"라벨 무결성 오류 발견: {len(bad)}건")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", default=WATCH_PATH_DEFAULT, help="watch_joined.csv 경로")
    parser.add_argument("--horizon", type=int, default=14, help="churn 판단 기간(예: 14 또는 30)")
    parser.add_argument("--out", default=None, help="출력 파일명(미지정 시 자동)")
    args = parser.parse_args()

    print("=== make_churn_dataset_full 시작 ===")
    print("작업 폴더:", os.getcwd())

    watch = pd.read_csv(args.watch)

    # 필수 컬럼 체크
    need = ["user_id", "session_id", "watch_date"]
    for c in need:
        if c not in watch.columns:
            raise ValueError(f"필수 컬럼 누락: {c}")

    watch["watch_date"] = pd.to_datetime(watch["watch_date"], errors="coerce")
    watch = watch.dropna(subset=["user_id", "session_id", "watch_date"])

    # 세션 중복 제거(안전)
    watch = watch.drop_duplicates(subset=["session_id"], keep="last")

    # churn 데이터 생성
    ds = build_churn_dataset(watch, horizon_days=args.horizon)

    # 분포 출력
    print("\n[churn 분포]")
    print(ds["churn"].value_counts(dropna=False))
    print("churn rate:", float(ds["churn"].mean()))
    print("유저 수:", ds["user_id"].nunique(), "| 행 수:", len(ds), "| 컬럼 수:", ds.shape[1])

    # 저장
    out_path = args.out or f"user_features_churn_full_h{args.horizon}.csv"
    ds.to_csv(out_path, index=False)
    print("\n저장 완료:", out_path)
    print("=== 끝 ===")


if __name__ == "__main__":
    main()