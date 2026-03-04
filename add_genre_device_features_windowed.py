import pandas as pd
import numpy as np

WATCH_PATH = "watch_joined.csv"
USER_FEAT_PATH = "user_features_churn.csv"
OUT_PATH = "user_features_churn_plus.csv"

# user_features_churn.csv가 "30일 행동 → 14일 horizon"으로 만들어졌다는 전제
FEATURE_WINDOW_DAYS = 30
HORIZON_DAYS = 14

TOP_GENRES = 10
TOP_DEVICES = 5


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def top_share_pivot(df, user_col, cat_col, top_k, prefix):
    tmp = df[[user_col, cat_col]].dropna()
    if tmp.empty:
        return pd.DataFrame(columns=[])

    cnt = tmp.groupby([user_col, cat_col]).size().rename("n").reset_index()

    top_cats = (
        cnt.groupby(cat_col)["n"].sum()
        .sort_values(ascending=False)
        .head(top_k)
        .index
        .tolist()
    )

    user_total = cnt.groupby(user_col)["n"].sum().rename("total")
    cnt = cnt.join(user_total, on=user_col)
    cnt["share"] = cnt["n"] / cnt["total"]

    cnt = cnt[cnt[cat_col].isin(top_cats)]
    pivot = cnt.pivot(index=user_col, columns=cat_col, values="share").fillna(0.0)
    pivot.columns = [f"{prefix}{str(c)}" for c in pivot.columns]
    return pivot


def top_category(df, user_col, cat_col, new_col):
    tmp = df[[user_col, cat_col]].dropna()
    if tmp.empty:
        return pd.Series(dtype="object", name=new_col)

    cnt = tmp.groupby([user_col, cat_col]).size().rename("n").reset_index()
    idx = cnt.groupby(user_col)["n"].idxmax()
    s = cnt.loc[idx, [user_col, cat_col]].set_index(user_col)[cat_col]
    s.name = new_col
    return s


def diversity_count(df, user_col, cat_col, new_col):
    tmp = df[[user_col, cat_col]].dropna()
    if tmp.empty:
        return pd.Series(dtype="float64", name=new_col)
    s = tmp.groupby(user_col)[cat_col].nunique()
    s.name = new_col
    return s


def entropy_per_user(df, user_col, cat_col, new_col):
    tmp = df[[user_col, cat_col]].dropna()
    if tmp.empty:
        return pd.Series(dtype="float64", name=new_col)

    cnt = tmp.groupby([user_col, cat_col]).size().rename("n").reset_index()
    ent_map = {}
    for uid, sub in cnt.groupby(user_col):
        ent_map[uid] = entropy_from_counts(sub["n"].to_numpy())
    return pd.Series(ent_map, name=new_col)


def main():
    watch = pd.read_csv(WATCH_PATH)
    user_feat = pd.read_csv(USER_FEAT_PATH)

    watch["watch_date"] = pd.to_datetime(watch["watch_date"], errors="coerce")
    watch = watch.dropna(subset=["user_id", "watch_date"])

    # ====== 중요: 기존 user_features_churn과 동일한 컷오프/윈도우로 맞추기 ======
    max_date = watch["watch_date"].max()
    cutoff_date = max_date - pd.to_timedelta(HORIZON_DAYS, unit="D")
    feat_start = cutoff_date - pd.to_timedelta(FEATURE_WINDOW_DAYS, unit="D")

    in_feat = (watch["watch_date"] > feat_start) & (watch["watch_date"] <= cutoff_date)
    feat_watch = watch.loc[in_feat].copy()

    print("cutoff_date:", cutoff_date.date(),
          "| feature window:", feat_start.date(), "~", cutoff_date.date(),
          "| feat rows:", len(feat_watch))

    # user list (유저 피처 파일에 있는 유저만 대상으로)
    uindex = pd.Index(user_feat["user_id"].unique(), name="user_id")
    feats = pd.DataFrame(index=uindex)

    # ====== Genre features ======
    if "genre_primary" in feat_watch.columns:
        feats = feats.join(top_category(feat_watch, "user_id", "genre_primary", "top_genre"), how="left")
        feats = feats.join(diversity_count(feat_watch, "user_id", "genre_primary", "genre_diversity"), how="left")
        feats = feats.join(entropy_per_user(feat_watch, "user_id", "genre_primary", "genre_entropy"), how="left")
        feats = feats.join(top_share_pivot(feat_watch, "user_id", "genre_primary", TOP_GENRES, "genre_share_"), how="left")
    else:
        print("[주의] genre_primary 없음 → 장르 피처 스킵")

    # ====== Device features ======
    if "device_type" in feat_watch.columns:
        feats = feats.join(top_category(feat_watch, "user_id", "device_type", "top_device"), how="left")
        feats = feats.join(diversity_count(feat_watch, "user_id", "device_type", "device_diversity"), how="left")
        feats = feats.join(top_share_pivot(feat_watch, "user_id", "device_type", TOP_DEVICES, "device_share_"), how="left")
    else:
        print("[주의] device_type 없음 → 디바이스 피처 스킵")

    # 수치형 결측은 0
    for col in feats.columns:
        if feats[col].dtype.kind in "biufc":
            feats[col] = feats[col].fillna(0)

    out = user_feat.merge(feats.reset_index(), on="user_id", how="left")
    out.to_csv(OUT_PATH, index=False)

    print("완료! 저장:", OUT_PATH)
    print("추가된 컬럼 수:", out.shape[1] - user_feat.shape[1])
    print("최종 shape:", out.shape)


if __name__ == "__main__":
    main()