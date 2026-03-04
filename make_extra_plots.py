import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

H14_PATH = "user_features_churn_full_h14.csv"
H30_PATH = "user_features_churn_full_h30.csv"

# 넷플릭스 오리지널 share 계산용(있으면 사용, 없으면 스킵)
WATCH_JOINED_PATH = "watch_joined.csv"

# 출력 파일명
OUT_GENRE_H14 = "genre_vs_churn_h14.png"
OUT_GENRE_H30 = "genre_vs_churn_h30.png"

OUT_DEVICE_H14 = "device_vs_churn_h14.png"
OUT_DEVICE_H30 = "device_vs_churn_h30.png"

OUT_ORIGINAL_BAR_H14 = "original_vs_churn_h14.png"
OUT_ORIGINAL_BAR_H30 = "original_vs_churn_h30.png"

OUT_ORIGINAL_DECI_H14 = "original_share_deciles_vs_churn_h14.png"
OUT_ORIGINAL_DECI_H30 = "original_share_deciles_vs_churn_h30.png"


def _save_barh(labels, values, title, xlabel, out_png, xlim=(0, 1)):
    plt.figure(figsize=(8, max(4, len(labels) * 0.35)))
    plt.barh(list(reversed(labels)), list(reversed(values)))
    plt.title(title)
    plt.xlabel(xlabel)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png)


def plot_sharecols_vs_churn(
    df: pd.DataFrame,
    share_prefix: str,
    tag: str,
    out_png: str,
    top_quantile: float = 0.8,
    min_users: int = 30,
    exclude_cols=None,
):
    """
    share_prefix (예: 'genre_share_' 또는 'device_share_') 로 시작하는 컬럼들에 대해
    '상위 (1-top_quantile)% heavy viewers' 집단의 churn rate를 비교 barh로 그림.
    """
    if exclude_cols is None:
        exclude_cols = set()

    if "churn" not in df.columns:
        print(f"[Skip] churn not found in {tag}")
        return

    cols = [c for c in df.columns if c.startswith(share_prefix)]
    cols = [c for c in cols if c not in exclude_cols]

    if len(cols) == 0:
        print(f"[Skip] {share_prefix}* columns not found in {tag}")
        return

    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        thr = float(s.quantile(top_quantile))
        if thr <= 0:
            continue

        mask = s >= thr
        n = int(mask.sum())
        if n < min_users:
            continue

        churn_rate = float(df.loc[mask, "churn"].mean())
        label = col.replace(share_prefix, "")
        rows.append((label, churn_rate, n, thr))

    if len(rows) == 0:
        print(f"[Skip] Not enough signal for {share_prefix} in {tag}. (shares mostly 0?)")
        return

    # churn 낮은 순 정렬
    rows.sort(key=lambda x: x[1])

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    title = f"{share_prefix[:-1].title()} vs Churn Rate (Top {int((1-top_quantile)*100)}% heavy viewers) - {tag}"
    _save_barh(labels, values, title, "Churn rate", out_png, xlim=(0, 1))


def compute_original_share_from_watch_joined(
    user_df: pd.DataFrame,
    watch_joined_path: str,
    feature_window_days: int = 30,
) -> pd.Series:
    """
    user_df(유저 피처 데이터) 안의 user_id + cutoff_date를 이용해서,
    watch_joined.csv에서 cutoff 기준 과거 feature_window_days 동안
    is_netflix_original=1(또는 True) 세션 비율(original_share)을 유저별로 계산.
    """
    if not os.path.exists(watch_joined_path):
        raise FileNotFoundError(watch_joined_path)

    w = pd.read_csv(watch_joined_path)

    need = ["user_id", "session_id", "watch_date", "is_netflix_original"]
    for c in need:
        if c not in w.columns:
            raise ValueError(f"watch_joined.csv에 필수 컬럼이 없습니다: {c}")

    w["watch_date"] = pd.to_datetime(w["watch_date"], errors="coerce")
    w = w.dropna(subset=["user_id", "session_id", "watch_date"])

    # session 중복 안전 제거
    w = w.drop_duplicates(subset=["session_id"], keep="last")

    # is_netflix_original 정리(0/1, True/False 모두 처리)
    def to01(x):
        if pd.isna(x):
            return 0
        if isinstance(x, (bool, np.bool_)):
            return int(x)
        try:
            return int(float(x))
        except Exception:
            s = str(x).strip().lower()
            return 1 if s in ["true", "t", "yes", "y"] else 0

    w["is_netflix_original"] = w["is_netflix_original"].apply(to01)

    # cutoff_date 읽기
    if "cutoff_date" not in user_df.columns:
        raise ValueError("user_features 파일에 cutoff_date 컬럼이 없습니다. (원본 생성 코드에서 유지되는 컬럼)")
    cutoff = pd.to_datetime(user_df["cutoff_date"].iloc[0], errors="coerce")
    if pd.isna(cutoff):
        raise ValueError("cutoff_date 파싱 실패. user_features 파일의 cutoff_date 값을 확인하세요.")

    start = cutoff - pd.to_timedelta(feature_window_days, unit="D")

    # feature window 내 로그만
    w = w[(w["watch_date"] > start) & (w["watch_date"] <= cutoff)]

    # 이번 user_df에 있는 유저만
    users = set(user_df["user_id"].astype(str).tolist())
    w["user_id"] = w["user_id"].astype(str)
    w = w[w["user_id"].isin(users)]

    if w.empty:
        # 전부 0으로 반환
        return pd.Series(0.0, index=user_df["user_id"].astype(str).tolist(), name="original_share")

    g = w.groupby("user_id")["is_netflix_original"]
    # 세션 기준 평균 = original 세션 비율
    original_share = g.mean().rename("original_share")

    # user_df에 없는 유저도 0 채우기
    full = pd.Series(0.0, index=user_df["user_id"].astype(str).tolist(), name="original_share")
    full.update(original_share)
    return full


def plot_original_vs_churn(user_df: pd.DataFrame, tag: str, out_png_bar: str, out_png_deciles: str):
    """
    original_share를 계산한 다음:
    1) 상위 20% vs 나머지 churn rate bar
    2) original_share decile(10분위)별 churn rate line/bar
    """
    if "churn" not in user_df.columns:
        print(f"[Skip] churn not found in {tag}")
        return

    # original_share 확보: 있으면 쓰고, 없으면 watch_joined로 계산 시도
    if "original_share" in user_df.columns:
        s = pd.to_numeric(user_df["original_share"], errors="coerce").fillna(0.0)
    else:
        if not os.path.exists(WATCH_JOINED_PATH):
            print(f"[Skip] {tag}: watch_joined.csv가 없어서 넷플릭스 오리지널 그래프를 만들 수 없습니다.")
            print("       (watch_joined.csv가 프로젝트 폴더에 있어야 original_share 계산이 가능합니다.)")
            return
        try:
            s = compute_original_share_from_watch_joined(user_df, WATCH_JOINED_PATH, feature_window_days=30)
        except Exception as e:
            print(f"[Skip] {tag}: original_share 계산 실패:", repr(e))
            return

    df = user_df.copy()
    df["original_share"] = s.values
    df["original_share"] = df["original_share"].clip(lower=0, upper=1)

    # (1) 상위 20% heavy original viewers vs rest
    q = df["original_share"].quantile(0.8)
    heavy = df["original_share"] >= q
    if heavy.sum() < 30:
        print(f"[Skip] {tag}: original_share 상위 20% 유저가 너무 적습니다.")
    else:
        churn_heavy = float(df.loc[heavy, "churn"].mean())
        churn_rest = float(df.loc[~heavy, "churn"].mean())

        plt.figure()
        plt.bar(["Top20% Original-heavy", "Rest"], [churn_heavy, churn_rest])
        plt.title(f"Netflix Original Share vs Churn - {tag}")
        plt.ylabel("Churn rate")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_png_bar, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved:", out_png_bar)

    # (2) decile별 churn
    # decile bin이 너무 한쪽으로 몰리면 qcut이 실패할 수 있어서 예외 처리
    try:
        df["decile"] = pd.qcut(df["original_share"], 10, labels=False, duplicates="drop")
        grp = df.groupby("decile")["churn"].mean()
        plt.figure()
        plt.plot(grp.index, grp.values, marker="o")
        plt.title(f"Churn rate by Netflix Original Share deciles - {tag}")
        plt.xlabel("Original share decile (0=low, 9=high)")
        plt.ylabel("Churn rate")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_png_deciles, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved:", out_png_deciles)
    except Exception as e:
        print(f"[Skip] {tag}: decile plot 생성 실패:", repr(e))


def main():
    df14 = pd.read_csv(H14_PATH)
    df30 = pd.read_csv(H30_PATH)

    # user_id 타입 통일
    df14["user_id"] = df14["user_id"].astype(str)
    df30["user_id"] = df30["user_id"].astype(str)

    # 1) 장르 vs churn (이미 genre_share_*가 있을 때)
    plot_sharecols_vs_churn(
        df14, "genre_share_", "HORIZON=14", OUT_GENRE_H14,
        top_quantile=0.8, min_users=30, exclude_cols={"genre_share_other"}
    )
    plot_sharecols_vs_churn(
        df30, "genre_share_", "HORIZON=30", OUT_GENRE_H30,
        top_quantile=0.8, min_users=30, exclude_cols={"genre_share_other"}
    )

    # 2) 시청 방법(디바이스) vs churn (device_share_*가 있을 때)
    plot_sharecols_vs_churn(
        df14, "device_share_", "HORIZON=14", OUT_DEVICE_H14,
        top_quantile=0.8, min_users=30
    )
    plot_sharecols_vs_churn(
        df30, "device_share_", "HORIZON=30", OUT_DEVICE_H30,
        top_quantile=0.8, min_users=30
    )

    # 3) 넷플릭스 오리지널 여부 vs churn
    # - user_features에 original_share 컬럼이 있으면 바로 사용
    # - 없으면 watch_joined.csv에서 is_netflix_original을 이용해 feature window 기준 original_share 계산
    plot_original_vs_churn(df14, "HORIZON=14", OUT_ORIGINAL_BAR_H14, OUT_ORIGINAL_DECI_H14)
    plot_original_vs_churn(df30, "HORIZON=30", OUT_ORIGINAL_BAR_H30, OUT_ORIGINAL_DECI_H30)

    print("Done.")


if __name__ == "__main__":
    main()