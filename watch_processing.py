import pandas as pd
import numpy as np

# 1) 로드
movies = pd.read_csv("movies.csv")
watch = pd.read_csv("watch_history.csv")

# 2) movies 중복 제거: movie_id 기준으로 added_to_platform 가장 최신을 남김
movies["added_to_platform"] = pd.to_datetime(movies["added_to_platform"], errors="coerce")
movies = movies.sort_values(["movie_id", "added_to_platform"], ascending=[True, True])
movies = movies.drop_duplicates(subset=["movie_id"], keep="last")

# 3) 타입 정리
movies["duration_minutes"] = pd.to_numeric(movies["duration_minutes"], errors="coerce")
movies.loc[movies["duration_minutes"] <= 0, "duration_minutes"] = np.nan  # 0/음수는 결측으로

watch["watch_date"] = pd.to_datetime(watch["watch_date"], errors="coerce")
watch["watch_duration_minutes"] = pd.to_numeric(watch["watch_duration_minutes"], errors="coerce")
watch["progress_percentage"] = pd.to_numeric(watch["progress_percentage"], errors="coerce")

# 4) 조인 (watch 로그에 콘텐츠 메타 붙이기)
df = watch.merge(
    movies[[
        "movie_id","title","content_type","genre_primary","genre_secondary",
        "release_year","duration_minutes","rating","language","country_of_origin",
        "imdb_rating","is_netflix_original"
    ]],
    on="movie_id",
    how="left"
)

# 5) 핵심 지표 생성
# (a) 완주율(0~1) : progress_percentage가 신뢰 가능하면 우선 사용
df["completion_rate"] = df["progress_percentage"] / 100.0

# (b) watch_ratio : 실제 본 시간 / 콘텐츠 길이 (duration이 있을 때만)
df["watch_ratio"] = df["watch_duration_minutes"] / df["duration_minutes"]

# (c) 이상치 정리(선택): watch_ratio가 너무 큰 값(예: 3배)은 로그 오류로 보고 결측 처리
df.loc[(df["watch_ratio"] < 0) | (df["watch_ratio"] > 3), "watch_ratio"] = np.nan
df.loc[(df["completion_rate"] < 0) | (df["completion_rate"] > 1), "completion_rate"] = np.nan

# 6) 저장
df.to_csv("watch_joined.csv", index=False)
print("완료:", df.shape, "저장: watch_joined.csv")

import pandas as pd

movies = pd.read_csv("movies.csv")
watch = pd.read_csv("watch_history.csv")

# movie_id 집합 생성
movies_ids = set(movies["movie_id"])
watch_ids = set(watch["movie_id"])

print("movies 고유 movie_id 수:", len(movies_ids))
print("watch 고유 movie_id 수:", len(watch_ids))

# 교집합
common_ids = movies_ids & watch_ids

# movies에는 있는데 watch에는 없는 것
only_in_movies = movies_ids - watch_ids

# watch에는 있는데 movies에는 없는 것
only_in_watch = watch_ids - movies_ids

print("공통 movie_id 수:", len(common_ids))
print("movies에만 있는 movie_id 수:", len(only_in_movies))
print("watch에만 있는 movie_id 수:", len(only_in_watch))

