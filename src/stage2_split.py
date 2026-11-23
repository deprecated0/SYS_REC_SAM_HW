"""Стадия 2. Разделение данных на train и test."""

import numpy as np
import pandas as pd

from .config import SPLIT_DIR, TABLES_DIR, ensure_dirs, load_params
from .io_utils import (
    add_indices,
    build_index_maps,
    load_raw_data,
    make_id2idx,
    save_index_maps,
    save_json,
)


def main():
    params = load_params()
    ensure_dirs()
    ratings, _, _, _ = load_raw_data()

    min_train = params["min_train_ratings"]
    holdout = params["test_holdout"]

    # Holdout только у пользователей с >= min_train оценками,
    # чтобы у каждого из них в train осталось хотя бы несколько взаимодействий.
    user_counts = ratings.user_id.value_counts()
    eval_eligible = user_counts.index[user_counts >= min_train]

    shuffled = (
        ratings[ratings.user_id.isin(eval_eligible)]
        .sample(frac=1.0, random_state=params["random_state"])
        .reset_index(drop=True)
    )
    shuffled["rank_in_user"] = shuffled.groupby("user_id").cumcount()
    test = shuffled[shuffled.rank_in_user < holdout].drop(columns="rank_in_user")
    train = pd.concat(
        [
            shuffled[shuffled.rank_in_user >= holdout].drop(columns="rank_in_user"),
            ratings[~ratings.user_id.isin(eval_eligible)],
        ],
        ignore_index=True,
    )

    user_ids, book_ids = build_index_maps(ratings)
    uid2i = make_id2idx(user_ids)
    bid2i = make_id2idx(book_ids)
    train = add_indices(train, uid2i, bid2i)
    test = add_indices(test, uid2i, bid2i)

    train.to_csv(SPLIT_DIR / "train.csv", index=False)
    test.to_csv(SPLIT_DIR / "test.csv", index=False)
    save_index_maps(SPLIT_DIR / "index_maps.npz", user_ids, book_ids)

    train_counts = (
        train.groupby("user_idx").size().reindex(np.arange(len(user_ids)), fill_value=0)
    )
    summary = {
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "test_share_percent": round(len(test) / len(ratings) * 100, 1),
        "eval_users": int(test.user_id.nunique()),
        "total_users": int(len(user_ids)),
        "total_books": int(len(book_ids)),
        "users_without_train": int((train_counts == 0).sum()),
        "users_lt5_train": int((train_counts < 5).sum()),
        "users_lt5_train_percent": round(float((train_counts < 5).mean() * 100), 1),
        "books_without_train": int(len(book_ids) - train.book_idx.nunique()),
    }
    save_json(summary, TABLES_DIR / "split_summary.json")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
