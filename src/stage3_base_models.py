"""Стадия 3. Базовые модели: популярная и контентная (TF-IDF)."""

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from .config import (
    DATA_DIR,
    MODELS_DIR,
    SPLIT_DIR,
    TABLES_DIR,
    ensure_dirs,
    load_params,
)
from .io_utils import load_index_maps, load_raw_data, make_id2idx, prepare_book_tags
from .models import (
    META_TAGS,
    build_book_profiles,
    build_favorites,
    find_demo_book,
    fit_item_profiles,
    get_similar_books,
    popularity_scores,
)


def main():
    params = load_params()
    ensure_dirs()

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    user_ids, book_ids = load_index_maps(SPLIT_DIR / "index_maps.npz")
    n_users, n_books = len(user_ids), len(book_ids)
    bid2i = make_id2idx(book_ids)

    _, books, tags, book_tags = load_raw_data()
    bt = prepare_book_tags(books, book_tags, tags)
    books_by_id = books.set_index("id")

    # Популярная модель: средний рейтинг при достаточном числе оценок
    pop_scores, pop_stats = popularity_scores(
        train, n_books, params["min_ratings_popular"]
    )
    np.save(MODELS_DIR / "pop_scores.npy", pop_scores)
    pop_stats.to_csv(TABLES_DIR / "pop_stats.csv", index_label="book_idx")
    n_pass = int((pop_scores > 0).sum())
    print(
        f"Книг проходят порог {params['min_ratings_popular']} оценок: {n_pass} из {n_books}"
    )

    pop_top = (
        pop_stats[pop_stats.n >= params["min_ratings_popular"]]
        .nlargest(10, "mean")
        .reset_index()
    )
    pop_top["book_id"] = book_ids[pop_top["book_idx"].values]
    pop_top["title"] = books_by_id.loc[pop_top["book_id"].values, "title"].values
    pop_top["authors"] = books_by_id.loc[pop_top["book_id"].values, "authors"].values
    pop_top[["book_id", "title", "authors", "n", "mean"]].to_csv(
        TABLES_DIR / "popular_top10_train.csv", index=False
    )

    # Контентная модель: профиль книги из названия и тегов
    profiles = build_book_profiles(books, bt, book_ids, bid2i, META_TAGS)
    item_profiles = fit_item_profiles(profiles, min_df=params["content"]["min_df"])
    save_npz(MODELS_DIR / "item_profiles.npz", item_profiles)
    print(
        f"Матрица профилей: {item_profiles.shape}, ненулевых элементов: {item_profiles.nnz}"
    )

    demo_rows = []
    for pattern in ("Hunger Games", "Hobbit", "1984"):
        book_id = find_demo_book(books, pattern)
        idx = bid2i[book_id]
        top_idx, sims = get_similar_books(item_profiles, idx, n=5)
        source_title = books_by_id.loc[book_id, "title"]
        print(f'Похожие на "{source_title}" (book_id={book_id}):')
        for rank, (j, s) in enumerate(zip(top_idx, sims), start=1):
            nb_id = int(book_ids[j])
            print(f"  {rank}. {books_by_id.loc[nb_id, 'title']} (cos={s:.3f})")
            demo_rows.append(
                {
                    "source_book_id": book_id,
                    "source_title": source_title,
                    "rank": rank,
                    "book_id": nb_id,
                    "title": books_by_id.loc[nb_id, "title"],
                    "authors": books_by_id.loc[nb_id, "authors"],
                    "cosine": round(float(s), 3),
                }
            )
    pd.DataFrame(demo_rows).to_csv(TABLES_DIR / "similar_books_demo.csv", index=False)

    # Профили пользователей по любимым книгам
    fav_train = build_favorites(train, n_users, n_books, params["relevance_threshold"])
    save_npz(MODELS_DIR / "fav_train.npz", fav_train)
    print(f"fav_train: {fav_train.shape}, ненулевых элементов: {fav_train.nnz}")


if __name__ == "__main__":
    main()
