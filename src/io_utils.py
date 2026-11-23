"""Чтение данных, индексные отображения, стиль графиков."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix

from .config import DATA_DIR, FIGURES_DIR


def setup_plots():
    """Единый стиль графиков."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["figure.figsize"] = (10, 5)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 55)


def save_fig(fig, name):
    """Сохраняет фигуру в artifacts/figures и закрывает её."""
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def save_json(obj, path):
    """Сохраняет словарь как JSON с кириллицей."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def load_raw_data():
    """Читает исходные таблицы и убирает дубликаты пар пользователь-книга."""
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    books = pd.read_csv(DATA_DIR / "books.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    book_tags = pd.read_csv(DATA_DIR / "book_tags.csv")
    ratings = ratings.drop_duplicates(["user_id", "book_id"]).reset_index(drop=True)
    return ratings, books, tags, book_tags


def prepare_book_tags(books, book_tags, tags):
    """Склеивает book_tags с books и tags, чистит имена тегов."""
    bt = (
        book_tags.merge(
            books[["id", "book_id"]],
            left_on="goodreads_book_id",
            right_on="book_id",
            how="inner",
        )
    ).merge(tags, on="tag_id", how="left")
    bt["tag_clean"] = bt["tag_name"].str.lower().str.strip()
    bt = bt[bt["tag_clean"].str.match(r"^[a-z][a-z0-9 &'/-]{1,40}$", na=False)]
    return bt


def build_index_maps(ratings):
    """Сквозные отсортированные идентификаторы пользователей и книг."""
    user_ids = np.sort(ratings.user_id.unique()).astype(int)
    book_ids = np.sort(ratings.book_id.unique()).astype(int)
    return user_ids, book_ids


def make_id2idx(ids):
    """Отображение исходный идентификатор -> позиция в массиве."""
    return {int(v): i for i, v in enumerate(ids)}


def save_index_maps(path, user_ids, book_ids):
    np.savez(path, user_ids=user_ids, book_ids=book_ids)


def load_index_maps(path):
    data = np.load(path)
    return data["user_ids"], data["book_ids"]


def add_indices(df, uid2i, bid2i):
    """Добавляет в датафрейм колоночные индексы user_idx и book_idx."""
    df = df.copy()
    df["user_idx"] = df.user_id.map(uid2i)
    df["book_idx"] = df.book_id.map(bid2i)
    return df


def build_interaction_matrices(df, n_users, n_books, relevance_threshold):
    """Разреженные матрицы взаимодействий: оценки и бинарный фидбек."""
    R = csr_matrix(
        (df.rating.values.astype(np.float32), (df.user_idx.values, df.book_idx.values)),
        shape=(n_users, n_books),
    )
    B = R.copy()
    B.data = (B.data >= relevance_threshold).astype(np.float32)
    return R, B
