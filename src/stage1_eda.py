"""Стадия 1. Разведочный анализ данных (EDA)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, TABLES_DIR, ensure_dirs, load_params
from .io_utils import load_raw_data, prepare_book_tags, save_fig, save_json, setup_plots


def plot_rating_distribution(ratings):
    rating_counts = ratings["rating"].value_counts().sort_index()
    share = rating_counts / rating_counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 4.3))
    bars = ax.bar(
        rating_counts.index.astype(int),
        rating_counts.values,
        color=sns.color_palette("crest", len(rating_counts)),
    )
    ax.bar_label(bars, labels=[f"{p:.1f}%" for p in share.values], padding=2)
    mean_r = ratings["rating"].mean()
    ax.axvline(
        mean_r - 0.5,
        color="darkred",
        ls="--",
        lw=1.3,
        label=f"средняя оценка = {mean_r:.2f}",
    )
    ax.set_title("Распределение оценок по шкале 1–5 (Goodbooks-10k)")
    ax.set_xlabel("Оценка")
    ax.set_ylabel("Число оценок")
    ax.set_xticks(rating_counts.index.astype(int))
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "rating_distribution.png")

    return {
        "mean_rating": round(float(mean_r), 3),
        "median_rating": float(ratings.rating.median()),
        "share_ge_4_percent": round(float((ratings.rating >= 4).mean() * 100), 1),
        "share_le_2_percent": round(float((ratings.rating <= 2).mean() * 100), 1),
    }


def plot_user_activity(ratings):
    user_activity = ratings.groupby("user_id").size()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax1.hist(user_activity.values, bins=48, color="#4c72b0", edgecolor="white")
    ax1.set_yscale("log")
    ax1.set_title("Гистограмма активности пользователей")
    ax1.set_xlabel("Число оценок у пользователя")
    ax1.set_ylabel("Число пользователей (лог-шкала)")

    sorted_act = np.sort(user_activity.values)[::-1]
    ax2.plot(sorted_act, np.arange(1, len(sorted_act) + 1), color="#4c72b0", lw=2)
    for q, c, lbl in [
        (50, "green", "медиана"),
        (95, "darkorange", "p95"),
        (99, "darkred", "p99"),
    ]:
        v = int(np.percentile(user_activity, q))
        ax2.axvline(v, color=c, ls="--", lw=1.2)
        ax2.annotate(
            f"{lbl} = {v}",
            xy=(v, len(sorted_act) * 0.62),
            color=c,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=9,
        )
    ax2.set_title("Взаимоотношение количества пользователей\nи количества оценок")
    ax2.set_xlabel("Минимальное число оценок у пользователя")
    ax2.set_ylabel("Число таких пользователей")
    fig.tight_layout()
    save_fig(fig, "user_activity.png")

    q = user_activity.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).astype(int)
    return {
        "ratings_per_user_quantiles": {str(k): int(v) for k, v in q.items()},
        "top1_percent_threshold": int(np.percentile(user_activity, 99)),
        "top1_percent_users": int(
            (user_activity >= np.percentile(user_activity, 99)).sum()
        ),
        "min_ratings_per_user": int(user_activity.min()),
        "users_lt6_ratings_percent": round(float((user_activity < 6).mean() * 100), 1),
    }


def plot_book_popularity(ratings, books):
    book_pop = ratings.groupby("book_id").agg(
        n_ratings=("rating", "size"), mean_rating=("rating", "mean")
    )
    total_sorted = (
        books["work_ratings_count"]
        .dropna()
        .sort_values(ascending=False)
        .values.astype(float)
    )
    total_cum = np.cumsum(total_sorted) / total_sorted.sum() * 100
    share_axis = np.arange(1, len(total_sorted) + 1) / len(total_sorted) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ax1.hist(book_pop["n_ratings"], bins=40, color="#4c72b0", edgecolor="white")
    ax1.axvline(
        100, color="darkred", ls="--", lw=1.3, label="потолок выгрузки = 100 оценок"
    )
    ax1.set_title("Оценок на книгу в датасете (выборка ограничена сверху)")
    ax1.set_xlabel("Количество оценок у книги (ratings.csv)")
    ax1.set_ylabel("Число книг")
    ax1.legend()

    ax2.plot(share_axis, total_cum, color="#4c72b0", lw=2, label="кривая Лоренца")
    ax2.plot(
        [0, 100],
        [0, 100],
        color="grey",
        ls=":",
        lw=1.2,
        label="равномерное распределение",
    )
    ax2.set_title(
        "Истинная популярность: кривая Лоренца\n(общее число оценок из books.csv)"
    )
    ax2.set_xlabel("Доля книг (отсортированных по популярности), %")
    ax2.set_ylabel("Накопленная доля всех оценок, %")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, "book_popularity.png")

    lorenz = {
        f"top_{p}_percent_share": round(
            float(total_cum[int(len(total_sorted) * p / 100) - 1]), 1
        )
        for p in (1, 5, 10)
    }
    stats = {
        "median_ratings_per_book": int(book_pop.n_ratings.median()),
        "max_ratings_per_book": int(book_pop.n_ratings.max()),
        "min_ratings_per_book": int(book_pop.n_ratings.min()),
        "books_with_full_budget": int((book_pop.n_ratings == 100).sum()),
        "total_median_ratings": int(np.median(total_sorted)),
        "total_max_ratings": int(total_sorted.max()),
        **lorenz,
    }
    return book_pop, stats


def save_top10_popular(books, book_pop):
    top10 = books.nlargest(10, "work_ratings_count").reset_index()[
        ["id", "title", "authors", "work_ratings_count", "average_rating"]
    ]
    top10["n_ratings_dataset"] = book_pop.loc[top10["id"], "n_ratings"].values
    top10.to_csv(TABLES_DIR / "top10_popular_books.csv", index=False)
    return top10


def plot_rating_vs_count(book_pop, min_ratings_popular):
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.scatter(
        book_pop["n_ratings"], book_pop["mean_rating"], s=10, alpha=0.3, color="#4c72b0"
    )
    ax.axvline(
        min_ratings_popular,
        color="darkred",
        ls="--",
        lw=1.3,
        label=f"порог популярной модели = {min_ratings_popular} оценок",
    )
    ax.set_title(
        "Средний рейтинг против числа оценок: зачем нужен порог в популярной модели"
    )
    ax.set_xlabel("Число оценок книги в датасете")
    ax.set_ylabel("Средний рейтинг книги")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "rating_vs_count.png")

    few = book_pop[book_pop.n_ratings < 90].nlargest(5, "mean_rating")
    return few.reset_index()[["book_id", "n_ratings", "mean_rating"]]


def plot_top_tags(bt):
    top_tags = bt.groupby("tag_clean")["count"].sum().nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(
        top_tags.index, top_tags.values / 1000, color=sns.color_palette("crest", 20)
    )
    ax.set_title("Топ-20 самых частых тегов (суммарное число простановок, тыс.)")
    ax.set_xlabel("Суммарное число простановок, тыс.")
    ax.set_ylabel("Тег")
    fig.tight_layout()
    save_fig(fig, "tags_top20.png")

    return {
        "unique_tags": int(bt.tag_clean.nunique()),
        "avg_tags_per_book": round(float(bt.groupby("id").size().mean()), 0),
    }


def main():
    params = load_params()
    ensure_dirs()
    setup_plots()

    ratings, books, tags, book_tags = load_raw_data()
    summary = {
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings.user_id.nunique()),
        "n_books": int(ratings.book_id.nunique()),
        "n_tags": int(len(tags)),
        "n_book_tags": int(len(book_tags)),
    }
    print(
        f"Оценок: {summary['n_ratings']:,} | пользователей: {summary['n_users']:,} "
        f"| книг: {summary['n_books']:,}"
    )

    summary["ratings"] = plot_rating_distribution(ratings)
    summary["users"] = plot_user_activity(ratings)

    book_pop, pop_stats = plot_book_popularity(ratings, books)
    summary["books"] = pop_stats
    save_top10_popular(books, book_pop)

    trap = plot_rating_vs_count(book_pop, params["min_ratings_popular"])
    trap = trap.merge(
        books.set_index("id")[["title"]], left_on="book_id", right_index=True
    )
    trap[["title", "book_id", "n_ratings", "mean_rating"]].to_csv(
        TABLES_DIR / "trap_low_counts.csv", index=False
    )

    bt = prepare_book_tags(books, book_tags, tags)
    summary["tags"] = plot_top_tags(bt)

    n_users = ratings.user_id.nunique()
    n_books = ratings.book_id.nunique()
    n_cells = n_users * n_books
    summary["matrix"] = {
        "cells": int(n_cells),
        "filled": int(len(ratings)),
        "density_percent": round(len(ratings) / n_cells * 100, 3),
        "dense_size_gb": round(n_cells * 4 / 1024**3, 1),
    }
    print(
        f"Матрица {n_users:,} x {n_books:,}: плотность "
        f"{summary['matrix']['density_percent']}%, dense-вес "
        f"{summary['matrix']['dense_size_gb']} ГБ"
    )

    save_json(summary, TABLES_DIR / "eda_summary.json")
    print("EDA завершён, артефакты сохранены в artifacts/")


if __name__ == "__main__":
    main()
