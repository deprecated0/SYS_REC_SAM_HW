"""Стадия 7. Гибридные модели и эксперимент с холодным стартом."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz

from .config import (
    FIGURES_DIR,
    MODELS_DIR,
    SPLIT_DIR,
    TABLES_DIR,
    ensure_dirs,
    load_params,
)
from .io_utils import build_interaction_matrices, load_index_maps, save_fig, setup_plots
from .metrics import build_recs, evaluate
from .models import (
    build_favorites,
    fit_svd,
    item_similarity,
    make_content_scorer,
    make_hybrid_scorer,
    make_itemcf_scorer,
    make_svd_scorer,
    popularity_scores,
)


def rel_info(test, n_users, n_books, threshold):
    """Множества релевантных тестовых книг и список пользователей для оценки."""
    rel_mask = (test.rating >= threshold).values
    test_rel = csr_matrix(
        (
            np.ones(rel_mask.sum(), np.float32),
            (test.user_idx.values[rel_mask], test.book_idx.values[rel_mask]),
        ),
        shape=(n_users, n_books),
    )
    rel_sets = {
        u: set(test_rel.getrow(u).indices)
        for u in np.unique(test.user_idx.values[rel_mask])
    }
    return rel_sets, np.array(sorted(rel_sets))


def main():
    params = load_params()
    ensure_dirs()
    setup_plots()

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    user_ids, book_ids = load_index_maps(SPLIT_DIR / "index_maps.npz")
    n_users, n_books = len(user_ids), len(book_ids)
    n_recs, chunk = params["n_recs"], params["chunk"]

    R_train = load_npz(MODELS_DIR / "R_train.npz")
    S_topk = load_npz(MODELS_DIR / "S_topk.npz")
    item_mean = np.load(MODELS_DIR / "item_mean.npy")
    pop_scores = np.load(MODELS_DIR / "pop_scores.npy")
    item_profiles = load_npz(MODELS_DIR / "item_profiles.npz")
    fav_train = load_npz(MODELS_DIR / "fav_train.npz")
    svd_bundle = np.load(MODELS_DIR / "svd_bundle.npz")

    content_score = make_content_scorer(fav_train, item_profiles, pop_scores)
    itemcf_score = make_itemcf_scorer(R_train, S_topk, item_mean)
    svd_score = make_svd_scorer(svd_bundle)

    rel_sets, eval_users = rel_info(
        test, n_users, n_books, params["relevance_threshold"]
    )
    pop_prior = np.log1p(
        train.groupby("book_idx")
        .size()
        .reindex(np.arange(n_books), fill_value=0)
        .values
    )
    R_seen = R_train.sign().astype(np.bool_)

    w_main = params["hybrid"]["w_main"]
    w_prior = params["hybrid"]["w_prior"]

    recs = {
        "Hybrid (SVD + ItemCF)": build_recs(
            make_hybrid_scorer(svd_score, itemcf_score, pop_prior, w_main, w_prior),
            R_seen,
            n_recs,
            chunk,
        ),
        "Hybrid (Content + ItemCF + pop)": build_recs(
            make_hybrid_scorer(content_score, itemcf_score, pop_prior, w_main, w_prior),
            R_seen,
            n_recs,
            chunk,
        ),
    }

    results = pd.read_csv(TABLES_DIR / "metrics_main.csv", index_col="model")
    new_rows = pd.DataFrame(
        [evaluate(name, recs[name][eval_users], rel_sets, eval_users) for name in recs]
    ).set_index("model")
    results = pd.concat([results, new_rows])
    results.to_csv(TABLES_DIR / "metrics_full.csv", index_label="model")
    print(results)

    ax = (
        results[["P@10", "R@10", "nDCG@10"]]
        .sort_values("nDCG@10")
        .plot(kind="barh", figsize=(10, 5), colormap="crest")
    )
    ax.set_title("Итоговое сравнение моделей (метрики top-10)")
    ax.set_xlabel("Значение метрики")
    ax.set_ylabel("Модель")
    ax.figure.tight_layout()
    save_fig(ax.figure, "metrics_final.png")
    print(f"Лучшая модель по nDCG@10: {results['nDCG@10'].idxmax()}")

    # Эксперимент с холодным стартом: у части пользователей оставляем
    # только несколько оценок и переобучаем модели на урезанных данных.
    cold = params["cold"]
    rng_cold = np.random.default_rng(params["random_state"])
    cold_users = np.sort(rng_cold.choice(n_users, size=cold["n_users"], replace=False))

    tr_sh = train.sample(frac=1.0, random_state=params["random_state"])
    tr_sh["rk"] = tr_sh.groupby("user_idx").cumcount()
    keep_mask = ~(tr_sh.user_idx.isin(cold_users) & (tr_sh.rk >= cold["keep_ratings"]))
    train_cold = tr_sh[keep_mask].drop(columns="rk")
    print(
        f"Холодных пользователей: {len(cold_users):,}; "
        f"размер train после обрезки: {len(train_cold):,}"
    )

    R_cold, _ = build_interaction_matrices(
        train_cold, n_users, n_books, params["relevance_threshold"]
    )
    R_seen_cold = R_cold.sign().astype(np.bool_)
    S_cold = item_similarity(
        R_cold, lmbda=params["shrinkage_lambda"], k=params["top_k_neighbours"]
    )
    itemcf_cold_score = make_itemcf_scorer(R_cold, S_cold, item_mean)

    pop_cold_scores, _ = popularity_scores(
        train_cold, n_books, params["min_ratings_popular"]
    )

    def pop_cold_score(idx):
        return np.broadcast_to(pop_cold_scores, (len(idx), n_books)).copy()

    fav_cold = build_favorites(
        train_cold, n_users, n_books, params["relevance_threshold"]
    )
    content_cold_score = make_content_scorer(fav_cold, item_profiles, pop_cold_scores)

    svd_cold_bundle = fit_svd(
        train_cold,
        user_ids,
        book_ids,
        n_factors=params["svd"]["n_factors"],
        n_epochs=cold["svd_epochs"],
        seed=params["random_state"],
    )
    svd_cold_score = make_svd_scorer(svd_cold_bundle)

    cold_recs = {
        "Popularity": build_recs(
            pop_cold_score, R_seen_cold, n_recs, chunk, subset=cold_users
        ),
        "Content (1-3 книги)": build_recs(
            content_cold_score, R_seen_cold, n_recs, chunk, subset=cold_users
        ),
        "SVD (переобучена)": build_recs(
            svd_cold_score, R_seen_cold, n_recs, chunk, subset=cold_users
        ),
        "ItemKNN CF": build_recs(
            itemcf_cold_score, R_seen_cold, n_recs, chunk, subset=cold_users
        ),
        "Hybrid (content + pop)": build_recs(
            make_hybrid_scorer(
                content_cold_score, pop_cold_score, pop_prior, cold["w_content"], 0.0
            ),
            R_seen_cold,
            n_recs,
            chunk,
            subset=cold_users,
        ),
    }

    cold_eval_mask = np.isin(cold_users, eval_users)
    cold_eval = cold_users[cold_eval_mask]
    print(
        f"Оцениваем на {len(cold_eval):,} холодных пользователях с релевантным тестом"
    )

    cold_results = pd.DataFrame(
        [
            evaluate(name, m[cold_eval_mask], rel_sets, cold_eval)
            for name, m in cold_recs.items()
        ]
    ).set_index("model")
    cold_results.to_csv(TABLES_DIR / "metrics_cold.csv", index_label="model")
    print(cold_results)


if __name__ == "__main__":
    main()
