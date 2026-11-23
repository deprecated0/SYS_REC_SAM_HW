"""Стадия 5. Матричное разложение: SVD (библиотека surprise)."""

import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from .config import MODELS_DIR, SPLIT_DIR, TABLES_DIR, ensure_dirs, load_params
from .io_utils import load_index_maps
from .models import fit_svd, make_itemcf_scorer, make_svd_scorer, predict_pairs


def main():
    params = load_params()
    ensure_dirs()

    train = pd.read_csv(SPLIT_DIR / "train.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    user_ids, book_ids = load_index_maps(SPLIT_DIR / "index_maps.npz")

    t0 = time.perf_counter()
    bundle = fit_svd(
        train,
        user_ids,
        book_ids,
        n_factors=params["svd"]["n_factors"],
        n_epochs=params["svd"]["n_epochs"],
        seed=params["random_state"],
    )
    print(
        f"SVD обучена за {time.perf_counter() - t0:.0f} с "
        f"({params['svd']['n_factors']} факторов, {params['svd']['n_epochs']} эпох, "
        f"{len(train):,} наблюдений)"
    )
    np.savez(
        MODELS_DIR / "svd_bundle.npz",
        mu=bundle["mu"],
        bu=bundle["bu"],
        bi=bundle["bi"],
        pu=bundle["pu"],
        qi=bundle["qi"],
    )

    fact = test.rating.values
    svd_pred = predict_pairs(
        make_svd_scorer(bundle),
        test.user_idx.values,
        test.book_idx.values,
        params["chunk"],
    )
    svd_rmse = float(np.sqrt(np.mean((svd_pred - fact) ** 2)))
    print(f"RMSE SVD на тесте: {svd_rmse:.4f}")

    # Таблица RMSE/MAE: базовые статистики, ItemKNN и SVD
    R_train = load_npz(MODELS_DIR / "R_train.npz")
    S_topk = load_npz(MODELS_DIR / "S_topk.npz")
    item_mean = np.load(MODELS_DIR / "item_mean.npy")
    itemcf_score = make_itemcf_scorer(R_train, S_topk, item_mean)
    itemcf_pred = predict_pairs(
        itemcf_score, test.user_idx.values, test.book_idx.values, params["chunk"]
    )

    glob_mean = float(train.rating.mean())
    user_mean = np.full(len(user_ids), glob_mean, np.float32)
    um = train.groupby("user_idx")["rating"].mean()
    user_mean[um.index] = um.values.astype(np.float32)

    preds = {
        "Global mean": np.full(len(test), glob_mean, np.float32),
        "User mean": user_mean[test.user_idx.values],
        "Item mean": item_mean[test.book_idx.values],
        f"ItemKNN CF (cos, K={params['top_k_neighbours']})": itemcf_pred,
        f"SVD (surprise, d={params['svd']['n_factors']})": svd_pred,
    }
    table = pd.DataFrame(
        {
            "RMSE": {
                n: float(np.sqrt(np.mean((p - fact) ** 2))) for n, p in preds.items()
            },
            "MAE": {n: float(np.mean(np.abs(p - fact))) for n, p in preds.items()},
        }
    ).sort_values("RMSE")
    table.to_csv(TABLES_DIR / "rmse_table.csv", index_label="model")
    print(table)


if __name__ == "__main__":
    main()
