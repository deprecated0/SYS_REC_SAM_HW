"""Метрики ранжирования и сборка топ-N рекомендаций."""

import numpy as np


def precision_at_k(recommended, relevant, k):
    rec = np.asarray(recommended)[:k]
    return float(np.isin(rec, list(relevant)).mean())


def recall_at_k(recommended, relevant, k):
    rec = np.asarray(recommended)[:k]
    return float(np.isin(np.asarray(list(relevant)), rec).mean())


def ndcg_at_k(recommended, relevant, k):
    rec = np.asarray(recommended)[:k]
    hits = np.isin(rec, list(relevant)).astype(float)
    dcg = float((hits / np.log2(np.arange(2, len(rec) + 2))).sum())
    ideal_hits = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_hits + 2))).sum())
    return dcg / idcg if ideal_hits > 0 else 0.0


def topn_rows(scores, n):
    """Индексы n максимальных значений в каждой строке, по убыванию."""
    part = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    order = np.argsort(np.take_along_axis(-scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def build_recs(score_fn, seen_csr, n, chunk, subset=None):
    """Матрица рекомендаций (m x n), считается чанками по пользователям."""
    subset = np.arange(seen_csr.shape[0]) if subset is None else subset
    recs = np.empty((len(subset), n), dtype=np.int32)
    for st in range(0, len(subset), chunk):
        idx = subset[st : st + chunk]
        scores = np.asarray(score_fn(idx), dtype=np.float32)
        seen_mask = seen_csr[idx].toarray().astype(bool)
        scores[seen_mask] = -np.inf
        recs[st : st + chunk] = topn_rows(scores, n)
    return recs


def evaluate(name, recs_matrix, rel_sets, users):
    """Считает P@K, R@K, nDCG@K для одной модели по множествам релевантных книг."""
    res = {"model": name}
    for k in (5, 10):
        p = np.mean(
            [
                precision_at_k(recs_matrix[i], rel_sets[u], k)
                for i, u in enumerate(users)
            ]
        )
        r = np.mean(
            [recall_at_k(recs_matrix[i], rel_sets[u], k) for i, u in enumerate(users)]
        )
        nd = np.mean(
            [ndcg_at_k(recs_matrix[i], rel_sets[u], k) for i, u in enumerate(users)]
        )
        res.update({f"P@{k}": p, f"R@{k}": r, f"nDCG@{k}": nd})
    return res
