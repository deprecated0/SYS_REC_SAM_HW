"""Модели: популярность, контентная TF-IDF, Item-Based CF, SVD, гибриды."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from surprise import SVD as SurpriseSVD
from surprise import Dataset, Reader

# Служебные полки: описывают поведение читателя, а не содержание книги
META_TAGS = {
    "to-read",
    "currently-reading",
    "owned",
    "books-i-own",
    "favorites",
    "favourites",
    "favorite",
    "favourite",
    "favorite-books",
    "favourite-books",
    "default",
    "to-buy",
    "owned-books",
    "my-books",
    "my-library",
    "library",
    "kindle",
    "ebooks",
    "ebook",
    "audiobook",
    "audiobooks",
    "audio",
    "want-to-read",
    "did-not-finish",
    "dnf",
    "book-club",
    "re-read",
    "reread",
    "owned-to-read",
    "i-owned",
    "school",
    "class",
    "favourites-read",
    "favorites-read",
    "books",
    "read",
    "read-in-2015",
    "read-in-2016",
    "read-2015",
    "read-2016",
    "shelfari-favorites",
    "all-time-favorites",
}


def popularity_scores(train_df, n_books, min_ratings):
    """Скор книги равен среднему рейтингу при достаточном числе оценок, иначе -1."""
    stats = train_df.groupby("book_idx").agg(
        mean=("rating", "mean"), n=("rating", "size")
    )
    scores = np.full(n_books, -1.0, dtype=np.float32)
    eligible = stats.index[stats.n >= min_ratings]
    scores[eligible] = stats.loc[eligible, "mean"].values
    return scores, stats


def find_demo_book(books, pattern):
    """Возвращает books.id первой книги, в названии которой встречается паттерн."""
    row = books[books.title.str.contains(pattern, case=False, na=False, regex=False)]
    return int(row["id"].iloc[0])


def build_book_profiles(books, bt, book_ids, bid2i, meta_tags):
    """Текстовый профиль книги: слова названия плюс очищенные теги."""
    n_books = len(book_ids)
    books_by_id = books.set_index("id")
    titles = books_by_id.loc[book_ids, "original_title"].fillna(
        books_by_id.loc[book_ids, "title"]
    )

    bt = bt.copy()
    bt["book_idx"] = bt["id"].map(bid2i)
    bt = bt[bt["book_idx"].notna()]
    bt_content = bt[~bt["tag_clean"].isin(meta_tags)]
    profile_tags = (
        bt_content.sort_values(["book_idx", "count"], ascending=[True, False])
        .groupby("book_idx")["tag_clean"]
        .agg(" ".join)
    )

    titles_clean = (
        titles.str.lower().str.replace(r"[^\w\s]", " ", regex=True).fillna("")
    )
    tags_text = profile_tags.reindex(np.arange(n_books)).fillna("")
    return titles_clean.reset_index(drop=True).str.cat(
        tags_text.reset_index(drop=True), sep=" "
    )


def fit_item_profiles(profiles, min_df=3):
    """TF-IDF векторизация профилей с L2-нормировкой: косинус равен скалярному произведению."""
    tfidf = TfidfVectorizer(
        min_df=min_df, sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b"
    )
    item_profiles = tfidf.fit_transform(profiles.values)
    return normalize(item_profiles)


def get_similar_books(item_profiles, idx, n):
    """n ближайших книг по косинусной близости профилей."""
    sims = (item_profiles @ item_profiles[idx].T).toarray().ravel()
    sims[idx] = -1.0
    top = np.argsort(-sims)[:n]
    return top, sims[top]


def build_favorites(df, n_users, n_books, relevance_threshold):
    """Нормированная матрица любимых книг пользователя (оценки >= порога)."""
    fav = df[df.rating >= relevance_threshold]
    no_fav = set(df.user_idx.unique()) - set(fav.user_idx.unique())
    if no_fav:
        fav = pd.concat([fav, df[df.user_idx.isin(no_fav)]])
    w = 1.0 / fav.groupby("user_idx")["book_idx"].transform("size")
    return csr_matrix(
        (w.values, (fav.user_idx.values, fav.book_idx.values)),
        shape=(n_users, n_books),
    )


def make_content_scorer(fav_csr, item_profiles, fallback_scores):
    """Косинус между профилем пользователя и профилем каждой книги."""

    def score_fn(user_indices):
        prof = fav_csr[user_indices] @ item_profiles
        scores = np.asarray((prof @ item_profiles.T).todense(), dtype=np.float32)
        empty = np.asarray(prof.getnnz(axis=1) == 0)
        if empty.any():
            scores[empty] = fallback_scores
        return scores

    return score_fn


def keep_topk_per_row(S, k):
    """Оставляет в каждой строке разреженной матрицы только k максимальных значений."""
    S = S.tocsr()
    rows, cols, vals = [], [], []
    indptr, indices, data = S.indptr, S.indices, S.data
    for i in range(S.shape[0]):
        lo, hi = indptr[i], indptr[i + 1]
        if hi - lo > k:
            sel = np.argpartition(-data[lo:hi], k)[:k]
        else:
            sel = np.arange(hi - lo)
        rows.append(np.full(len(sel), i))
        cols.append(indices[lo:hi][sel])
        vals.append(data[lo:hi][sel])
    return csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=S.shape,
    )


def item_similarity(R, lmbda=10.0, k=20):
    """Косинусные близости книг со сжатием; в каждой строке остаётся топ-k соседей."""
    item_users = normalize(R.T.tocsr())
    S = (item_users @ item_users.T).tocsr()
    co = (R.sign().T @ R.sign()).tocsr()
    shrink = co.copy()
    shrink.data = shrink.data / (shrink.data + lmbda)
    S = S.multiply(shrink).tocsr()
    S.setdiag(0.0)
    S.eliminate_zeros()
    return keep_topk_per_row(S, k)


def predict_rating_itemcf(user_idx, book_idx, S_topk, R_train, item_mean, k):
    """Прогноз оценки: взвешенное среднее оценок пользователя по соседям книги."""
    neighbours = S_topk.getrow(book_idx)
    order = np.argsort(-neighbours.data)[:k]
    nb_idx = neighbours.indices[order]
    nb_sim = neighbours.data[order]
    user_row = R_train.getrow(user_idx)
    rated = dict(zip(user_row.indices, user_row.data))
    common = [(j, s) for j, s in zip(nb_idx, nb_sim) if j in rated]
    if not common:
        return float(item_mean[book_idx]), 0
    num = sum(s * rated[j] for j, s in common)
    den = sum(abs(s) for _, s in common)
    return float(num / den), len(common)


def make_itemcf_scorer(R, S_k, item_mean):
    """Скоринг всех книг для списка пользователей: взвешенная сумма близостей."""
    Rb = R.sign()
    S_kt = S_k.T.tocsr()

    def score_fn(user_indices):
        num = np.asarray((R[user_indices] @ S_kt).todense(), dtype=np.float32)
        den = np.asarray((Rb[user_indices] @ S_kt).todense(), dtype=np.float32)
        out = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
        fallback = den <= 0
        if fallback.any():
            out[fallback] = np.broadcast_to(item_mean, out.shape)[fallback]
        return out.astype(np.float32)

    return score_fn


def fit_svd(train_df, user_ids, book_ids, n_factors, n_epochs, seed):
    """Обучает surprise.SVD и раскладывает параметры в массивы по сквозным индексам."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["user_id", "book_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    algo = SurpriseSVD(n_factors=n_factors, n_epochs=n_epochs, random_state=seed)
    algo.fit(trainset)

    # Отображения "сырой" идентификатор -> внутренний. Строим их через
    # обратное преобразование, чтобы не зависеть от версии библиотеки:
    # в новых версиях атрибуты raw2inner_id_users/raw2inner_id_items удалены.
    raw2inner_u = {trainset.to_raw_uid(iu): iu for iu in trainset.all_users()}
    raw2inner_i = {trainset.to_raw_iid(ii): ii for ii in trainset.all_items()}

    n_users, n_books = len(user_ids), len(book_ids)
    u_inner = np.array([raw2inner_u.get(int(u), -1) for u in user_ids])
    i_inner = np.array([raw2inner_i.get(int(b), -1) for b in book_ids])

    bu = np.zeros(n_users, np.float32)
    pu = np.zeros((n_users, n_factors), np.float32)
    mask_u = u_inner >= 0
    bu[mask_u] = algo.bu[u_inner[mask_u]]
    pu[mask_u] = algo.pu[u_inner[mask_u]]

    bi = np.zeros(n_books, np.float32)
    qi = np.zeros((n_books, n_factors), np.float32)
    mask_i = i_inner >= 0
    bi[mask_i] = algo.bi[i_inner[mask_i]]
    qi[mask_i] = algo.qi[i_inner[mask_i]]

    return {
        "mu": np.float32(trainset.global_mean),
        "bu": bu,
        "bi": bi,
        "pu": pu,
        "qi": qi,
    }


def make_svd_scorer(bundle):
    """Полный прогноз SVD: глобальное среднее + смещения + факторы."""
    mu, bu, bi, pu, qi = (
        bundle["mu"],
        bundle["bu"],
        bundle["bi"],
        bundle["pu"],
        bundle["qi"],
    )

    def score_fn(user_indices):
        return (
            mu + bu[user_indices, None] + bi[None, :] + pu[user_indices] @ qi.T
        ).astype(np.float32)

    return score_fn


def predict_pairs(score_fn, user_idx, book_idx, chunk):
    """Предсказания для списка пар (пользователь, книга), чанками без плотной матрицы."""
    order = np.argsort(user_idx, kind="stable")
    us = user_idx[order]
    out = np.empty(len(user_idx), np.float32)
    uniq = np.unique(us)
    for st in range(0, len(uniq), chunk):
        part = uniq[st : st + chunk]
        scores = score_fn(part)
        lo = np.searchsorted(us, part[0])
        hi = np.searchsorted(us, part[-1], "right")
        sel = order[lo:hi]
        local = np.searchsorted(part, user_idx[sel])
        out[sel] = scores[local, book_idx[sel]]
    return out


def z_norm(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + 1e-9)


def make_hybrid_scorer(score_a, score_b, prior, w_a=0.5, w_prior=0.05):
    """Смесь двух скорингов и априорного сигнала, всё через z-нормировку по кандидатам."""

    def score_fn(user_indices):
        a = z_norm(np.asarray(score_a(user_indices), np.float32))
        b = z_norm(np.asarray(score_b(user_indices), np.float32))
        c = z_norm(np.broadcast_to(prior, a.shape).astype(np.float32))
        return (w_a * a + (1 - w_a - w_prior) * b + w_prior * c).astype(np.float32)

    return score_fn
