import numpy as np
from edmkit.simplex_projection import knn
from pandas import Series


# ------------------------------------------------------------
# pyEDM argument conventions
# ------------------------------------------------------------
def _names(spec):
    """columns/target specification -> list of names (pyEDM Validate())"""
    return spec.split() if isinstance(spec, str) else list(spec)


def _dimension(kwargs):
    """Embedding dimension: E, or number of columns if embedded"""
    return len(_names(kwargs["columns"])) if kwargs["embedded"] else kwargs["E"]


# ------------------------------------------------------------
def transform_args(kwargs):
    """pyEDM Simplex kwargs -> edmkit simplex_projection kwargs.

    pyEDM defaults knn to E + 1; edmkit defaults k to E + 1 from the
    embedding width, which differs for multivariate embeddings, so k is
    always passed explicitly.
    """
    knn_ = kwargs["knn"]
    return dict(k=knn_ if knn_ > 0 else _dimension(kwargs) + 1)


# ------------------------------------------------------------
def _embedding(data, kwargs):
    """Takens embedding over all data rows (pyEDM API.Embed()).
    Rows lacking history for a full vector contain nan."""
    columns = _names(kwargs["columns"])

    if kwargs["embedded"]:
        return data[columns].to_numpy(dtype=float)

    E, tau = kwargs["E"], kwargs["tau"]
    N = data.shape[0]
    row = np.arange(N)

    emb = np.full((N, len(columns) * E), np.nan)
    for c, name in enumerate(columns):
        x = data[name].to_numpy(dtype=float)
        for lag in range(E):
            src = row + tau * lag
            ok = (src >= 0) & (src < N)
            emb[ok, c * E + lag] = x[src[ok]]
    return emb


# ------------------------------------------------------------
def _indices(data, kwargs):
    """0-offset lib_i, pred_i from 1-offset lib/pred span pairs,
    following pyEDM EDM.CreateIndices() + RemoveNan() + validLib:
      - lib spans shrunk by the embedding shift |tau| * (E - 1) and Tp
      - embedding rows containing nan removed from lib_i and pred_i
      - lib_i restricted to validLib rows
    """
    E, tau, Tp = _dimension(kwargs), kwargs["tau"], kwargs["Tp"]
    embedded = kwargs["embedded"]
    embedShift = abs(tau) * (E - 1)

    lib, pred = kwargs["lib"], kwargs["pred"]
    libPairs = [(lib[i], lib[i + 1]) for i in range(0, len(lib), 2)]

    lib_i = []
    for r, (start, stop) in enumerate(libPairs):
        if not embedded:
            if tau < 0:
                start = start + embedShift
            else:
                stop = stop - embedShift
        if Tp < 0:
            if not embedded:
                start = max(start, start + abs(Tp) - 1)
        elif r == len(libPairs) - 1:
            stop = stop - Tp
        lib_i.extend(range(start - 1, stop))
    lib_i = np.asarray(lib_i, dtype=int)

    pred_i = []
    for i in range(0, len(pred), 2):
        pred_i.extend(range(pred[i] - 1, pred[i + 1]))
    pred_i = np.asarray(pred_i, dtype=int)

    # lib : pred overlap requires self-match neighbor exclusion
    libOverlap = len(np.intersect1d(lib_i, pred_i)) > 0

    # RemoveNan (ignoreNan): drop embedding rows containing nan
    emb = _embedding(data, kwargs)
    nanRow = np.isnan(emb).any(axis=1)
    lib_i = lib_i[~nanRow[lib_i]]
    pred_i = pred_i[~nanRow[pred_i]]

    if len(kwargs["validLib"]):
        validLib = np.asarray(kwargs["validLib"], dtype=bool)
        lib_i = lib_i[validLib[lib_i]]

    return emb, lib_i, pred_i, libOverlap


# ------------------------------------------------------------
def transform_data(data, kwargs):
    """DataFrame + pyEDM Simplex kwargs -> simplex_projection data args.

    X : library embedding vectors        emb[lib_i]
    Y : library targets Tp rows ahead    target[lib_i + Tp]
    Q : query embedding vectors          emb[pred_i]

    pyEDM excludes library neighbors per query (the self-match when lib
    and pred overlap, rows within exclusionRadius).  edmkit's mask is
    per library row, not per query, so exclusions are expressed through
    the batch dimension: one batch per query with its own library mask.
    """
    emb, lib_i, pred_i, libOverlap = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)

    X = emb[lib_i]
    Y = target[lib_i + kwargs["Tp"]]
    Q = emb[pred_i]

    exclusionRadius = kwargs["exclusionRadius"]
    if not (libOverlap or exclusionRadius > 0):
        return dict(X=X, Y=Y, Q=Q, mask=None)

    if exclusionRadius > 0:
        mask = abs(pred_i[:, None] - lib_i[None, :]) > exclusionRadius
    else:
        mask = pred_i[:, None] != lib_i[None, :]

    B = len(pred_i)
    return dict(
        X=np.tile(X, (B, 1, 1)),
        Y=np.tile(Y[:, None], (B, 1, 1)),
        Q=Q[:, None, :],
        mask=mask,
    )


# ------------------------------------------------------------
def transform_result(predictions):
    """simplex_projection return value -> Series of predictions in query
    order, nan (missing library targets) dropped.  Batched (B, 1, 1)
    and unbatched (M,) results flatten identically."""
    smplx = Series(np.asarray(predictions).reshape(-1), name="Predictions")
    return smplx.dropna().reset_index(drop=True)


# ------------------------------------------------------------
def transform_valid(dfv):
    """pyEDM validation DataFrame -> Series of Predictions without the
    nan padding rows, aligned with transform_result()."""
    return dfv.get("Predictions").dropna().reset_index(drop=True)


# ------------------------------------------------------------
def knn_neighbors(data, kwargs):
    """(N_pred, knn) neighbor data-row indices via edmkit knn, matching
    pyEDM Simplex(returnObject = True).knn_neighbors."""
    emb, lib_i, pred_i, libOverlap = _indices(data, kwargs)
    k = transform_args(kwargs)["k"]

    X, Q = emb[lib_i], emb[pred_i]

    exclusionRadius = kwargs["exclusionRadius"]
    if not (libOverlap or exclusionRadius > 0):
        _, idx = knn(X, Q, k)
        return lib_i[idx]

    neighbors = np.empty((len(pred_i), k), dtype=int)
    for b, row in enumerate(pred_i):
        if exclusionRadius > 0:
            keep = abs(lib_i - row) > exclusionRadius
        else:
            keep = lib_i != row
        _, idx = knn(X[keep], Q[b : b + 1], k)
        neighbors[b] = lib_i[keep][idx[0]]
    return neighbors
