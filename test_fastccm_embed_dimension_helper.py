"""Parked FastCCM EmbedDimension compatibility helpers.

We are not pursuing pyEDM EmbedDimension reproduction in the FastCCM mirror.
This module remains only as historical context for the parked parity tests.

The cleaner FastCCM-native E/tau search is
`fastccm.ccm_utils.Functions.find_optimal_embedding_params`; it uses one
unified minimum common prediction length after embedding.  pyEDM fixtures use
independent prediction lengths for each embedding dimension, so matching those
numbers requires the handcrafted per-E indexing below.
"""

from fastccm import PairwiseCCM
from numpy import arange, asarray, corrcoef, full, isfinite, isnan, nan, zeros
from numpy import round as npround
from pandas import DataFrame
from scipy.spatial import KDTree

from conftest import SimplexArgs


# ------------------------------------------------------------
def _names(spec):
    """columns/target specification -> list of names."""
    return spec.split() if isinstance(spec, str) else list(spec)


# ------------------------------------------------------------
def _dimension(kwargs):
    """Embedding dimension: E, or number of columns if embedded."""
    return len(_names(kwargs["columns"])) if kwargs["embedded"] else kwargs["E"]


# ------------------------------------------------------------
def _embedding(data, kwargs):
    """Takens embedding over all data rows."""
    columns = _names(kwargs["columns"])

    if kwargs["embedded"]:
        return data[columns].to_numpy(dtype=float)

    E, tau = kwargs["E"], kwargs["tau"]
    N = data.shape[0]
    row = arange(N)

    emb = full((N, len(columns) * E), nan)
    for c, name in enumerate(columns):
        x = data[name].to_numpy(dtype=float)
        for lag in range(E):
            src = row + tau * lag
            ok = (src >= 0) & (src < N)
            emb[ok, c * E + lag] = x[src[ok]]
    return emb


# ------------------------------------------------------------
def _indices(data, kwargs):
    """0-offset lib_i, pred_i from pyEDM-style 1-offset spans."""
    E, tau, Tp = _dimension(kwargs), kwargs["tau"], kwargs["Tp"]
    embedded = kwargs["embedded"]
    embed_shift = abs(tau) * (E - 1)

    lib, pred = kwargs["lib"], kwargs["pred"]
    lib_pairs = [(lib[i], lib[i + 1]) for i in range(0, len(lib), 2)]

    lib_i = []
    for r, (start, stop) in enumerate(lib_pairs):
        if not embedded:
            if tau < 0:
                start += embed_shift
            else:
                stop -= embed_shift
        if Tp < 0:
            if not embedded:
                start = max(start, start + abs(Tp) - 1)
        elif r == len(lib_pairs) - 1:
            stop -= Tp
        lib_i.extend(range(start - 1, stop))
    lib_i = asarray(lib_i, dtype=int)

    pred_i = []
    for i in range(0, len(pred), 2):
        pred_i.extend(range(pred[i] - 1, pred[i + 1]))
    pred_i = asarray(pred_i, dtype=int)

    lib_overlap = len(set(lib_i).intersection(set(pred_i))) > 0

    emb = _embedding(data, kwargs)
    nan_row = isnan(emb).any(axis=1)
    lib_i = lib_i[~nan_row[lib_i]]
    pred_i = pred_i[~nan_row[pred_i]]

    if len(kwargs["validLib"]):
        valid_lib = asarray(kwargs["validLib"], dtype=bool)
        lib_i = lib_i[valid_lib[lib_i]]

    return emb, lib_i, pred_i, lib_overlap


# ------------------------------------------------------------
def compute_error_rho(obs, pred, digits=6):
    """Correlation rho for EmbedDimension's Simplex outputs."""
    obs = asarray(obs, dtype=float)
    pred = asarray(pred, dtype=float)

    keep = isfinite(pred)
    pred = pred[keep]
    obs = obs[keep]

    keep = isfinite(obs)
    pred = pred[keep]
    obs = obs[keep]

    if len(pred) < 5:
        return nan
    return npround(corrcoef(obs, pred)[0, 1], digits).item()


# ------------------------------------------------------------
def simplex_observations(data, kwargs):
    """pyEDM Simplex observation vector aligned with FastCCM query order."""
    _, _, pred_i, _ = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)
    target_i = pred_i + kwargs["Tp"]

    obs = full(len(pred_i), nan, dtype=float)
    ok = (target_i >= 0) & (target_i < len(target))
    obs[ok] = target[target_i[ok]]
    return obs


# ------------------------------------------------------------
def _neighbors(emb, lib_i, pred_i, lib_overlap, exclusion_radius, knn):
    """Exact neighbor rows for the pyEDM-style per-query fallback."""
    x_rad_knn_factor = 5

    exclusion_radius_knn = False
    if exclusion_radius > 0:
        if lib_overlap:
            exclusion_radius_knn = True
        else:
            exclude_row = 0
            if pred_i[0] > lib_i[-1]:
                exclude_row = pred_i[0] - lib_i[-1]
            elif lib_i[0] > pred_i[-1]:
                exclude_row = lib_i[0] - pred_i[-1]
            if exclusion_radius >= exclude_row:
                exclusion_radius_knn = True

    k_query = knn
    if exclusion_radius_knn:
        k_query = min(knn * x_rad_knn_factor, len(lib_i))
    elif lib_overlap:
        k_query += 1

    kd_tree = KDTree(emb[lib_i], leafsize=20, compact_nodes=True, balanced_tree=True)
    _, nn = kd_tree.query(emb[pred_i], k=k_query, eps=0, p=2, workers=-1)
    if k_query == 1:
        nn = nn[:, None]
    nn = lib_i[nn]

    if not (lib_overlap or exclusion_radius_knn or k_query > knn):
        return nn

    pred_col = pred_i[:, None]
    if exclusion_radius_knn:
        mask = abs(pred_col - nn) <= exclusion_radius
    elif lib_overlap:
        mask = pred_col == nn
    else:
        mask = zeros(nn.shape, dtype=bool)

    valid = ~mask
    cs = valid.cumsum(axis=1)
    first_k = valid & (cs <= knn)

    deficient = cs[:, -1] < knn
    if deficient.any():
        first_k[deficient] = False
        first_k[deficient, :knn] = True

    order = (~first_k).argsort(axis=1, kind="stable")
    col = order[:, : min(knn, nn.shape[1])]
    return nn[arange(len(pred_i))[:, None], col]


# ------------------------------------------------------------
def _fastccm_simplex_predictions(data, kwargs):
    """FastCCM Simplex predictions, exact-neighbor fallback when needed."""
    emb, lib_i, pred_i, lib_overlap = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)
    knn = kwargs["knn"] if kwargs["knn"] > 0 else _dimension(kwargs) + 1

    if not lib_overlap and kwargs["exclusionRadius"] == 0:
        ccm = PairwiseCCM(device="cpu", dtype="float64", compute_dtype="float64")
        pred = ccm.predict_matrix(
            X_lib_emb=[emb[lib_i]],
            Y_lib_emb=[target[lib_i + kwargs["Tp"], None]],
            X_pred_emb=[emb[pred_i]],
            library_size=len(lib_i),
            method="simplex",
            nbrs_num=knn,
            tp=0,
            exclusion_window=None,
            seed=1,
            batch_size=None,
            target_batch_size=None,
            clean_after=False,
        )
        return asarray(pred)[:, 0, 0, 0]

    # Future parity target: exact per-query neighbors match the baseline, but
    # active elegant tests avoid cases that reach this branch.
    neighbors = _neighbors(emb, lib_i, pred_i, lib_overlap, kwargs["exclusionRadius"], knn)

    ccm = PairwiseCCM(device="cpu", dtype="float64", compute_dtype="float64")
    predictions = []
    for pred_row, neighbor_rows in zip(pred_i, neighbors):
        pred = ccm.predict_matrix(
            X_lib_emb=[emb[neighbor_rows].copy()],
            Y_lib_emb=[target[neighbor_rows + kwargs["Tp"], None].copy()],
            X_pred_emb=[emb[pred_row : pred_row + 1].copy()],
            library_size=len(neighbor_rows),
            method="simplex",
            nbrs_num=knn,
            tp=0,
            exclusion_window=None,
            seed=1,
            batch_size=None,
            target_batch_size=None,
            clean_after=False,
        )
        predictions.append(pred[0, 0, 0, 0])
    return asarray(predictions, dtype=float)


# ------------------------------------------------------------
def embed_dimension(data, kwargs):
    """Parked pyEDM EmbedDimension compatibility mirror."""
    # Do not extend this parity harness as active FastCCM coverage. Prefer
    # ccm_utils.find_optimal_embedding_params for FastCCM-native E/tau search.
    rho = []
    for E in range(1, kwargs["maxE"] + 1):
        simplex_kwargs = SimplexArgs.copy()
        simplex_kwargs.update(
            dict(
                columns=kwargs["columns"],
                target=kwargs["target"],
                lib=kwargs["lib"],
                pred=kwargs["pred"],
                E=E,
                Tp=kwargs["Tp"],
                tau=kwargs["tau"],
                exclusionRadius=kwargs["exclusionRadius"],
                embedded=kwargs["embedded"],
                validLib=kwargs["validLib"],
                noTime=kwargs["noTime"],
                ignoreNan=kwargs["ignoreNan"],
            )
        )

        rho.append(
            compute_error_rho(
                simplex_observations(data, simplex_kwargs),
                _fastccm_simplex_predictions(data, simplex_kwargs),
            )
        )

    return DataFrame({"E": range(1, kwargs["maxE"] + 1), "rho": rho})
