from edmkit.simplex_projection import simplex_projection
from numpy import (
    arange,
    asarray,
    broadcast_to,
    corrcoef,
    empty,
    full,
    isfinite,
    nan,
    zeros,
)
from pandas import DataFrame
from scipy.spatial import KDTree

from test_edmkit_simplex_projection_helper import (
    _indices,
    _names,
    transform_args,
)

# ------------------------------------------------------------
# pyEDM EmbedDimension is a Simplex sweep over E = 1..maxE, scoring
# each E by ComputeError rho over the prediction set.  The edmkit
# mirror reuses the validated simplex_projection helpers per E; only
# the rho scoring (AuxFunc.ComputeError) is added here.
#
# The sample data carry 4 decimals, so at small E many library points
# are exactly equidistant from a query and the neighbor choice at the
# knn boundary is a tie-break.  pyEDM resolves ties by scipy KDTree
# traversal order, which a distance sort cannot reproduce, so the
# neighbor sets are taken from KDTree exactly as pyEDM FindNeighbors
# does and passed to edmkit as per-query library masks.  edmkit still
# computes the distances, weights and projection.
# ------------------------------------------------------------


# ------------------------------------------------------------
def _simplex_kwargs(kwargs, E):
    """pyEDM EmbedDimension kwargs -> pyEDM Simplex kwargs for one E
    (pyEDM PoolFunc.EmbedDimSimplexFunc)."""
    return dict(
        columns=kwargs["columns"],
        target=kwargs["target"],
        lib=kwargs["lib"],
        pred=kwargs["pred"],
        E=E,
        Tp=kwargs["Tp"],
        knn=0,
        tau=kwargs["tau"],
        exclusionRadius=kwargs["exclusionRadius"],
        embedded=kwargs["embedded"],
        validLib=kwargs["validLib"],
    )


# ------------------------------------------------------------
def _neighbors(emb, lib_i, pred_i, libOverlap, exclusionRadius, knn):
    """(N_pred, knn) neighbor data-row indices replicating pyEDM
    Neighbors.FindNeighbors: KDTree over-query, exclusion mask, first
    knn valid neighbors per row (ties resolved by KDTree order)."""
    xRadKnnFactor = 5

    exclusionRadius_knn = False
    if exclusionRadius > 0:
        if libOverlap:
            exclusionRadius_knn = True
        else:
            excludeRow = 0
            if pred_i[0] > lib_i[-1]:
                excludeRow = pred_i[0] - lib_i[-1]
            elif lib_i[0] > pred_i[-1]:
                excludeRow = lib_i[0] - pred_i[-1]
            if exclusionRadius >= excludeRow:
                exclusionRadius_knn = True

    k_query = knn
    if exclusionRadius_knn:
        k_query = min(knn * xRadKnnFactor, len(lib_i))
    elif libOverlap:
        k_query = k_query + 1

    kdTree = KDTree(emb[lib_i], leafsize=20, compact_nodes=True, balanced_tree=True)
    _, nn = kdTree.query(emb[pred_i], k=k_query, eps=0, p=2, workers=-1)
    if k_query == 1:
        nn = nn[:, None]
    nn = lib_i[nn]

    if not (libOverlap or exclusionRadius_knn or k_query > knn):
        return nn

    pred_col = pred_i[:, None]
    if exclusionRadius_knn:
        mask = abs(pred_col - nn) <= exclusionRadius
    elif libOverlap:
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
def transform_data(data, kwargs):
    """DataFrame + pyEDM Simplex kwargs -> simplex_projection data args
    with a per-query (batched) library mask holding exactly the pyEDM
    FindNeighbors neighbor set."""
    emb, lib_i, pred_i, libOverlap = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)

    X = emb[lib_i]
    Y = target[lib_i + kwargs["Tp"]]
    Q = emb[pred_i]

    knn = transform_args(kwargs)["k"]
    nn = _neighbors(emb, lib_i, pred_i, libOverlap, kwargs["exclusionRadius"], knn)

    pos = empty(emb.shape[0], dtype=int)
    pos[lib_i] = arange(len(lib_i))
    mask = zeros((len(pred_i), len(lib_i)), dtype=bool)
    mask[arange(len(pred_i))[:, None], pos[nn]] = True

    B = len(pred_i)
    return dict(
        X=broadcast_to(X, (B, *X.shape)),
        Y=broadcast_to(Y[:, None], (B, len(Y), 1)),
        Q=Q[:, None, :],
        mask=mask,
    )


# ------------------------------------------------------------
def _rho(data, kwargs, predictions):
    """Pearson rho between predictions and observations over finite
    pairs, rounded to 6 digits (pyEDM AuxFunc.ComputeError)."""
    _, _, pred_i, _ = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)

    obs_i = pred_i + kwargs["Tp"]
    inRange = (obs_i >= 0) & (obs_i < len(target))
    obs = full(len(pred_i), nan)
    obs[inRange] = target[obs_i[inRange]]

    pred = asarray(predictions).reshape(-1)
    finite = isfinite(pred) & isfinite(obs)
    return round(corrcoef(obs[finite], pred[finite])[0, 1], 6)


# ------------------------------------------------------------
def simplex_rho(data, kwargs):
    """pyEDM Simplex kwargs -> ComputeError rho via the edmkit
    simplex_projection mirror."""
    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    return _rho(data, kwargs, pred)


# ------------------------------------------------------------
def embed_dimension(data, kwargs):
    """pyEDM EmbedDimension(data, **kwargs) via edmkit
    simplex_projection: DataFrame of rho over E = 1..maxE."""
    Evals = list(range(1, kwargs["maxE"] + 1))
    rhoList = [simplex_rho(data, _simplex_kwargs(kwargs, E)) for E in Evals]
    return DataFrame({"E": Evals, "rho": rhoList})
