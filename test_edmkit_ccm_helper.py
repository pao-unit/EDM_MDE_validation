import numpy as np
from edmkit.ccm import ccm
from edmkit.simplex_projection import simplex_projection
from pandas import DataFrame
from scipy.spatial import KDTree

from test_edmkit_simplex_projection_helper import _embedding, _names

# ------------------------------------------------------------
# pyEDM CCM and edmkit ccm share the same bootstrap skeleton: for each
# library size, sample library points, cross map every prediction point
# from that library, score the sample with Pearson rho, and aggregate
# over samples.  The mirror therefore drives edmkit.ccm.ccm directly
# and injects the pyEDM specifics through its extension points:
#
#   - sample_func replicates the pyEDM Project() library sampling:
#     points drawn without replacement and sorted, with a fresh
#     generator per (libSize, direction) task.  pyEDM rebuilds each
#     task generator as SeedSequence(child.entropy), which discards
#     the spawn key, so with a fixed seed every task draws the same
#     sequence; the mirror reproduces that through the same .entropy.
#
#   - predict_func takes the neighbor sets from scipy KDTree exactly
#     as pyEDM _ccm_for_libsize does (over-query headroom, self-match
#     and temporal exclusion, first-knn compaction), because distance
#     ties at the knn boundary follow KDTree traversal order (see
#     test_edmkit_edim_helper), and hands the selected neighbors to
#     edmkit simplex_projection as per-query libraries.
#
# The per-sample Pearson rho and the aggregation over samples are
# edmkit's own (pearson_correlation, aggregate_func nanmean).
# ------------------------------------------------------------


# ------------------------------------------------------------
def _lib_sizes(libSizes):
    """pyEDM CCM Validate() libSizes: a 3-element [start, stop,
    increment] spec with increment < stop generates the sequence,
    anything else is already the list of library sizes."""
    if isinstance(libSizes, str):
        libSizes = libSizes.split()
    libSizes = [int(L) for L in libSizes]

    if len(libSizes) == 3:
        start, stop, increment = libSizes
        if increment < stop:
            libSizes = list(range(start, stop + 1, increment))
    return libSizes


# ------------------------------------------------------------
def _tp_valid(N, vec, Tp):
    """pyEDM CCM _tp_valid_mask: rows whose Tp-shifted target is in
    bounds and not nan."""
    shifted = np.arange(N) + Tp
    inBounds = (shifted >= 0) & (shifted < N)
    return inBounds & ~np.isnan(vec[np.clip(shifted, 0, N - 1)])


# ------------------------------------------------------------
def _ccm_direction(X, Y, timeIdx, libSizes, knn, exclusionRadius, sample, entropies):
    """One cross map direction over all library sizes via edmkit ccm
    (the pyEDM Project() fwd or rev task sequence)."""
    M = X.shape[0]
    sizes = np.array([min(L, M) for L in libSizes])  # _ccm_for_libsize min(L, M)

    # pyEDM _ccm_for_libsize library sampling: without replacement,
    # sorted, from a fresh per-task generator.  edmkit bootstrap
    # consumes samples in exactly the pyEDM task order (libSize-major,
    # sample-minor), so the draws are pre-generated and handed out from
    # two cursors: one for sample_func, one for predict_func (which
    # needs the indices for self-match and temporal exclusion).
    draws = []
    for i, L in enumerate(sizes):
        rng = np.random.default_rng(np.random.SeedSequence(entropies[i]))
        for _ in range(sample):
            lib = rng.choice(M, size=int(L), replace=False)
            lib.sort()
            draws.append(lib)
    sampleCursor = iter(draws)
    libCursor = iter(draws)

    def sample_func(pool, size):
        return next(sampleCursor)

    rowIdx = np.arange(M)[:, None]

    def predict_func(X, Y, Q, *, mask=None):
        """pyEDM _ccm_for_libsize neighbor selection, then edmkit
        simplex_projection over the selected neighbor sets."""
        B, L, _ = X.shape
        k = min(knn, L - 1)
        kQuery = k + 1 + (2 * exclusionRadius if exclusionRadius > 0 else 0)
        kQuery = min(kQuery, L)

        preds = np.empty((B, M, 1))
        for b in range(B):
            libIdx = next(libCursor)
            _, nnLocal = KDTree(X[b]).query(Q[b], k=kQuery)
            if nnLocal.ndim == 1:
                nnLocal = nnLocal[:, None]
            nnGlobal = libIdx[nnLocal]

            mask = nnGlobal == rowIdx
            if exclusionRadius > 0:
                mask |= abs(timeIdx[:, None] - timeIdx[nnGlobal]) <= exclusionRadius

            valid = ~mask
            cs = np.cumsum(valid, axis=1)
            firstK = valid & (cs <= k)
            insufficient = cs[:, -1] < k
            firstK[insufficient] = False

            nnCols = np.zeros((M, k), dtype=np.intp)
            _, colPos = np.where(firstK)
            nnCols[~insufficient] = colPos.reshape(-1, k)
            nnLib = nnLocal[rowIdx, nnCols]  # (M, k) library positions

            p = simplex_projection(
                X=X[b][nnLib], Y=Y[b][nnLib], Q=Q[b][:, None, :], k=k
            ).reshape(M)
            p[insufficient] = np.nan
            preds[b, :, 0] = p
        return preds

    return ccm(
        X,
        Y,
        sizes,
        predict_func,
        n_samples=sample,
        library_pool=np.arange(M),
        prediction_pool=np.arange(M),
        sample_func=sample_func,
        aggregate_func=np.nanmean,
    )


# ------------------------------------------------------------
def edmkit_ccm(data, kwargs):
    """pyEDM CCM(data, **kwargs) via edmkit ccm: DataFrame of LibSize
    and mean cross map rho for both directions."""
    columns, target = _names(kwargs["columns"]), _names(kwargs["target"])
    E = len(columns) if kwargs["embedded"] else kwargs["E"]
    knn = kwargs["knn"] if kwargs["knn"] > 0 else E + 1
    libSizes = _lib_sizes(kwargs["libSizes"])

    tgt_vec = data[target[0]].to_numpy(dtype=float)
    col_vec = data[columns[0]].to_numpy(dtype=float)
    N = data.shape[0]

    root = np.random.SeedSequence(kwargs["seed"])
    entropies = [seq.entropy for seq in root.spawn(2 * len(libSizes))]

    # fwd: columns embedding cross maps target[0] (even task seeds);
    # rev: target embedding cross maps columns[0] (odd task seeds)
    rho = []
    for d, (embCols, predVec, ownVec) in enumerate(
        [(columns, tgt_vec, col_vec), (target, col_vec, tgt_vec)]
    ):
        emb = _embedding(
            data,
            dict(columns=embCols, embedded=kwargs["embedded"], E=E, tau=kwargs["tau"]),
        )
        valid = ~np.isnan(emb).any(axis=1)
        if len(kwargs["validLib"]):
            valid &= np.asarray(kwargs["validLib"], dtype=bool)
        valid &= _tp_valid(N, predVec, kwargs["Tp"]) & ~np.isnan(ownVec)
        idx = np.flatnonzero(valid)

        rho.append(
            _ccm_direction(
                X=np.ascontiguousarray(emb[idx]),
                Y=predVec[idx + kwargs["Tp"]],
                timeIdx=idx,
                libSizes=libSizes,
                knn=knn,
                exclusionRadius=kwargs["exclusionRadius"],
                sample=kwargs["sample"],
                entropies=entropies[d::2],
            )
        )

    return DataFrame(
        {
            "LibSize": libSizes,
            f"{columns[0]}:{target[0]}": rho[0],
            f"{target[0]}:{columns[0]}": rho[1],
        }
    )
