import numpy as np
from edmkit.smap import smap, weights
from edmkit.util import pairwise_distance_np
from numpy.linalg import solve

from test_edmkit_simplex_projection_helper import _indices, _names

# ------------------------------------------------------------
# pyEDM SMap conventions vs edmkit smap:
#   - pyEDM defaults knn to len(lib_i) - 1: each query regresses on its
#     knn nearest library points, dropping the farthest one (and the
#     self-match when lib and pred overlap).  edmkit uses every unmasked
#     library point, so the knn selection is expressed through a
#     per-query (batched) library mask.
#   - pyEDM multiplies the design-matrix rows by w and solves lstsq,
#     minimizing sum(w^2 r^2).  edmkit applies w once in the normal
#     equations, minimizing sum(w r^2).  With w = exp(-theta d / d_mean)
#     doubling theta makes the two objectives identical.
#   - pyEDM lstsq applies no regularization; edmkit's default Tikhonov
#     alpha = 1e-10 shifts near-singular solutions visibly at 6 decimal
#     places, so alpha is shrunk to 1e-13.
# ------------------------------------------------------------


# ------------------------------------------------------------
def transform_args(kwargs):
    """pyEDM SMap kwargs -> edmkit smap kwargs (see conventions above)."""
    return dict(theta=2.0 * kwargs["theta"], alpha=1e-13)


# ------------------------------------------------------------
def _library(data, kwargs):
    """X, Y, Q, query : library distances D and the (N_pred, N_lib) mask
    selecting each query's knn nearest library points (pyEDM
    FindNeighbors), excluded points ranked last."""
    emb, lib_i, pred_i, libOverlap = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)

    X = emb[lib_i]
    Y = target[lib_i + kwargs["Tp"]]
    Q = emb[pred_i]

    D = np.sqrt(pairwise_distance_np(Q, X))

    exclusionRadius = kwargs["exclusionRadius"]
    if exclusionRadius > 0:
        excluded = abs(pred_i[:, None] - lib_i[None, :]) <= exclusionRadius
    elif libOverlap:
        excluded = pred_i[:, None] == lib_i[None, :]
    else:
        excluded = np.zeros(D.shape, dtype=bool)

    knn = kwargs["knn"] if kwargs["knn"] > 0 else len(lib_i) - 1

    order = np.argsort(np.where(excluded, np.inf, D), axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(D.shape[0])[:, None], order] = np.tile(
        np.arange(D.shape[1]), (D.shape[0], 1)
    )
    mask = (rank < knn) & ~excluded

    return X, Y, Q, D, mask


# ------------------------------------------------------------
def transform_data(data, kwargs):
    """DataFrame + pyEDM SMap kwargs -> edmkit smap data args.

    X : library embedding vectors        emb[lib_i]
    Y : library targets Tp rows ahead    target[lib_i + Tp]
    Q : query embedding vectors          emb[pred_i]

    One batch per query so each query carries its own knn library mask.
    """
    X, Y, Q, _, mask = _library(data, kwargs)

    B = len(Q)
    return dict(
        X=np.tile(X, (B, 1, 1)),
        Y=np.tile(Y[:, None], (B, 1, 1)),
        Q=Q[:, None, :],
        mask=mask,
    )


# ------------------------------------------------------------
def nan_target_predictions(data, kwargs):
    """(N_pred,) predictions when the target column contains nan.

    pyEDM keeps nan-target neighbors in the mean-distance weight
    normalization but drops them from the regression; edmkit's mask
    drops them from both.  Calling smap per query with theta rescaled
    by the ratio of the two mean distances reproduces pyEDM's weights
    exactly.
    """
    X, Y, Q, D, mask = _library(data, kwargs)
    regMask = mask & ~np.isnan(Y)[None, :]

    alpha = transform_args(kwargs)["alpha"]
    predictions = np.empty(len(Q))
    for b in range(len(Q)):
        keep = regMask[b]
        theta = 2.0 * kwargs["theta"] * D[b, keep].mean() / D[b, mask[b]].mean()
        predictions[b] = smap(X[keep], Y[keep], Q[b : b + 1], theta=theta, alpha=alpha)
    return predictions


# ------------------------------------------------------------
def smap_coefficients(data, kwargs):
    """(N_pred, E + 1) rows of [C0, dTarget/dColumn ...] matching pyEDM
    SMap(returnObject = True).coefficients.

    edmkit smap does not return the local linear maps, so they are
    recomputed from edmkit's weights() with the same regularized normal
    equations smap solves internally.
    """
    X, Y, _, D, mask = _library(data, kwargs)
    E = X.shape[-1]
    args = transform_args(kwargs)

    W = weights(D[:, None, :], args["theta"], mask=mask, min_points=E + 1)  # (B, 1, N)
    X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=-1)  # (N, E+1)

    XTX = np.einsum("bpn,ni,nj->bpij", W, X_aug, X_aug)  # (B, 1, E+1, E+1)
    XTY = np.einsum("bpn,ni,n->bpi", W, X_aug, Y)  # (B, 1, E+1)

    reg = np.identity(E + 1)
    reg[0, 0] = 0  # do not regularize the intercept term
    tr = np.maximum(np.trace(XTX, axis1=-2, axis2=-1), 1e-12)  # (B, 1)
    XTX = XTX + (args["alpha"] * tr)[..., None, None] * reg

    C = solve(XTX, XTY[..., None])  # (B, 1, E+1, 1)
    return C[:, 0, :, 0]
