from fastccm import PairwiseCCM
from numpy import asarray, corrcoef, full, isfinite, nan
from numpy import round as npround
from pandas import DataFrame
from pyEDM import ComputeError, Simplex

from conftest import SimplexArgs
from test_edmkit_simplex_projection_helper import _dimension, _indices, _names
from test_fastccm_simplex_projection_helper import (
    simplex_projection,
    transform_args,
    transform_data,
)


# ------------------------------------------------------------
def compute_error_rho(obs, pred, digits=6):
    """pyEDM ComputeError rho for EmbedDimension's Simplex outputs."""
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
def _is_contiguous(indices):
    return len(indices) > 0 and (indices == range(indices[0], indices[0] + len(indices))).all()


# ------------------------------------------------------------
def _fastccm_simplex_predictions(data, kwargs):
    """FastCCM Simplex predictions, vectorized when native exclusion is aligned."""
    emb, lib_i, pred_i, lib_overlap = _indices(data, kwargs)

    native_exclusion = (
        lib_overlap
        and _is_contiguous(lib_i)
        and _is_contiguous(pred_i)
        and lib_i[0] == pred_i[0]
    )
    no_exclusion_needed = not lib_overlap and kwargs["exclusionRadius"] == 0

    if not (native_exclusion or no_exclusion_needed):
        pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
        return asarray(pred)[:, 0, 0, 0]

    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)
    lib_target_i = lib_i + kwargs["Tp"]
    exclusion_window = None
    if native_exclusion:
        exclusion_window = max(0, kwargs["exclusionRadius"])

    pred = PairwiseCCM(
        device="cpu", dtype="float64", compute_dtype="float64"
    ).predict_matrix(
        X_lib_emb=[emb[lib_i]],
        Y_lib_emb=[target[lib_target_i, None]],
        X_pred_emb=[emb[pred_i]],
        library_size=len(lib_i),
        method="simplex",
        nbrs_num=kwargs["knn"] if kwargs["knn"] > 0 else _dimension(kwargs) + 1,
        tp=0,
        exclusion_window=exclusion_window,
        seed=1,
        batch_size=None,
        target_batch_size=None,
        clean_after=False,
    )
    return asarray(pred)[:, 0, 0, 0]


# ------------------------------------------------------------
def embed_dimension(data, kwargs):
    """FastCCM-backed equivalent of pyEDM EmbedDimension."""
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


# ------------------------------------------------------------
def pyedm_embed_dimension(data, kwargs):
    """Sequential pyEDM EmbedDimension reference for parity tests."""
    rho = []
    for E in range(1, kwargs["maxE"] + 1):
        df = Simplex(
            dataFrame=data,
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
        rho.append(ComputeError(df["Observations"], df["Predictions"])["rho"])
    return DataFrame({"E": range(1, kwargs["maxE"] + 1), "rho": rho})
