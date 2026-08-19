from importlib import import_module

import pyEDM as EDM
from fastccm import PairwiseCCM
from fastccm.utils.utils import get_td_embedding_np
from numpy import array_equal, asarray, diff, isfinite
from pandas import DataFrame


# ------------------------------------------------------------
def time_delay_embedding(series, E, tau):
    """1D series -> FastCCM scalar embedding array shaped (1, N, E)."""
    x = asarray(series, dtype=float)[:, None]
    return get_td_embedding_np(x, E, tau).transpose(2, 0, 1)

# ------------------------------------------------------------
def simplex_score_matrix(**kwargs):
    """Run FastCCM's score_matrix simplex path in double precision."""
    return PairwiseCCM(
        device="cpu", dtype="float64", compute_dtype="float64"
    ).score_matrix(**kwargs)


# ------------------------------------------------------------
def fastccm_simplex_rho(x, y, E=3, tau=1, **kwargs):
    """FastCCM X -> Y simplex score for one scalar pair."""
    score = simplex_score_matrix(
        X_emb=time_delay_embedding(x, E=E, tau=tau),
        Y_emb=time_delay_embedding(y, E=E, tau=tau),
        method="simplex",
        **kwargs,
    )
    return float(score[-1].squeeze())


# ------------------------------------------------------------
def pyedm_simplex_rho(data, **kwargs):
    """pyEDM Simplex prediction/observation correlation for X -> Y."""
    result = EDM.Simplex(
        data,
        columns="X",
        target="Y",
        noTime=True,
        **kwargs,
    )
    return float(
        result[["Observations", "Predictions"]].corr()["Predictions"]["Observations"]
    )


# ------------------------------------------------------------
def pyedm_ccm_x_to_y(data, **kwargs):
    """pyEDM CCM X:Y value for one full-library row."""
    result = EDM.CCM(
        data,
        columns="X",
        target="Y",
        noTime=True,
        parallel=False,
        **kwargs,
    )
    return float(result["X:Y"].iloc[0])


# ------------------------------------------------------------
def xy_dataframe(x, y):
    """Two scalar series -> pyEDM noTime DataFrame."""
    return DataFrame(dict(X=x, Y=y))


# ------------------------------------------------------------
def _manual_simplex_score_curve(x, y, lib_sizes, E, tau, **kwargs):
    """FastCCM simplex rho values over increasing library sizes."""
    x_emb = time_delay_embedding(x, E=E, tau=tau)
    y_emb = time_delay_embedding(y, E=E, tau=tau)

    rho = []
    for lib_size in lib_sizes:
        score = simplex_score_matrix(
            X_emb=x_emb,
            Y_emb=y_emb,
            library_size=lib_size,
            method="simplex",
            **kwargs,
        )
        rho.append(score[-1].squeeze())
    return asarray(rho, dtype=float)


# ------------------------------------------------------------
def _ccm_utils_simplex_score_curve(x, y, lib_sizes, E, tau, **kwargs):
    """FastCCM ccm_utils convergence_test rho values for X -> Y."""
    ccm_utils = import_module("fastccm.ccm_utils")
    functions = ccm_utils.Functions(
        device="cpu",
        dtype="float64",
        compute_dtype="float64",
    )
    ccm_kwargs = dict(kwargs)
    ccm_kwargs.pop("clean_after", None)

    result = functions.convergence_test(
        time_delay_embedding(x, E=E, tau=tau),
        time_delay_embedding(y, E=E, tau=tau),
        library_sizes=lib_sizes,
        method="simplex",
        trials=1,
        **ccm_kwargs,
    )

    if not array_equal(result["library_sizes"], asarray(lib_sizes)):
        raise ValueError("ccm_utils returned unexpected library_sizes.")

    rho = asarray(result["X_to_Y"][:, 0, -1, 0, 0], dtype=float)
    if rho.shape != (len(lib_sizes),) or not isfinite(rho).all():
        raise ValueError("ccm_utils returned invalid convergence rho values.")
    return rho


# ------------------------------------------------------------
def simplex_score_curve(x, y, lib_sizes, E, tau, **kwargs):
    """FastCCM simplex rho curve, preferring ccm_utils with manual fallback."""
    try:
        return _ccm_utils_simplex_score_curve(x, y, lib_sizes, E, tau, **kwargs)
    except ImportError, TypeError, ValueError, KeyError, IndexError, RuntimeError:
        return _manual_simplex_score_curve(x, y, lib_sizes, E, tau, **kwargs)


# ------------------------------------------------------------
def convergence_summary(lib_sizes, rho, min_gain=0.0, min_final_rho=0.0):
    """Small pyEDM-style convergence check for CCM validation tests."""
    lib_sizes = asarray(lib_sizes, dtype=float)
    rho = asarray(rho, dtype=float)
    if lib_sizes.ndim != 1 or rho.ndim != 1 or len(lib_sizes) != len(rho):
        raise ValueError("lib_sizes and rho must be same-length 1D arrays.")
    if len(rho) < 2:
        raise ValueError("At least two rho values are required.")
    if (diff(lib_sizes) <= 0).any():
        raise ValueError("lib_sizes must be strictly increasing.")

    gain = rho[-1] - rho[0]
    return dict(
        initial_rho=rho[0],
        final_rho=rho[-1],
        gain=gain,
        converged=bool(gain >= min_gain and rho[-1] >= min_final_rho),
    )
