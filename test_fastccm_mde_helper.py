"""Parked FastCCM-backed MDE compatibility helpers.

The current MDE mirror uses the parked EmbedDimension compatibility harness as
a downstream gate for candidate variables.  Keep this module only as historical
context unless MDE is rewritten around FastCCM-native search semantics.
"""

from numpy import append, array, greater, nan_to_num
from pandas import DataFrame
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

from conftest import EmbedDimensionArgs, SimplexArgs
from test_fastccm_ccm_helper import simplex_score_curve
from test_fastccm_embed_dimension_helper import (
    _fastccm_simplex_predictions,
    compute_error_rho,
    embed_dimension,
    simplex_observations,
)


# ------------------------------------------------------------
def _numeric_frame(data, kwargs):
    """MDE Validate()/PrepareNumericFrame subset needed by the tests."""
    data = data.copy()
    if kwargs["removeTime"]:
        data = data.drop(columns=data.columns[0])
    return data


# ------------------------------------------------------------
def _default_spans(data, kwargs):
    kwargs = dict(kwargs)
    if not len(kwargs["lib"]):
        kwargs["lib"] = [1, int(data.shape[0] / 2)]
    if not len(kwargs["pred"]):
        kwargs["pred"] = [int(data.shape[0] / 2) + 1, data.shape[0]]
    return kwargs


# ------------------------------------------------------------
def _simplex_kwargs(kwargs, columns):
    return SimplexArgs | dict(
        columns=columns,
        target=kwargs["target"],
        lib=kwargs["lib"],
        pred=kwargs["pred"],
        E=0,
        Tp=kwargs["Tp"],
        tau=kwargs["tau"],
        exclusionRadius=kwargs["exclusionRadius"],
        embedded=True,
        validLib=[],
        noTime=True,
        ignoreNan=True,
    )


# ------------------------------------------------------------
def cross_map_columns(data, columns, kwargs):
    """FastCCM-backed CrossMapColumns-style rho dictionary."""
    rhoD = {}
    for column_list in columns:
        simplex_kwargs = _simplex_kwargs(kwargs, column_list)
        pred = _fastccm_simplex_predictions(data, simplex_kwargs)
        rho = compute_error_rho(simplex_observations(data, simplex_kwargs), pred)
        rhoD[f"{','.join(column_list)}:{kwargs['target']}"] = (rho, list(column_list))
    return rhoD


# ------------------------------------------------------------
def _embed_dimension(data, column, kwargs):
    # Parked downstream parity path. Do not extend this unless MDE is still
    # expected to reproduce pyEDM's independent per-embedding lengths.
    edim_kwargs = EmbedDimensionArgs | dict(
        columns=column,
        target=kwargs["target"],
        maxE=kwargs["maxE"],
        lib=kwargs["lib"],
        pred=kwargs["pred"],
        Tp=kwargs["Tp"],
        tau=kwargs["tau"],
        exclusionRadius=kwargs["exclusionRadius"],
        embedded=False,
        validLib=[],
        noTime=True,
        ignoreNan=True,
    )
    return embed_dimension(data, edim_kwargs)


# ------------------------------------------------------------
def _first_or_global_emax(edim_df, first_emax):
    if first_emax:
        local_max = argrelextrema(edim_df["rho"].to_numpy(), greater)[0]
        return local_max[0] if len(local_max) else len(edim_df["E"]) - 1
    return edim_df["rho"].round(4).argmax()


# ------------------------------------------------------------
def _ccm_slope(data, column, E, lib_sizes, lib_sizes_vec, kwargs):
    """FastCCM-native target:column convergence slope used by MDE."""
    E = int(E)
    ccm_vals = simplex_score_curve(
        data[kwargs["target"]].to_numpy(dtype=float),
        data[column].to_numpy(dtype=float),
        lib_sizes,
        E=E,
        tau=abs(kwargs["tau"]),
        sample_size=None,
        exclusion_window=kwargs["exclusionRadius"],
        tp=kwargs["Tp"],
        seed=kwargs["ccmSeed"],
        nbrs_num=E + 1,
        trials=kwargs["sample"],
        batch_size=None,
        clean_after=False,
    )
    lm = LinearRegression().fit(lib_sizes_vec, nan_to_num(ccm_vals))
    return round(lm.coef_[0], 5)


# ------------------------------------------------------------
def fastccm_mde(data, kwargs):
    """MDE Run() subset backed by FastCCM cross-map and EDim."""
    data = _numeric_frame(data, kwargs)
    kwargs = _default_spans(data, kwargs)

    lib_sizes = [int(data.shape[0] * (p / 100)) for p in kwargs["pLibSizes"]]
    lib_sizes_vec = array(lib_sizes, dtype=float).reshape(-1, 1)
    lib_sizes_vec = lib_sizes_vec / lib_sizes_vec[-1]

    data_columns = list(set(data.columns) - set(kwargs["removeColumns"]))
    mde_columns = []
    mde_rho = array([], dtype=float)
    edim_cache = {}
    ccm_cache = {}

    for d in range(1, kwargs["D"] + 1):
        columns = list(set(data_columns) - set(mde_columns))
        columns = [[c] + mde_columns for c in columns]

        rhoD = cross_map_columns(data, columns, kwargs)
        ranked = sorted(rhoD.values(), key=lambda x: x[0], reverse=True)
        rho = array([_[0] for _ in ranked])
        keep_n = int((rho > kwargs["crossMapRhoMin"]).sum())
        if keep_n < 1:
            continue
        ranked = ranked[:keep_n]

        if kwargs["noCCM"]:
            new_column = ranked[0][1][0]
            mde_columns.append(new_column)
            mde_rho = append(mde_rho, ranked[0][0])
            continue

        max_col_i = None
        new_column = None
        for col_i, (cross_map_rho, columns_i) in enumerate(ranked):
            new_column = columns_i[0]

            if kwargs["E"] > 0:
                max_edim = kwargs["E"]
                max_rho_edim = round(float(cross_map_rho), 4)
            elif new_column in edim_cache:
                max_edim, max_rho_edim = edim_cache[new_column]
            else:
                edim = _embed_dimension(data, new_column, kwargs)
                i_max = _first_or_global_emax(edim, kwargs["firstEMax"])
                max_rho_edim = edim["rho"].iloc[i_max].round(4)
                max_edim = edim["E"].iloc[i_max]
                edim_cache[new_column] = (max_edim, max_rho_edim)

            if max_rho_edim < kwargs["embedDimRhoMin"]:
                continue

            if new_column in ccm_cache:
                slope = ccm_cache[new_column]
            else:
                slope = _ccm_slope(
                    data, new_column, max_edim, lib_sizes, lib_sizes_vec, kwargs
                )
                ccm_cache[new_column] = slope

            if slope > kwargs["ccmSlope"]:
                max_col_i = col_i
                break

        if max_col_i is None:
            break

        mde_columns.append(new_column)
        mde_rho = append(mde_rho, ranked[max_col_i][0])

    return DataFrame({"variables": mde_columns, "rho": mde_rho})
