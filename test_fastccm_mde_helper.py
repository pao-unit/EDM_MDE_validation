"""FastCCM-backed MDE helpers for the vectorizable validation path.

MDE still uses pyEDM's EmbedDimension as the candidate gate when E is not fixed.
The FastCCM part under test is the expensive cross-map and CCM scoring work,
restricted to the cases that map cleanly to batched FastCCM calls.
"""

from multiprocessing import cpu_count

from fastccm import PairwiseCCM
from fastccm import utils as fastccm_utils
from numpy import (
    append,
    array,
    asarray,
    corrcoef,
    full,
    greater,
    isfinite,
    nan,
    nan_to_num,
)
from numpy import round as npround
from pandas import DataFrame
from pyEDM import EmbedDimension
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

from conftest import EmbedDimensionArgs


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
def _compute_error_rho(obs, pred, digits=6):
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
def cross_map_columns(data, columns, kwargs):
    """Batched FastCCM CrossMapColumns-style rho dictionary.

    This intentionally supports only disjoint lib/pred with no exclusion radius:
    that is the path where every candidate can be evaluated in one vectorized
    FastCCM predict_matrix call.  Exclusion-radius parity needs query-specific
    neighbor masking and is deliberately left out of the active MDE mirror.
    """
    if kwargs["exclusionRadius"] > 0:
        raise NotImplementedError(
            "FastCCM MDE active tests cover only disjoint lib/pred with "
            "exclusionRadius=0, so CrossMapColumns can stay fully batched."
        )

    lib_start, lib_end = [int(v) for v in kwargs["lib"]]
    pred_start, pred_end = [int(v) for v in kwargs["pred"]]
    tp = int(kwargs["Tp"])
    target = array(data[kwargs["target"]].to_numpy(dtype=float), copy=True)[:, None]
    groups = {}
    for column_list in columns:
        groups.setdefault(len(column_list), []).append(column_list)

    rhoD = {}
    simplex = PairwiseCCM(device="cpu", memory_budget_gb=3)

    for group in groups.values():
        X_lib = []
        X_pred = []
        obs = None

        for column_list in group:
            X = array(
                data.loc[:, column_list].to_numpy(dtype=float),
                copy=True,
                order="C",
            )
            X_lib.append(X[lib_start - 1 : lib_end])

            pred_src_start = pred_start - 1
            pred_src_end = pred_end
            if tp > 0:
                pred_tgt_start = pred_start - 1 + tp
                pred_tgt_end = pred_end + tp
            elif tp < 0:
                pred_src_start = pred_start - 1 - tp
                pred_src_end = pred_end - tp
                pred_tgt_start = pred_start - 1
                pred_tgt_end = pred_end
            else:
                pred_tgt_start = pred_start - 1
                pred_tgt_end = pred_end

            pred_src_start = max(0, pred_src_start)
            pred_tgt_start = max(0, pred_tgt_start)
            pred_src_end = min(len(X), pred_src_end)
            pred_tgt_end = min(len(target), pred_tgt_end)
            valid_len = min(pred_src_end - pred_src_start, pred_tgt_end - pred_tgt_start)
            if valid_len <= 0:
                raise ValueError("Prediction interval leaves no prediction samples.")

            pred_src_end = pred_src_start + valid_len
            pred_tgt_end = pred_tgt_start + valid_len
            X_pred.append(X[pred_src_start:pred_src_end])

            if obs is None:
                obs = target[pred_tgt_start:pred_tgt_end, 0]

        pred = simplex.predict_matrix(
            X_lib_emb=X_lib,
            Y_lib_emb=[target[lib_start - 1 : lib_end]],
            X_pred_emb=X_pred,
            library_size=lib_end - lib_start + 1,
            method="simplex",
            tp=tp,
            exclusion_window=kwargs["exclusionRadius"],
            batch_size=None,
            target_batch_size=None,
            clean_after=False,
        )

        for i, column_list in enumerate(group):
            rho = _compute_error_rho(obs, asarray(pred)[:, 0, 0, i])
            rhoD[f"{','.join(column_list)}:{kwargs['target']}"] = (
                rho,
                list(column_list),
            )

    return rhoD


# ------------------------------------------------------------
def _embed_dimension(data, column, kwargs):
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

    edim_kwargs.update(
        mpMethod=kwargs["mpMethod"],
        verbose=kwargs["verbose"],
        numProcess=max(1, min(kwargs["maxE"], cpu_count() or 1)),
        kdWorkers=1,
        showPlot=False,
    )
    return EmbedDimension(dataFrame=data, **edim_kwargs)


# ------------------------------------------------------------
def _first_or_global_emax(edim_df, first_emax):
    if first_emax:
        local_max = argrelextrema(edim_df["rho"].to_numpy(), greater)[0]
        return local_max[0] if len(local_max) else len(edim_df["E"]) - 1
    return edim_df["rho"].round(4).argmax()


# ------------------------------------------------------------
def _ccm_slope(curves, column, E, lib_sizes_vec):
    """Linear convergence slope for an already-computed CCM rho curve."""
    lm = LinearRegression().fit(lib_sizes_vec, nan_to_num(curves[(column, int(E))]))
    return round(lm.coef_[0], 5)


# ------------------------------------------------------------
def _fastccm_embedding(series, E, tau):
    return array(
        fastccm_utils.embed(asarray(series, dtype=float)[:, None], E=E, tau=tau)[0],
        copy=True,
        order="C",
    )


# ------------------------------------------------------------
def _ccm_curves_grouped_by_e(data, columns_by_e, lib_sizes, kwargs):
    """FastCCM CCM rho curves for candidate groups sharing the same E."""
    tau = abs(int(kwargs["tau"]))
    sample = max(1, int(kwargs["sample"]))
    target = data[kwargs["target"]].to_numpy(dtype=float)
    curves = {}
    ccm = PairwiseCCM(device="cpu", memory_budget_gb=3)

    for E, columns in columns_by_e.items():
        E = int(E)
        X_emb = [
            _fastccm_embedding(data[column].to_numpy(dtype=float), E, tau)
            for column in columns
        ]
        Y_emb = [_fastccm_embedding(target, E, tau)]
        column_curves = full((len(lib_sizes), len(columns)), nan, dtype=float)

        for lib_i, lib_size in enumerate(lib_sizes):
            trial_scores = []
            for trial_i in range(sample):
                trial_seed = (
                    None if kwargs["ccmSeed"] is None else int(kwargs["ccmSeed"]) + trial_i
                )
                scores = ccm.score_matrix(
                    X_emb=X_emb,
                    Y_emb=Y_emb,
                    library_size=int(lib_size),
                    sample_size=None,
                    exclusion_window=kwargs["exclusionRadius"],
                    tp=kwargs["Tp"],
                    method="simplex",
                    seed=trial_seed,
                    clean_after=False,
                )
                trial_scores.append(asarray(scores, dtype=float)[0, 0, :])

            column_curves[lib_i, :] = asarray(trial_scores, dtype=float).mean(axis=0)

        for column_i, column in enumerate(columns):
            curves[(column, E)] = column_curves[:, column_i]

    return curves


# ------------------------------------------------------------
def fastccm_mde(data, kwargs):
    """MDE Run() subset backed by FastCCM cross-map and EDim."""
    data = _numeric_frame(data, kwargs)
    kwargs = _default_spans(data, kwargs)

    lib_sizes = [int(data.shape[0] * (p / 100)) for p in kwargs["pLibSizes"]]
    lib_sizes_vec = array(lib_sizes, dtype=float).reshape(-1, 1)
    lib_sizes_vec = lib_sizes_vec / lib_sizes_vec[-1]

    data_columns = sorted(set(data.columns) - set(kwargs["removeColumns"]))
    mde_columns = []
    mde_rho = array([], dtype=float)
    edim_cache = {}
    ccm_cache = {}

    for d in range(1, kwargs["D"] + 1):
        columns = sorted(set(data_columns) - set(mde_columns))
        columns = [[c] + mde_columns for c in columns]

        rhoD = cross_map_columns(data, columns, kwargs)
        ranked = sorted(rhoD.values(), key=lambda x: (-float(x[0]), tuple(x[1])))
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

        gated = []
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

            gated.append((col_i, new_column, int(max_edim)))

        missing_by_e = {}
        for _, new_column, max_edim in gated:
            ccm_key = (new_column, max_edim)
            if ccm_key not in ccm_cache:
                missing_by_e.setdefault(max_edim, []).append(new_column)

        if missing_by_e:
            ccm_cache.update(
                _ccm_curves_grouped_by_e(data, missing_by_e, lib_sizes, kwargs)
            )

        max_col_i = None
        new_column = None
        for col_i, new_column, max_edim in gated:
            slope = _ccm_slope(ccm_cache, new_column, max_edim, lib_sizes_vec)

            if slope > kwargs["ccmSlope"]:
                max_col_i = col_i
                break

        if max_col_i is None:
            break

        mde_columns.append(new_column)
        mde_rho = append(mde_rho, ranked[max_col_i][0])

    return DataFrame({"variables": mde_columns, "rho": mde_rho})
