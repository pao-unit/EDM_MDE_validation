"""FastCCM CCM mirrors for vectorized public score_matrix paths."""

import numpy as np
import pytest
from fastccm import PairwiseCCM
from pandas import DataFrame
from pyEDM import sampleData

from conftest import CCMArgs, ValidData


def _names(spec):
    return spec.split() if isinstance(spec, str) else list(spec)


def _lib_sizes(spec):
    return list(range(spec[0], spec[1] + 1, spec[2])) if len(spec) == 3 else list(spec)


def _embedding(data, columns, E, tau):
    rows = np.arange(len(data))
    parts = []
    valid = np.ones(len(data), dtype=bool)

    for column in _names(columns):
        x = data[column].to_numpy(dtype=float)
        emb = np.full((len(x), E), np.nan)
        for lag in range(E):
            src = rows + tau * lag
            ok = (src >= 0) & (src < len(x))
            emb[ok, lag] = x[src[ok]]
        parts.append(emb)
        valid &= ~np.isnan(emb).any(axis=1)

    return np.hstack(parts), valid


def _ccm_arrays(data, columns, target, kwargs):
    Tp = kwargs["Tp"]
    target_values = data[_names(target)[0]].to_numpy(dtype=float)
    target_i = np.arange(len(data)) + Tp
    target_ok = (target_i >= 0) & (target_i < len(data))

    emb, valid = _embedding(data, columns, kwargs["E"], kwargs["tau"])
    valid &= target_ok & ~np.isnan(target_values[np.clip(target_i, 0, len(data) - 1)])
    rows = np.flatnonzero(valid)

    return emb[rows].copy(), target_values[rows + Tp, None].copy()


def _rho_curve(data, columns, target, lib_sizes, kwargs):
    X, Y = _ccm_arrays(data, columns, target, kwargs)
    ccm = PairwiseCCM(device="cpu", dtype="float64", compute_dtype="float64")

    rho = []
    for lib_size in lib_sizes:
        trial_scores = []
        for trial in range(kwargs["sample"]):
            score = ccm.score_matrix(
                X_emb=[X],
                Y_emb=[Y],
                library_size=lib_size,
                sample_size=None,
                exclusion_window=kwargs["exclusionRadius"],
                tp=0,
                method="simplex",
                seed=kwargs["seed"] + trial,
                batch_size=None,
                clean_after=False,
            )
            trial_scores.append(np.asarray(score)[-1, 0, 0])
        rho.append(np.mean(trial_scores))

    return np.asarray(rho)


def fastccm_ccm(data, kwargs):
    lib_sizes = _lib_sizes(kwargs["libSizes"])
    columns = _names(kwargs["columns"])[0]
    target = _names(kwargs["target"])[0]

    return DataFrame(
        {
            "LibSize": lib_sizes,
            f"{columns}:{target}": _rho_curve(
                data, kwargs["columns"], kwargs["target"], lib_sizes, kwargs
            ),
            f"{target}:{columns}": _rho_curve(
                data, kwargs["target"], kwargs["columns"], lib_sizes, kwargs
            ),
        }
    )


@pytest.mark.parametrize(
    ("sample_name", "updates", "valid_file", "tol"),
    [
        (
            "sardine_anchovy_sst",
            dict(
                columns="anchovy",
                target="np_sst",
                libSizes=[10, 20, 30, 40, 50, 60, 70, 75],
                sample=100,
                E=3,
                seed=123,
            ),
            "CCM_anch_sst_valid.csv",
            0.035,
        ),
        (
            "Lorenz5D",
            dict(
                columns="V3 V5",
                target="V1",
                libSizes=[20, 200, 500, 950],
                sample=30,
                E=5,
                Tp=10,
                tau=-5,
                seed=123,
            ),
            "CCM_Lorenz5D_MV_valid.csv",
            0.018,
        ),
        (
            "circle",
            dict(
                columns="x",
                target="y",
                libSizes=[20, 200, 50],
                sample=30,
                E=2,
                Tp=-5,
                seed=123,
            ),
            "CCM_NegativeTp.csv",
            0.025,
        ),
        (
            "Lorenz5D",
            dict(
                columns="V1",
                target="V5",
                libSizes=[50, 1000, 50],
                sample=30,
                E=5,
                Tp=10,
                tau=-5,
                exclusionRadius=20,
                seed=123,
            ),
            "CCM_exclusionRadius.csv",
            0.01,
        ),
        (
            "circle",
            dict(
                columns="x",
                target="y",
                libSizes=[20, 120, 10],
                sample=100,
                E=2,
                tau=3,
                seed=123,
            ),
            "CCM_positiveTau.csv",
            0.026,
        ),
    ],
)
def test_ccm(sample_name, updates, valid_file, tol):
    kwargs = CCMArgs | updates
    ccm = fastccm_ccm(sampleData[sample_name], kwargs)
    valid = ValidData(valid_file)

    assert ccm.iloc[:, 1:].to_numpy() == pytest.approx(
        valid.iloc[:, 1:].to_numpy(),
        abs=tol,
    )
