"""Validation tests for edmkit smap against pyEDM ValidOutput"""

import pyEDM as EDM
import pytest
from edmkit.smap import smap
from numpy import nan

from conftest import SMapArgs, ValidData
from test_edmkit_simplex_projection_helper import transform_result, transform_valid
from test_edmkit_smap_helper import (
    nan_target_predictions,
    smap_coefficients,
    transform_args,
    transform_data,
)


# ------------------------------------------------------------
def test_smap1():
    """embedded = False"""
    data = EDM.sampleData["circle"]
    kwargs = SMapArgs.copy()
    kwargs.update(
        dict(columns="x", target="x", lib=[1, 100], pred=[110, 160], E=4, theta=3.0)
    )

    pred = smap(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("SMap_circle_E4_valid.csv")

    smap_ = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smap_.equals(valid)


# ------------------------------------------------------------
def test_smap2():
    """SMap embedded = True coefficients"""
    data = EDM.sampleData["circle"]
    kwargs = SMapArgs.copy()
    kwargs.update(
        dict(
            columns=["x", "y"],
            target="x",
            lib=[1, 200],
            pred=[1, 200],
            theta=3.0,
            embedded=True,
        )
    )

    pred = smap(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("SMap_circle_E2_embd_valid.csv")

    smap_ = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smap_.equals(valid)

    C = smap_coefficients(data, kwargs)
    assert C[:, 1].mean().round(5) == 0.99801  # ∂x/∂x
    assert C[:, 2].mean().round(5) == 0.06311  # ∂x/∂y


# ------------------------------------------------------------
def test_smap3():
    """SMap nan"""
    data = EDM.sampleData["circle"]
    dfn = data.copy()
    dfn.iloc[[5, 6, 12], 1] = nan
    dfn.iloc[[10, 11, 17], 2] = nan

    kwargs = SMapArgs.copy()
    kwargs.update(
        dict(columns="x", target="y", lib=[1, 50], pred=[1, 50], E=2, theta=3.0)
    )

    pred = nan_target_predictions(dfn, kwargs)
    dfv = ValidData("SMap_nan_valid.csv")

    smap_ = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smap_.equals(valid)


# ------------------------------------------------------------
def test_smap4():
    """SMap embedded = True coefficients"""
    data = EDM.sampleData["Lorenz5D"]
    kwargs = SMapArgs.copy()
    kwargs.update(
        dict(
            columns=["V1", "V2", "V3"],
            target="V5",
            lib=[1, 300],
            pred=[501, 600],
            Tp=5,
            theta=3.0,
            embedded=True,
        )
    )

    pred = smap(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("SMap_Lorenz5D_pred_valid.csv")
    dfcv = ValidData("SMap_Lorenz5D_coef_valid.csv")

    smap_ = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smap_.equals(valid)

    C = smap_coefficients(data, kwargs)
    coefValid = dfcv.drop(columns="Time").dropna().to_numpy()
    assert C == pytest.approx(coefValid, 1e-6)
