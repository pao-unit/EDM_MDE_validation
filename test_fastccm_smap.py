"""Validation tests for FastCCM SMap-equivalent predictions against pyEDM ValidOutput"""

import pyEDM as EDM
import pytest

from conftest import SMapArgs, ValidData
from test_fastccm_simplex_projection_helper import transform_result, transform_valid
from test_fastccm_smap_helper import (
    smap,
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

    smap_ = transform_result(pred)
    valid = transform_valid(dfv)
    assert smap_.to_numpy() == pytest.approx(valid.to_numpy(), abs=1e-6)


# ------------------------------------------------------------
def test_smap2():
    """embedded = True disjoint lib/pred"""
    data = EDM.sampleData["circle"]
    kwargs = SMapArgs.copy()
    kwargs.update(
        dict(
            columns=["x", "y"],
            target="x",
            lib=[1, 100],
            pred=[110, 160],
            theta=3.0,
            embedded=True,
        )
    )

    pred = smap(**transform_data(data, kwargs), **transform_args(kwargs))
    pyedm = EDM.SMap(data, **kwargs)

    smap_ = transform_result(pred)
    valid = transform_valid(pyedm["predictions"])
    assert smap_.to_numpy() == pytest.approx(valid.to_numpy(), abs=1e-6)


# ------------------------------------------------------------
def test_smap3():
    """Tp = 0 disjoint lib/pred"""
    data = EDM.sampleData["circle"]
    kwargs = SMapArgs.copy()
    kwargs.update(
        dict(
            columns="x", target="y", lib=[1, 100], pred=[110, 160], E=4, theta=2.0, Tp=0
        )
    )

    pred = smap(**transform_data(data, kwargs), **transform_args(kwargs))
    pyedm = EDM.SMap(data, **kwargs)

    smap_ = transform_result(pred)
    valid = transform_valid(pyedm["predictions"])
    assert smap_.to_numpy() == pytest.approx(valid.to_numpy(), abs=1e-6)


# ------------------------------------------------------------
def test_smap4():
    """Tp = 5 disjoint lib/pred"""
    data = EDM.sampleData["circle"]
    kwargs = SMapArgs.copy()
    kwargs.update(
        dict(
            columns="x", target="y", lib=[1, 100], pred=[110, 160], E=4, theta=2.0, Tp=5
        )
    )

    pred = smap(**transform_data(data, kwargs), **transform_args(kwargs))
    pyedm = EDM.SMap(data, **kwargs)

    smap_ = transform_result(pred)
    valid = transform_valid(pyedm["predictions"])
    assert smap_.to_numpy() == pytest.approx(valid.to_numpy(), abs=1e-6)


# ------------------------------------------------------------
def test_smap5():
    """embedded = True overlapping lib/pred"""
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
