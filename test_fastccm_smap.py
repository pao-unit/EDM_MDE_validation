"""Validation tests for FastCCM SMap-equivalent predictions against pyEDM ValidOutput"""

import pytest
from pyEDM import sampleData

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
    data = sampleData["circle"]
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
@pytest.mark.skip(
    reason=(
        "Parity-only mirror: matching pyEDM's per-query SMap neighbor mask and "
        "coefficient table currently sacrifices FastCCM vectorization."
    )
)
def test_smap2():
    """Future parity target.

    Baseline case: circle, columns=["x", "y"], target="x",
    lib=[1, 200], pred=[1, 200], theta=3.0, embedded=True.
    Also checks mean coefficients dx/dx=0.99801 and dx/dy=0.06311.
    """


# ------------------------------------------------------------
@pytest.mark.skip(
    reason=(
        "Parity-only mirror: full Lorenz5D coefficient parity currently needs "
        "query-specific SMap libraries instead of a vectorized FastCCM call."
    )
)
def test_smap4():
    """Future parity target.

    Baseline case: Lorenz5D, columns=["V1", "V2", "V3"], target="V5",
    lib=[1, 300], pred=[501, 600], Tp=5, theta=3.0, embedded=True.
    Also checks full coefficient table against SMap_Lorenz5D_coef_valid.csv.
    """
