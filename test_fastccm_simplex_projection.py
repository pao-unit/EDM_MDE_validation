"""Validation tests for FastCCM simplex-equivalent predictions against pyEDM ValidOutput"""

import pytest
from pyEDM import sampleData

from conftest import SimplexArgs, ValidData
from test_fastccm_simplex_projection_helper import (
    simplex_projection,
    transform_args,
    transform_data,
    transform_result,
    transform_valid,
)


# ------------------------------------------------------------
def test_simplex1():
    """embedded = False"""
    data = sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update(dict(columns="x_t", target="x_t", lib=[1, 100], pred=[101, 195], E=3))

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("Smplx_E3_block_3sp_valid.csv")

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
def test_simplex2():
    """embedded = True"""
    data = sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(
            columns=["x_t", "y_t", "z_t"],
            target="x_t",
            lib=[1, 99],
            pred=[100, 198],
            E=3,
            embedded=True,
        )
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("Smplx_E3_embd_block_3sp_valid.csv")

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
def test_simplex3():
    """positive tau"""
    data = sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(
            columns="x_t", target="y_t", lib=[1, 100], pred=[101, 198], E=3, Tp=5, tau=3
        )
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("Smplx_posTau_block_3sp_valid.csv")

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
def test_simplex4():
    """positive tau negative Tp"""
    data = sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(
            columns="x_t",
            target="y_t",
            lib=[1, 100],
            pred=[101, 198],
            E=3,
            Tp=-4,
            tau=3,
        )
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("Smplx_negTp_posTau_block_3sp_valid.csv")

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
def test_simplex5():
    """embedded = True columns string"""
    data = sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(
            columns="x_t y_t z_t",
            target="x_t",
            lib=[1, 99],
            pred=[100, 198],
            E=3,
            embedded=True,
        )
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("Smplx_E3_embd_block_3sp_valid.csv")

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
@pytest.mark.skip(
    reason=(
        "Parity-only mirror: overlapping lib/pred with negative Tp currently "
        "uses per-query FastCCM calls instead of vectorized prediction."
    )
)
def test_simplex8():
    """Future parity target.

    Baseline case: block_3sp, columns="x_t", target="y_t",
    lib=[1, 100], pred=[50, 80], E=3, Tp=-2.
    """


# ------------------------------------------------------------
@pytest.mark.skip(
    reason=(
        "Parity-only mirror: validLib with overlapping lib/pred currently "
        "uses per-query FastCCM calls instead of vectorized prediction."
    )
)
def test_simplex9():
    """Future parity target.

    Baseline case: circle, columns="x", target="x", lib=[1, 200],
    pred=[1, 200], E=2, validLib=(x > 0.5 | x < -0.5).
    """


# ------------------------------------------------------------
@pytest.mark.skip(
    reason=(
        "Parity-only mirror: overlapping multiple lib spans currently use "
        "per-query FastCCM calls instead of vectorized prediction."
    )
)
def test_simplex10():
    """Future parity target.

    Baseline case: circle, columns="x", target="x",
    lib=[1, 40, 50, 130], pred=[80, 170], E=2, tau=-3.
    """


# ------------------------------------------------------------
@pytest.mark.skip(
    reason=(
        "Parity-only mirror: exclusionRadius parity currently uses per-query "
        "FastCCM calls instead of vectorized prediction."
    )
)
def test_simplex11():
    """Future parity target.

    Baseline case: circle, columns="x", target="y",
    lib=[1, 100], pred=[21, 81], E=2, exclusionRadius=5.
    """
