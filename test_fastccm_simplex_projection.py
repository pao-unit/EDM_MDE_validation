"""Validation tests for FastCCM simplex-equivalent predictions against pyEDM ValidOutput"""

import pyEDM as EDM

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
    data = EDM.sampleData["block_3sp"]
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
    data = EDM.sampleData["block_3sp"]
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
    data = EDM.sampleData["block_3sp"]
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
    data = EDM.sampleData["block_3sp"]
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
    data = EDM.sampleData["block_3sp"]
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
def test_simplex6():
    """negative Tp disjoint lib/pred"""
    data = EDM.sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(columns="x_t", target="y_t", lib=[1, 100], pred=[101, 195], E=3, Tp=-2)
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    pyedm = EDM.Simplex(data, **kwargs)

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(pyedm), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
def test_simplex7():
    """knn = 1 disjoint lib/pred"""
    data = EDM.sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(columns="x_t", target="y_t", lib=[1, 100], pred=[101, 195], E=3, knn=1)
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    pyedm = EDM.Simplex(data, **kwargs)

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(pyedm), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
def test_simplex8():
    """negative Tp with overlapping lib/pred"""
    data = EDM.sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(columns="x_t", target="y_t", lib=[1, 100], pred=[50, 80], E=3, Tp=-2)
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("Smplx_negTp_block_3sp_valid.csv")

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
def test_simplex9():
    """validLib with overlapping lib/pred"""
    data = EDM.sampleData["circle"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(
            columns="x",
            target="x",
            lib=[1, 200],
            pred=[1, 200],
            E=2,
            validLib=data.eval("x > 0.5 | x < -0.5"),
        )
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("Smplx_validLib_valid.csv")

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
def test_simplex10():
    """multiple lib spans with overlapping pred"""
    data = EDM.sampleData["circle"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(columns="x", target="x", lib=[1, 40, 50, 130], pred=[80, 170], E=2, tau=-3)
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("Smplx_disjointLib_valid.csv")

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smplx.equals(valid)


# ------------------------------------------------------------
def test_simplex11():
    """exclusion radius with overlapping lib/pred"""
    data = EDM.sampleData["circle"]
    kwargs = SimplexArgs.copy()
    kwargs.update(
        dict(
            columns="x", target="y", lib=[1, 100], pred=[21, 81], E=2, exclusionRadius=5
        )
    )

    pred = simplex_projection(**transform_data(data, kwargs), **transform_args(kwargs))
    dfv = ValidData("Smplx_exclRadius_valid.csv")

    smplx = round(transform_result(pred), 6)
    valid = round(transform_valid(dfv), 6)
    assert smplx.equals(valid)
