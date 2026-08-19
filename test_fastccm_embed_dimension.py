"""Validation tests for FastCCM EmbedDimension-equivalent rho against pyEDM"""

import pyEDM as EDM
import pytest

from conftest import EmbedDimensionArgs
from test_fastccm_embed_dimension_helper import embed_dimension, pyedm_embed_dimension

# FastCCM and pyEDM can choose different equal-distance neighbors for E=1 on
# rounded sample data.  The remaining E values are typically identical at 6 dp.
EDIM_RHO_ABS = 6e-3


# ------------------------------------------------------------
def test_edim1():
    """Lorenz V1 Tp=5 tau=-5"""
    data = EDM.sampleData["Lorenz5D"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(
        dict(columns="V1", target="V1", lib=[1, 1000], pred=[1, 1000], Tp=5, tau=-5)
    )

    fastccm = embed_dimension(data, kwargs)
    pyedm = pyedm_embed_dimension(data, kwargs)

    assert fastccm["E"].equals(pyedm["E"])
    assert fastccm["rho"].to_numpy() == pytest.approx(
        pyedm["rho"].to_numpy(), abs=EDIM_RHO_ABS
    )


# ------------------------------------------------------------
def test_edim2():
    """block_3sp cross map"""
    data = EDM.sampleData["block_3sp"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(dict(columns="x_t", target="z_t", lib=[1, 198], pred=[1, 198]))

    fastccm = embed_dimension(data, kwargs)
    pyedm = pyedm_embed_dimension(data, kwargs)

    assert fastccm["E"].equals(pyedm["E"])
    assert fastccm["rho"].to_numpy() == pytest.approx(
        pyedm["rho"].to_numpy(), abs=EDIM_RHO_ABS
    )


# ------------------------------------------------------------
def test_edim3():
    """SumFlow_1980-2005"""
    data = EDM.sampleData["SumFlow_1980-2005"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(
        dict(
            columns="S12.C.D.S333",
            target="S12.C.D.S333",
            lib=[1, 1379],
            pred=[1, 1379],
            exclusionRadius=5,
        )
    )

    fastccm = embed_dimension(data, kwargs)
    pyedm = pyedm_embed_dimension(data, kwargs)

    assert fastccm["E"].equals(pyedm["E"])
    assert fastccm["rho"].to_numpy() == pytest.approx(
        pyedm["rho"].to_numpy(), abs=EDIM_RHO_ABS
    )


# ------------------------------------------------------------
def test_edim4():
    """SumFlow_1980-2005 out of sample"""
    data = EDM.sampleData["SumFlow_1980-2005"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(
        dict(
            columns="S12.C.D.S333",
            target="S12.C.D.S333",
            lib=[1, 800],
            pred=[801, 1379],
            exclusionRadius=5,
        )
    )

    fastccm = embed_dimension(data, kwargs)
    pyedm = pyedm_embed_dimension(data, kwargs)

    assert fastccm["E"].equals(pyedm["E"])
    assert fastccm["rho"].to_numpy() == pytest.approx(
        pyedm["rho"].to_numpy(), abs=EDIM_RHO_ABS
    )


# ------------------------------------------------------------
def test_edim5():
    """TentMap"""
    data = EDM.sampleData["TentMapNoise"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(
        dict(columns="TentMap", target="TentMap", lib=[1, 999], pred=[1, 999], tau=-3)
    )

    fastccm = embed_dimension(data, kwargs)
    pyedm = pyedm_embed_dimension(data, kwargs)

    assert fastccm["E"].equals(pyedm["E"])
    assert fastccm["rho"].to_numpy() == pytest.approx(
        pyedm["rho"].to_numpy(), abs=EDIM_RHO_ABS
    )


# ------------------------------------------------------------
def test_edim6():
    """Lorenz V1 Tp=-5 tau=5"""
    data = EDM.sampleData["Lorenz5D"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(
        dict(columns="V1", target="V1", lib=[1, 1000], pred=[1, 1000], Tp=-5, tau=5)
    )

    fastccm = embed_dimension(data, kwargs)
    pyedm = pyedm_embed_dimension(data, kwargs)

    assert fastccm["E"].equals(pyedm["E"])
    assert fastccm["rho"].to_numpy() == pytest.approx(
        pyedm["rho"].to_numpy(), abs=EDIM_RHO_ABS
    )


# ------------------------------------------------------------
def test_edim7():
    """Lorenz V1:V4 Tp=5 tau=-5 xRad=20"""
    data = EDM.sampleData["Lorenz5D"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(
        dict(
            columns="V1",
            target="V4",
            lib=[1, 1000],
            pred=[1, 1000],
            Tp=5,
            tau=-5,
            exclusionRadius=20,
        )
    )

    fastccm = embed_dimension(data, kwargs)
    pyedm = pyedm_embed_dimension(data, kwargs)

    assert fastccm["E"].equals(pyedm["E"])
    assert fastccm["rho"].to_numpy() == pytest.approx(
        pyedm["rho"].to_numpy(), abs=EDIM_RHO_ABS
    )
