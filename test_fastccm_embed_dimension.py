"""Parked FastCCM EmbedDimension parity tests.

We are not pursuing pyEDM EmbedDimension reproduction in the FastCCM mirror.
These tests are kept only as historical parity targets.  The cleaner
FastCCM-native E/tau search is
`fastccm.ccm_utils.Functions.find_optimal_embedding_params`, which compares all
candidate embeddings over one unified minimum common prediction length after
embedding.  pyEDM's EmbedDimension fixtures use independent prediction lengths
for each embedding dimension, so reproducing them requires compatibility
machinery that is outside the current FastCCM focus.
"""

import pytest
from pyEDM import sampleData

from conftest import EmbedDimensionArgs, ValidData
from test_fastccm_embed_dimension_helper import embed_dimension

pytestmark = pytest.mark.skip(
    reason=(
        "Parked parity target: FastCCM mirror is no longer pursuing pyEDM "
        "EmbedDimension reproduction."
    )
)


# ------------------------------------------------------------
def test_edim1():
    """Lorenz V1 Tp=5 tau=-5"""
    data = sampleData["Lorenz5D"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(
        dict(columns="V1", target="V1", lib=[1, 1000], pred=[1, 1000], Tp=5, tau=-5)
    )

    E = embed_dimension(data, kwargs)
    Ev = ValidData("EDim_1_valid.csv")

    assert E.equals(Ev)


# ------------------------------------------------------------
def test_edim2():
    """block_3sp cross map"""
    data = sampleData["block_3sp"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(dict(columns="x_t", target="z_t", lib=[1, 198], pred=[1, 198]))

    E = embed_dimension(data, kwargs)
    Ev = ValidData("EDim_2_valid.csv")

    assert E.equals(Ev)


# ------------------------------------------------------------
def test_edim3():
    """SumFlow_1980-2005"""
    data = sampleData["SumFlow_1980-2005"]
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

    E = embed_dimension(data, kwargs)
    Ev = ValidData("EDim_3_valid.csv")

    assert E.equals(Ev)


# ------------------------------------------------------------
def test_edim4():
    """SumFlow_1980-2005 out of sample"""
    data = sampleData["SumFlow_1980-2005"]
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

    E = embed_dimension(data, kwargs)
    Ev = ValidData("EDim_4_valid.csv")

    assert E.equals(Ev)


# ------------------------------------------------------------
def test_edim5():
    """TentMap"""
    data = sampleData["TentMapNoise"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(
        dict(columns="TentMap", target="TentMap", lib=[1, 999], pred=[1, 999], tau=-3)
    )

    E = embed_dimension(data, kwargs)
    Ev = ValidData("EDim_5_valid.csv")

    assert E.equals(Ev)


# ------------------------------------------------------------
def test_edim6():
    """Lorenz V1 Tp=-5 tau=5"""
    data = sampleData["Lorenz5D"]
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update(
        dict(columns="V1", target="V1", lib=[1, 1000], pred=[1, 1000], Tp=-5, tau=5)
    )

    E = embed_dimension(data, kwargs)
    Ev = ValidData("EDim_6_valid.csv")

    assert E.equals(Ev)


# ------------------------------------------------------------
def test_edim7():
    """Lorenz V1:V4 Tp=5 tau=-5 xRad=20"""
    data = sampleData["Lorenz5D"]
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

    E = embed_dimension(data, kwargs)
    Ev = ValidData("EDim_7_valid.csv")

    assert E.equals(Ev)
