"""Parked FastCCM-backed MDE parity tests.

The current MDE mirror depends on the parked EmbedDimension compatibility gate
when selecting variables.  Since we are no longer pursuing pyEDM
EmbedDimension reproduction in FastCCM, these downstream MDE parity tests are
kept only as historical targets.
"""

import pytest
from pyEDM import sampleData

from conftest import MDE_FlyData, MDEArgs, ValidData
from test_fastccm_mde_helper import fastccm_mde

pytestmark = pytest.mark.skip(
    reason=(
        "Parked downstream parity target: current FastCCM MDE mirror depends "
        "on pyEDM EmbedDimension reproduction."
    )
)


# ------------------------------------------------------------
def test_mde1():
    """MDE on pyEDM Lorenz5D sample data"""
    data = sampleData["Lorenz5D"]
    kwargs = MDEArgs.copy()
    kwargs.update(
        dict(
            removeTime=True,
            noTime=True,
            removeColumns=["V5"],
            D=4,
            target="V5",
            tau=-5,
            exclusionRadius=10,
            crossMapRhoMin=0.2,
            embedDimRhoMin=0.2,
            firstEMax=True,
            ccmSeed=123,
        )
    )

    mde = fastccm_mde(data, kwargs)
    valid = ValidData("MDE_test1_valid.csv")

    assert round(mde.iloc[:, 1:], 6).equals(round(valid.iloc[:, 1:], 6))


# ------------------------------------------------------------
def test_mde2():
    """MDE on dimx Fly data"""
    data = MDE_FlyData()
    kwargs = MDEArgs.copy()
    kwargs.update(
        dict(
            removeTime=True,
            noTime=True,
            removeColumns=["FWD", "Left_Right"],
            D=7,
            target="FWD",
            tau=-5,
            crossMapRhoMin=0.2,
            embedDimRhoMin=0.2,
            firstEMax=True,
            ccmSeed=123,
        )
    )

    mde = fastccm_mde(data, kwargs)
    valid = ValidData("MDE_test2_valid.csv")

    assert round(mde.iloc[:, 1:], 6).equals(round(valid.iloc[:, 1:], 6))
