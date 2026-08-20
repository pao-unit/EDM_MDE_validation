"""FastCCM-backed MDE validation tests.

These tests keep MDE's pyEDM EmbedDimension gate, then exercise FastCCM for the
vectorizable cross-map and CCM scoring work.
"""

import pytest
from pandas import DataFrame
from pyEDM import sampleData

from conftest import MDE_FlyData, MDEArgs, ValidData
from test_fastccm_mde_helper import fastccm_mde


# ------------------------------------------------------------
@pytest.mark.skip(
    reason=(
        "Future parity target: exclusionRadius=10 forces query-specific "
        "neighbor masking, so it is not a full-vectorized FastCCM MDE case."
    )
)
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
    # Expected output follows pao-unit/MDE fastccm branch semantics.
    valid = DataFrame(
        {
            "variables": ["TS9", "TS35", "TS62", "TS39", "TS52", "TS44", "TS40"],
            "rho": [
                0.682302,
                0.815133,
                0.874745,
                0.890606,
                0.889129,
                0.896953,
                0.905181,
            ],
        }
    )

    assert mde["variables"].equals(valid["variables"])
    assert round(mde.iloc[:, 1:], 6).equals(round(valid.iloc[:, 1:], 6))
