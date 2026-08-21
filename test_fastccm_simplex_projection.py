"""FastCCM Simplex mirrors for disjoint vectorized prediction."""

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


@pytest.mark.parametrize(
    ("updates", "valid_file"),
    [
        (
            dict(columns="x_t", target="x_t", lib=[1, 100], pred=[101, 195], E=3),
            "Smplx_E3_block_3sp_valid.csv",
        ),
        (
            dict(
                columns=["x_t", "y_t", "z_t"],
                target="x_t",
                lib=[1, 99],
                pred=[100, 198],
                E=3,
                embedded=True,
            ),
            "Smplx_E3_embd_block_3sp_valid.csv",
        ),
        (
            dict(
                columns="x_t",
                target="y_t",
                lib=[1, 100],
                pred=[101, 198],
                E=3,
                Tp=5,
                tau=3,
            ),
            "Smplx_posTau_block_3sp_valid.csv",
        ),
        (
            dict(
                columns="x_t",
                target="y_t",
                lib=[1, 100],
                pred=[101, 198],
                E=3,
                Tp=-4,
                tau=3,
            ),
            "Smplx_negTp_posTau_block_3sp_valid.csv",
        ),
        (
            dict(
                columns="x_t y_t z_t",
                target="x_t",
                lib=[1, 99],
                pred=[100, 198],
                E=3,
                embedded=True,
            ),
            "Smplx_E3_embd_block_3sp_valid.csv",
        ),
    ],
)
def test_simplex_projection(updates, valid_file):
    kwargs = SimplexArgs | updates
    pred = simplex_projection(
        **transform_data(sampleData["block_3sp"], kwargs),
        **transform_args(kwargs),
    )

    assert round(transform_result(pred), 6).equals(
        round(transform_valid(ValidData(valid_file)), 6)
    )
