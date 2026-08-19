"""Validation tests for FastCCM CCM/simplex scores against pyEDM"""

import pytest
from numpy import linspace, roll, sin
from numpy.random import default_rng

from test_fastccm_ccm_helper import (
    convergence_summary,
    fastccm_simplex_rho,
    pyedm_ccm_x_to_y,
    pyedm_simplex_rho,
    simplex_score_curve,
    xy_dataframe,
)


# ------------------------------------------------------------
@pytest.mark.parametrize(
    ("tp", "exclusion_radius", "lib_size"),
    [
        (0, 0, 200),
        (0, 10, 200),
        (1, 0, 199),
        (1, 10, 199),
        (5, 0, 195),
        (5, 10, 195),
    ],
)
def test_simplex_score_matrix_matches_pyedm_simplex(tp, exclusion_radius, lib_size):
    """score_matrix method = simplex, compared to pyEDM Simplex rho"""
    rng = default_rng(1)
    x = rng.normal(0, 10, 200)
    y = rng.uniform(0, 10, 200)
    data = xy_dataframe(x, y)

    fastccm = fastccm_simplex_rho(
        x,
        y,
        library_size=lib_size,
        sample_size=lib_size,
        exclusion_window=exclusion_radius,
        tp=tp,
        nbrs_num=10,
        seed=1,
        batch_size=None,
        clean_after=False,
    )
    pyedm = pyedm_simplex_rho(
        data,
        lib="1 200",
        pred="1 200",
        E=3,
        tau=-1,
        Tp=tp,
        exclusionRadius=exclusion_radius,
        knn=10,
    )

    assert fastccm == pytest.approx(pyedm, abs=1e-12)


# ------------------------------------------------------------
@pytest.mark.parametrize(
    ("tp", "exclusion_radius", "lib_size"),
    [
        (0, 0, 200),
        (0, 10, 200),
        (1, 0, 199),
        (1, 10, 199),
    ],
)
def test_simplex_score_matrix_matches_pyedm_ccm(tp, exclusion_radius, lib_size):
    """score_matrix method = simplex, compared to direct pyEDM CCM X:Y"""
    rng = default_rng(1)
    x = rng.normal(0, 10, 200)
    y = rng.uniform(0, 10, 200)
    data = xy_dataframe(x, y)

    fastccm = fastccm_simplex_rho(
        x,
        y,
        library_size=lib_size,
        sample_size=lib_size,
        exclusion_window=exclusion_radius,
        tp=tp,
        nbrs_num=10,
        seed=1,
        batch_size=None,
        clean_after=False,
    )
    pyedm = pyedm_ccm_x_to_y(
        data,
        libSizes=str(lib_size),
        sample=1,
        E=3,
        tau=-1,
        Tp=tp,
        exclusionRadius=exclusion_radius,
        seed=1,
        knn=10,
    )

    assert fastccm == pytest.approx(pyedm, abs=1e-12)

# ------------------------------------------------------------
def test_simplex_score_curve_converges():
    """simplex score increases toward high rho as library size grows"""
    t = linspace(0, 8 * 3.141592653589793, 200)
    x = sin(t) + 0.1 * sin(3 * t)
    y = roll(x, -1)
    y[-1] = y[-2]

    lib_sizes = [20, 40, 80, 160]
    rho = simplex_score_curve(
        x,
        y,
        lib_sizes,
        E=3,
        tau=1,
        sample_size=None,
        exclusion_window=0,
        tp=0,
        seed=1,
        nbrs_num=4,
        batch_size=None,
        clean_after=False,
    )

    summary = convergence_summary(
        lib_sizes,
        rho,
        min_gain=0.015,
        min_final_rho=0.99,
    )
    assert summary["converged"]
