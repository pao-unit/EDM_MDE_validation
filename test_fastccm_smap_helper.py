from fastccm import PairwiseCCM

from test_edmkit_simplex_projection_helper import _indices, _names


# ------------------------------------------------------------
def transform_args(kwargs):
    """pyEDM SMap kwargs -> FastCCM predict_matrix kwargs."""
    return dict(
        method="smap",
        theta=kwargs["theta"],
        ridge=0.0,
        tp=0,
        exclusion_window=None,
        batch_size=None,
        clean_after=False,
    )


# ------------------------------------------------------------
def transform_data(data, kwargs):
    """DataFrame + pyEDM SMap kwargs -> FastCCM predict_matrix data args.

    Only the disjoint lib/pred, no-exclusion case is represented here because
    it maps cleanly to one vectorized FastCCM predict_matrix call.
    """
    emb, lib_i, pred_i, libOverlap = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)
    lib_target_i = lib_i + kwargs["Tp"]

    if libOverlap or kwargs["exclusionRadius"] > 0:
        raise NotImplementedError(
            "Future parity target: pyEDM SMap coefficient/neighbor parity "
            "needs vectorized query-specific masking before this mirror is active."
        )

    return dict(
        X_lib_emb=[emb[lib_i]],
        Y_lib_emb=[target[lib_target_i, None]],
        X_pred_emb=[emb[pred_i]],
        library_size=len(lib_i),
    )


# ------------------------------------------------------------
def smap(**kwargs):
    """Run FastCCM's SMap-equivalent prediction path in double precision."""
    ccm = PairwiseCCM(device="cpu", dtype="float64", compute_dtype="float64")
    return ccm.predict_matrix(**kwargs)
