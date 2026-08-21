from numpy import array_equal
from fastccm import PairwiseCCM

from test_edmkit_simplex_projection_helper import _indices, _names


def _self_prediction(lib_i, pred_i, libOverlap):
    """True for the "predict on library" overlap: pred_i extends lib_i as the
    same contiguous run. FastCCM's exclusion_window excludes neighbors by
    array *position*, not original time index, so it only reproduces pyEDM's
    exclusionRadius when position implies the same row in both arrays --
    true here, not for a general partial overlap.
    """
    n = len(lib_i)
    return libOverlap and array_equal(lib_i, pred_i[:n])


def transform_args(data, kwargs):
    _, lib_i, pred_i, libOverlap = _indices(data, kwargs)

    if libOverlap and not _self_prediction(lib_i, pred_i, libOverlap):
        raise NotImplementedError(
            "FastCCM mirror only covers disjoint lib/pred or lib == pred "
            "self-prediction; partial overlap needs per-query masking."
        )

    return dict(
        method="smap",
        theta=kwargs["theta"],
        ridge=0.0,
        tp=0,
        exclusion_window=kwargs["exclusionRadius"] if libOverlap else None,
        batch_size=None,
        clean_after=False,
    )


def transform_data(data, kwargs):
    emb, lib_i, pred_i, libOverlap = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)
    lib_target_i = lib_i + kwargs["Tp"]

    if libOverlap and not _self_prediction(lib_i, pred_i, libOverlap):
        raise NotImplementedError(
            "FastCCM mirror only covers disjoint lib/pred or lib == pred "
            "self-prediction; partial overlap needs per-query masking."
        )

    return dict(
        X_lib_emb=[emb[lib_i]],
        Y_lib_emb=[target[lib_target_i, None]],
        X_pred_emb=[emb[pred_i]],
        library_size=len(lib_i),
    )


def smap(**kwargs):
    ccm = PairwiseCCM(device="cpu", dtype="float64", compute_dtype="float64")
    return ccm.predict_matrix(**kwargs)
