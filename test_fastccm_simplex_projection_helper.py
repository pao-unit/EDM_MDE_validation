from fastccm import PairwiseCCM
from numpy import asarray
from pandas import Series

from test_edmkit_simplex_projection_helper import _dimension, _indices, _names


def transform_args(kwargs):
    knn = kwargs["knn"]
    return dict(
        method="simplex",
        nbrs_num=knn if knn > 0 else _dimension(kwargs) + 1,
        tp=0,
        exclusion_window=None,
        seed=1,
        batch_size=None,
        target_batch_size=None,
        clean_after=False,
    )


def transform_data(data, kwargs):
    emb, lib_i, pred_i, libOverlap = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)
    lib_target_i = lib_i + kwargs["Tp"]

    if libOverlap or kwargs["exclusionRadius"] > 0:
        raise NotImplementedError(
            "FastCCM mirror only covers disjoint lib/pred with no exclusion radius."
        )

    return dict(
        X_lib_emb=[emb[lib_i]],
        Y_lib_emb=[target[lib_target_i, None]],
        X_pred_emb=[emb[pred_i]],
        library_size=len(lib_i),
    )


def transform_result(predictions):
    pred = Series(asarray(predictions)[:, 0, 0, 0], name="Predictions")
    return pred.dropna().reset_index(drop=True)


def transform_valid(dfv):
    return dfv.get("Predictions").dropna().reset_index(drop=True)


def simplex_projection(**kwargs):
    ccm = PairwiseCCM(device="cpu", dtype="float64", compute_dtype="float64")
    return ccm.predict_matrix(**kwargs)
