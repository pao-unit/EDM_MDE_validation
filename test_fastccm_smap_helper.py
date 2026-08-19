from fastccm import PairwiseCCM
from numpy import asarray, isnan

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

    pyEDM's Tp conventions are applied here after lib rows are selected,
    so FastCCM receives a pre-aligned target vector with tp=0.
    """
    emb, lib_i, pred_i, libOverlap = _indices(data, kwargs)
    target = data[_names(kwargs["target"])[0]].to_numpy(dtype=float)
    lib_target_i = lib_i + kwargs["Tp"]

    if not (libOverlap or kwargs["exclusionRadius"] > 0):
        return dict(
            X_lib_emb=[emb[lib_i]],
            Y_lib_emb=[target[lib_target_i, None]],
            X_pred_emb=[emb[pred_i]],
            library_size=len(lib_i),
        )

    batches = []
    for pred_row in pred_i:
        keep = ~isnan(target[lib_target_i])
        if kwargs["exclusionRadius"] > 0:
            keep &= abs(lib_i - pred_row) > kwargs["exclusionRadius"]
        elif libOverlap:
            keep &= lib_i != pred_row

        batches.append(
            dict(
                X_lib_emb=[emb[lib_i[keep]]],
                Y_lib_emb=[target[lib_target_i[keep], None]],
                X_pred_emb=[emb[pred_row : pred_row + 1]],
                library_size=keep.sum(),
            )
        )
    return dict(_batches=batches)


# ------------------------------------------------------------
def smap(**kwargs):
    """Run FastCCM's SMap-equivalent prediction path in double precision."""
    ccm = PairwiseCCM(device="cpu", dtype="float64", compute_dtype="float64")
    batches = kwargs.pop("_batches", None)
    if batches is None:
        return ccm.predict_matrix(**kwargs)

    predictions = []
    for batch in batches:
        pred = ccm.predict_matrix(**batch, **kwargs)
        predictions.append(pred[0, 0, 0, 0])
    return asarray(predictions, dtype=float)[:, None, None, None]
