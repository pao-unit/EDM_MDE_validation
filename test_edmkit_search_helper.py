import numpy as np
from edmkit.search import neighborhood, state, strategy
from numpy.random import default_rng
from pandas import DataFrame
from pyEDM import CCM
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

from test_edmkit_edim_helper import embed_dimension, simplex_rho

# ------------------------------------------------------------
# dimx MDE Run() is a greedy forward selection over candidate columns:
# at each dimension d every remaining candidate is scored by the cross
# map rho of [candidate] + selected (pyEDM Simplex embedded = True +
# ComputeError), candidates at or below crossMapRhoMin are discarded,
# and the rest are walked in decreasing rho order through two per-column
# cached gates -- an EmbedDimension rho threshold (embedDimRhoMin) and a
# CCM convergence slope threshold (ccmSlope) -- until one passes.
#
# The edmkit mirror expresses that loop as an edmkit search:
#   - neighborhood.forward over the candidate indices expands the
#     selected state by one column per child (dimx [c] + MDEcolumns),
#   - the Energy scores each child with -rho from the validated edmkit
#     simplex_projection mirror of the dimx worker Simplex, and runs the
#     dimx rank-order gate walk inside: rejected and sub-threshold
#     candidates are +inf, so the accepted candidate is the argmin,
#   - strategy.greedy commits to that argmin per depth; the cutoff at
#     -crossMapRhoMin discards the +inf children, so a dimension where
#     every candidate fails ends the trajectory (dimx break).
#
# The EmbedDimension gate reuses the validated edmkit embed_dimension
# mirror.  The CCM gate has no edmkit counterpart yet and calls pyEDM
# CCM exactly as dimx does.  The tests pin ccmSeed (123, as test_CCM
# does) so the CCM library sampling -- and with it the whole variable
# selection -- is reproducible.
# ------------------------------------------------------------


# ------------------------------------------------------------
def _cross_map_rho(data, cols, kwargs):
    """dimx Parallel.SimplexWorker: cross map rho of cols -> target via
    the validated edmkit simplex_projection mirror of pyEDM Simplex
    (embedded = True, E = len(cols)) + ComputeError rho."""
    return simplex_rho(
        data,
        dict(
            columns=cols,
            target=kwargs["target"],
            lib=kwargs["lib"],
            pred=kwargs["pred"],
            E=0,
            Tp=kwargs["Tp"],
            knn=0,
            tau=kwargs["tau"],
            exclusionRadius=kwargs["exclusionRadius"],
            embedded=True,
            validLib=[],
        ),
    )


# ------------------------------------------------------------
def _edim_gate(data, column, kwargs):
    """dimx Run() EmbedDimension gate for one column: (maxEDim,
    maxRhoEDim) via the validated edmkit embed_dimension mirror.
    firstEMax takes the first local rho maximum, else the last E."""
    ekwargs = dict(
        columns=column,
        target=kwargs["target"],
        maxE=kwargs["maxE"],
        lib=kwargs["lib"],
        pred=kwargs["pred"],
        Tp=kwargs["Tp"],
        tau=kwargs["tau"],
        exclusionRadius=kwargs["exclusionRadius"],
        embedded=False,
        validLib=[],
    )
    EDimDF = embed_dimension(data, ekwargs)

    if kwargs["firstEMax"]:
        iMax = argrelextrema(EDimDF["rho"].to_numpy(), np.greater)[0]
        iMax = iMax[0] if len(iMax) else len(EDimDF["E"]) - 1
    else:
        iMax = EDimDF["rho"].round(4).argmax()

    return int(EDimDF["E"].iloc[iMax]), round(float(EDimDF["rho"].iloc[iMax]), 4)


# ------------------------------------------------------------
def _ccm_slope(data, column, E, kwargs, libSizes):
    """dimx Run() CCM convergence gate for one column: linear regression
    slope of CCM target:column rho over libSizes normalized to [0, 1].
    pyEDM CCM is called directly (no validated edmkit CCM mirror yet);
    parallel = False is an execution mode only (identical rho for a
    fixed seed) that avoids a process-pool spawn per call."""
    ccmDF = CCM(
        dataFrame=data,
        columns=column,
        target=kwargs["target"],
        libSizes=libSizes,
        sample=kwargs["sample"],
        E=E,
        Tp=kwargs["Tp"],
        tau=kwargs["tau"],
        exclusionRadius=kwargs["exclusionRadius"],
        seed=kwargs["ccmSeed"],
        mpMethod=kwargs["mpMethod"],
        parallel=False,
        noTime=True,
    )
    ccmVals = ccmDF[f"{kwargs['target']}:{column}"].to_numpy()

    libSizesVec = np.asarray(libSizes, dtype=float).reshape(-1, 1)
    libSizesVec = libSizesVec / libSizesVec[-1]
    lm = LinearRegression().fit(libSizesVec, np.nan_to_num(ccmVals))
    return round(lm.coef_[0], 5)


# ------------------------------------------------------------
def edmkit_search(data, kwargs):
    """dimx MDE(data, **kwargs).Run() via edmkit search: DataFrame of
    selected variables and their cross map rho, one row per dimension."""
    # dimx MDE.Validate() removeTime and Parallel.PrepareNumericFrame()
    # noTime each drop the first column; lib / pred default to the
    # first / second half of the rows
    if kwargs["removeTime"]:
        data = data.drop(columns=data.columns[0])
    if not kwargs["noTime"]:
        data = data.drop(columns=data.columns[0])
    N = data.shape[0]

    kwargs = dict(kwargs)
    if not len(kwargs["lib"]):
        kwargs["lib"] = [1, N // 2]
    if not len(kwargs["pred"]):
        kwargs["pred"] = [N // 2 + 1, N]

    candidates = [c for c in data.columns if c not in kwargs["removeColumns"]]

    # dimx Run(): CCM libSizes from pLibSizes percentiles
    libSizes = [int(N * (p / 100)) for p in kwargs["pLibSizes"]]

    edimCache = {}  # column : (maxEDim, maxRhoEDim)
    ccmCache = {}  # column : slope

    def gate(column, rho):
        """dimx Run() per-column validation with negative caching."""
        if kwargs["noCCM"]:
            return True
        if kwargs["E"] > 0:
            maxEDim, maxRhoEDim = kwargs["E"], round(float(rho), 4)
        else:
            if column not in edimCache:
                edimCache[column] = _edim_gate(data, column, kwargs)
            maxEDim, maxRhoEDim = edimCache[column]
        if maxRhoEDim < kwargs["embedDimRhoMin"]:
            return False
        if column not in ccmCache:
            ccmCache[column] = _ccm_slope(data, column, maxEDim, kwargs, libSizes)
        return ccmCache[column] > kwargs["ccmSlope"]

    def energy(states, _contexts):
        """Cross map rho for every child, then the dimx candidate walk:
        decreasing rho order, discard rho <= crossMapRhoMin, gate until
        the first pass.  Only the accepted child keeps a finite energy
        -rho; the +inf rows mean not-picked rather than scored, so this
        energy is only meaningful under greedy (beam width 1, one beam),
        where the argmin is exactly the dimx pick."""
        # dimx candidate combination [c] + MDEcolumns: forward appends
        # the new index last, dimx puts the new column first
        newColumns = [candidates[s[-1]] for s in states]
        rho = np.array(
            [
                _cross_map_rho(
                    data, [newColumns[i]] + [candidates[j] for j in s[:-1]], kwargs
                )
                for i, s in enumerate(states)
            ]
        )

        energies = np.full(len(states), np.inf)
        for i in np.argsort(-rho, kind="stable"):
            if rho[i] <= kwargs["crossMapRhoMin"]:
                break
            if gate(newColumns[i], rho[i]):
                energies[i] = -rho[i]
                break
        return energies, np.empty((len(states), 0))

    initial = strategy.Frontier(
        states=state.initial(), contexts=np.empty((1, 0)), energies=np.zeros(1)
    )
    search = strategy.greedy(
        energy,
        neighborhood.forward(len(candidates)),
        depth=kwargs["D"],
        cutoff=-kwargs["crossMapRhoMin"],
    )
    # the rng only orders children within a parent; selection is by rho
    # rank, so the seed matters only for exact rho ties
    trajectory = list(search(initial, default_rng(0)))

    return DataFrame(
        {
            "variables": [candidates[f.states[0][-1]] for f in trajectory],
            "rho": [-f.energies[0] for f in trajectory],
        }
    )
