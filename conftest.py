"""
conftest.py — shared resources for pyEDM pytest suite.

pytest loads this file automatically before any test collection.
Everything defined here is available to all test files in this
directory without an explicit import.

Contents:
  - GetMP_ContextName()  multiprocessing context helper
  - ValidData()          load a validation CSV by filename
  - *Args dicts          default keyword arguments for each EDM API function
"""

import os
from   multiprocessing import get_context, get_start_method

from pandas import read_csv

# ---------------------------------------------------------------------------
# Multiprocessing context helper  (remove when > Python 3.13)
# ---------------------------------------------------------------------------

def GetMP_ContextName():
    '''Until > Python 3.14, disallow "fork" multiprocessing context.'''
    allowedContext = ("forkserver", "spawn")
    current = get_start_method( allow_none = True )
    if current in allowedContext:
        return get_context( current )._name
    for method in allowedContext:
        try:
            return get_context( method )._name
        except ValueError:
            continue

# ---------------------------------------------------------------------------
# Validation file helper
# ---------------------------------------------------------------------------

VALID_DIR = os.path.join( os.path.dirname(os.path.abspath(__file__)),
                          "ValidOutput" )

def ValidData( filename ):
    '''Return DataFrame of validation CSV from the validation/ directory.'''
    return read_csv( os.path.join( VALID_DIR, filename ) )
    
# ---------------------------------------------------------------------------
# MDE data file helper
# ---------------------------------------------------------------------------

def MDE_FlyData():
    '''Get dimx Fly data'''
    try:
        import dimx as dx
    except ImportError:
        raise( "MDE_FlyData(): MDE package dimx not imported" )

    file_path = os.path.dirname(os.path.abspath(dx.__file__))
    return read_csv( file_path + "/data/Fly80XY_norm_1061.csv" )

# ---------------------------------------------------------------------------
# Default argument dictionaries — one per API function.
#
# Every parameter is listed. Parameters not actively tested carry a comment.
# Tests copy the relevant dict and update only the parameters under test
# ---------------------------------------------------------------------------

SimplexArgs = dict( columns         = "",
                    target          = "",
                    lib             = "",
                    pred            = "",
                    E               = 0,
                    Tp              = 1,
                    knn             = 0,
                    tau             = -1,
                    exclusionRadius = 0,
                    embedded        = False,
                    validLib        = [],
                    noTime          = False,    # not tested individually
                    generateSteps   = 0,        # tested in test_generate.py
                    generateConcat  = False,    # tested in test_generate.py
                    verbose         = False,    # not tested
                    showPlot        = False,    # not tested
                    ignoreNan       = True,
                    returnObject    = False )

SMapArgs = dict( columns         = "",
                 target          = "",
                 lib             = "",
                 pred            = "",
                 E               = 0,
                 Tp              = 1,
                 knn             = 0,
                 tau             = -1,
                 theta           = 0,
                 exclusionRadius = 0,
                 solver          = None,        # not tested
                 embedded        = False,
                 validLib        = [],
                 noTime          = False,       # tested in test_smap.py
                 generateSteps   = 0,           # tested in test_generate.py
                 generateConcat  = False,       # tested in test_generate.py
                 ignoreNan       = True,
                 showPlot        = False,       # not tested
                 verbose         = False,       # not tested
                 returnObject    = False )

CCMArgs = dict( columns         = "",
                target          = "",
                libSizes        = "",
                sample          = 30,
                E               = 0,
                Tp              = 0,
                knn             = 0,
                tau             = -1,
                exclusionRadius = 0,
                seed            = None,
                embedded        = False,
                validLib        = [],
                includeData     = False,        # not tested
                noTime          = False,        # not tested
                mpMethod        = None,         # not tested
                parallel        = True,         # not tested
                sharedMB        = 0.01,         # not tested
                verbose         = False,        # not tested
                showPlot        = False,        # not tested
                returnObject    = False,
                legacy          = False )       # not tested

EmbedDimensionArgs = dict( columns         = "",
                           target          = "",
                           maxE            = 10,
                           lib             = "",
                           pred            = "",
                           Tp              = 1,
                           tau             = -1,
                           exclusionRadius = 0,
                           embedded        = False,  # not tested
                           validLib        = [],     # not tested
                           noTime          = False,  # not tested
                           ignoreNan       = True,   # not tested
                           verbose         = False,  # not tested
                           numProcess      = 4,
                           mpMethod        = GetMP_ContextName(),
                           chunksize       = 1,      # not tested
                           showPlot        = False )

PredictNonlinearArgs = dict( columns         = "",
                             target          = "",
                             theta           = None,
                             lib             = "",
                             pred            = "",
                             E               = 1,
                             Tp              = 1,
                             knn             = 0,
                             tau             = -1,
                             exclusionRadius = 0,
                             solver          = None,   # not tested
                             embedded        = False,  # not tested
                             validLib        = [],     # not tested
                             noTime          = False,  # not tested
                             ignoreNan       = True,   # not tested
                             verbose         = False,  # not tested
                             numProcess      = 4,
                             mpMethod        = GetMP_ContextName(),
                             chunksize       = 1,      # not tested
                             showPlot        = False )

MDEArgs = dict( dataFile        = None,  # file name for DataFrame
                dataName        = None,  # dataName in npz archive
                removeTime      = False, # remove dataFrame first column
                noTime          = False, # first dataFrame column is data
                columnNames     = [],    # partial match columnNames
                initDataColumns = [],    # .npy .npz : see ReadData()
                removeColumns   = [],    # columns to remove from dataFrame
                D               = 3,     # MDE max dimension
                target          = None,  # target variable to predict
                lib             = [],    # EDM library start,stop 1-offset
                pred            = [],    # EDM prediction start,stop 1-offset
                Tp              = 1,     # prediction interval
                tau             = -1,    # CCM embedding delay
                exclusionRadius = 0,     # exclusion radius: CCM, CrossMap
                sample          = 20,    # CCM random sample
                pLibSizes       = [10, 15, 85, 90], # CCM libSizes percentiles
                noCCM           = False, # Do not validate with CCM
                ccmSlope        = 0.01,  # CCM convergence criteria
                ccmSeed         = None,  # CCM random seed
                E               = 0,     # Static E for all CCM
                crossMapRhoMin  = 0.5,   # threshold for L_rhoD in Run()
                embedDimRhoMin  = 0.5,   # maxRhoEDim threshold in Run()
                maxE            = 15,    # maximum embedding dim for CCM
                firstEMax       = False, # use first local peak for E-dim
                timeDelay       = 0,     # Number of time delays to add
                cores           = 5,     # Number of cores for CrossMapColumns
                mpMethod        = GetMP_ContextName(), # multiprocessing context
                chunksize       = 1,     # multiprocessing chunksize
                outDir          = './',  # use pathlib for windog
                outFile         = None,
                outCSV          = None,
                logFile         = None,
                consoleOut      = True,  # LogMsg() print() to console
                verbose         = False,
                debug           = False,
                plot            = False,
                title           = None,
                args            = None )
