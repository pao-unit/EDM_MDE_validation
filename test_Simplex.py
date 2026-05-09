
'''Validation tests for pyEDM Simplex'''

import pytest
from   numpy  import nan, array, array_equal
from   pandas import DataFrame

try:
    import pyEDM as EDM
except ImportError:
    raise( "test_Simplex(): pyEDM package not imported" )

from conftest import SimplexArgs, ValidData

#------------------------------------------------------------
def test_simplex1():
    '''embedded = False'''
    data = EDM.sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns = "x_t",
                        target  = "x_t",
                        lib     = [1, 100],
                        pred    = [101, 195],
                        E       = 3) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_E3_block_3sp_valid.csv")

    smplx = round( df.get ('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex2():
    '''embedded = True'''
    data = EDM.sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns  = ["x_t","y_t","z_t"],
                        target   = "x_t",
                        lib      = [1, 99],
                        pred     = [100, 198],
                        E        = 3,
                        embedded = True) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_E3_embd_block_3sp_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex3():
    '''embedded = True columns string'''
    data = EDM.sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns  = "x_t y_t z_t",
                        target   = "x_t",
                        lib      = [1, 99],
                        pred     = [100, 198],
                        E        = 3,
                        embedded = True) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_E3_embd_block_3sp_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex4():
    '''negative Tp'''
    data = EDM.sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns = "x_t",
                        target  = "y_t",
                        lib     = [1, 100],
                        pred    = [50,80],
                        E       = 3,
                        Tp      = -2) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_negTp_block_3sp_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex5():
    '''validLib'''
    data = EDM.sampleData["circle"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns  = 'x',
                        target   = 'x',
                        lib      = [1,200],
                        pred     = [1,200],
                        E        = 2,
                        validLib = data.eval('x > 0.5 | x < -0.5')) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_validLib_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex6():
    '''disjoint lib'''
    data = EDM.sampleData["circle"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns = 'x',
                        target  = 'x',
                        lib     = [1,40, 50,130],
                        pred    = [80,170],
                        E       = 2,
                        tau     = -3) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_disjointLib_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex7():
    '''disjoint pred w/ nan'''
    data = EDM.sampleData["Lorenz5D"]
    data.iloc[ [8,50,501], [1,2] ] = nan

    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns = 'V1',
                        target  = 'V2',
                        lib     = [1,50, 101,200, 251,500],
                        pred    = [1,10,151,155,551,555,881,885,991,1000],
                        E       = 5,
                        Tp      = 2) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_disjointPred_nan_valid.csv")

    smplx = round(  df.get('Predictions'), 5 )
    valid = round( dfv.get('Predictions'), 5 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex8():
    '''exclusion radius'''
    data = EDM.sampleData["circle"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns         = 'x',
                        target          = 'y',
                        lib             = [1,100],
                        pred            = [21,81],
                        E               = 2,
                        exclusionRadius = 5) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_exclRadius_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex9():
    '''nan'''
    data = EDM.sampleData["circle"]
    dfn = data.copy()
    dfn.iloc[ [5,6,12], 1 ] = nan
    dfn.iloc[ [10,11,17], 2 ] = nan

    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns = 'x',
                        target  = 'y',
                        lib     = [1,100],
                        pred    = [1,95],
                        E       = 2) )

    df  = EDM.Simplex(dfn, **kwargs)
    dfv = ValidData("Smplx_nan_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex10():
    '''nan'''
    data = EDM.sampleData["circle"]
    dfn = data.copy()
    dfn.iloc[ [5,6,12], 1 ] = nan
    dfn.iloc[ [10,11,17], 2 ] = nan

    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns = 'y',
                        target  = 'x',
                        lib     = [1,200],
                        pred    = [1,195],
                        E       = 2) )

    df  = EDM.Simplex(dfn, **kwargs)
    dfv = ValidData("Smplx_nan2_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex11():
    '''knn compare:  knn = 1 embedded = True'''
    data = EDM.sampleData["Lorenz5D"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns      = 'V5',
                        target       = 'V5',
                        lib          = [301,400],
                        pred         = [350,355],
                        knn          = 1,
                        embedded     = True,
                        returnObject = True) )

    df = EDM.Simplex(data, **kwargs)

    knn = df.knn_neighbors
    knnValid = array( [322,334,362,387,356,355] )[:,None]
    assert array_equal( knn, knnValid )

#------------------------------------------------------------
def test_simplex12():
    '''knn compare:  exclusion Radius'''
    data = EDM.sampleData["Lorenz5D"]
    x   = [i+1 for i in range(1000)]
    data = DataFrame({'Time':data['Time'], 'X':x, 'V1':data['V1']})

    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns         = 'X',
                        target          = 'V1',
                        lib             = [1,100],
                        pred            = [101,110],
                        E               = 5,
                        exclusionRadius = 10,
                        returnObject    = True) )

    df = EDM.Simplex(data, **kwargs)

    knn = df.knn_neighbors[:,0]
    knnValid = array( [89, 90, 91, 92, 93, 94, 95, 96, 97, 98] )
    assert array_equal( knn, knnValid )
    
#------------------------------------------------------------
def test_simplex13():
    '''positive tau'''
    data = EDM.sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns = "x_t",
                        target  = "y_t",
                        lib     = [1, 100],
                        pred    = [101,198],
                        E       = 3,
                        Tp      = 5,
                        tau     = 3) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_posTau_block_3sp_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )

#------------------------------------------------------------
def test_simplex14():
    '''positive tau negative Tp'''
    data = EDM.sampleData["block_3sp"]
    kwargs = SimplexArgs.copy()
    kwargs.update( dict(columns = "x_t",
                        target  = "y_t",
                        lib     = [1, 100],
                        pred    = [101,198],
                        E       = 3,
                        Tp      = -4,
                        tau     = 3 ) )

    df  = EDM.Simplex(data, **kwargs)
    dfv = ValidData("Smplx_negTp_posTau_block_3sp_valid.csv")

    smplx = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smplx.equals( valid )
