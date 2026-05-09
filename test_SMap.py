
'''Validation tests for pyEDM SMap'''
import pytest
from   numpy import nan
from   pandas import DataFrame

try:
    import pyEDM as EDM
except ImportError:
    raise( "test_SMap(): pyEDM package not imported" )

from conftest import SMapArgs, ValidData

#------------------------------------------------------------
def test_smap1():
    '''embedded = False'''
    data = EDM.sampleData["circle"]
    kwargs = SMapArgs.copy()
    testArgs = dict(columns = 'x',
                    target  = 'x',
                    lib     = [1,100],
                    pred    = [110,160],
                    E       = 4,
                    theta   = 3.)
    kwargs.update(testArgs)
        
    S = EDM.SMap(data, **kwargs)

    df  = S['predictions']
    dfv = ValidData("SMap_circle_E4_valid.csv")

    smap  = round( df.get ('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smap.equals( valid )

#------------------------------------------------------------
def test_smap2():
    '''SMap embedded = True coefficients'''
    data = EDM.sampleData["circle"]
    kwargs = SMapArgs.copy()
    testArgs = dict(columns  = ['x', 'y'],
                    target   = 'x',
                    lib      = [1,200],
                    pred     = [1,200],
                    theta    = 3.,
                    embedded = True, )
    kwargs.update(testArgs)

    S = EDM.SMap(data, **kwargs)

    df  = S['predictions']
    dfc = S['coefficients']
    dfv = ValidData("SMap_circle_E2_embd_valid.csv")

    smap  = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smap.equals( valid )

    assert dfc['∂x/∂x'].mean().round(5) == 0.99801
    assert dfc['∂x/∂y'].mean().round(5) == 0.06311

#------------------------------------------------------------
def test_smap3():
    '''SMap nan'''
    data = EDM.sampleData["circle"]
    dfn = data.copy()
    dfn.iloc[ [5,6,12], 1 ] = nan
    dfn.iloc[ [10,11,17], 2 ] = nan
    
    kwargs = SMapArgs.copy()
    testArgs = dict(columns = 'x',
                    target  = 'y',
                    lib     = [1,50],
                    pred    = [1,50],
                    E       = 2,
                    theta   = 3.)
    kwargs.update(testArgs)

    S = EDM.SMap(dfn, **kwargs)

    df  = S['predictions']
    dfv = ValidData("SMap_nan_valid.csv")

    smap  = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smap.equals( valid )

#------------------------------------------------------------
def test_smap4():
    '''SMap embedded = True coefficients'''
    data = EDM.sampleData["Lorenz5D"]
    kwargs = SMapArgs.copy()
    testArgs = dict(columns  = ['V1', 'V2', 'V3'],
                    target   = 'V5',
                    lib      = [1,300],
                    pred     = [501,600],
                    Tp       = 5,
                    theta    = 3.,
                    embedded = True)
    kwargs.update(testArgs)

    S = EDM.SMap(data, **kwargs)

    df   = S['predictions']
    dfc  = S['coefficients']
    dfv  = ValidData("SMap_Lorenz5D_pred_valid.csv")
    dfcv = ValidData("SMap_Lorenz5D_coef_valid.csv")

    smap  = round(  df.get('Predictions'), 6 )
    valid = round( dfv.get('Predictions'), 6 )
    assert smap.equals( valid )

    assert dfc.to_numpy() == pytest.approx( dfcv.to_numpy(), 1E-6,
                                            nan_ok=True )
