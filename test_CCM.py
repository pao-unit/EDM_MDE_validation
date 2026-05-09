
'''Validation tests for pyEDM CCM'''
import pytest
from   numpy import nan
from   pandas import DataFrame

try:
    import pyEDM as EDM
except ImportError:
    raise( "test_CCM(): pyEDM package not imported" )

from conftest import CCMArgs, ValidData

#------------------------------------------------------------
def test_ccm1():
    '''sardine_anchovy_sst'''
    data = EDM.sampleData["sardine_anchovy_sst"]
    kwargs = CCMArgs.copy()
    kwargs.update( dict(columns  = 'anchovy',
                        target   = 'np_sst',
                        libSizes = [10,20,30,40,50,60,70,75],
                        sample   = 100,
                        E        = 3,
                        seed     = 777) )

    df  = EDM.CCM(data, **kwargs)
    dfv = ValidData("CCM_anch_sst_valid.csv")

    ccm   = round(  df.iloc[:,1:], 2 )
    valid = round( dfv.iloc[:,1:], 2 )
    assert ccm.equals( valid )

#------------------------------------------------------------
def test_ccm2():
    '''CCM Multivariate'''
    data = EDM.sampleData["Lorenz5D"]
    kwargs = CCMArgs.copy()
    kwargs.update( dict(columns  = 'V3 V5',
                        target   = 'V1',
                        libSizes = [20, 200, 500, 950],
                        sample   = 30,
                        E        = 5,
                        Tp       = 10,
                        tau      = -5,
                        seed     = 777) )

    df  = EDM.CCM(data, **kwargs)
    dfv = ValidData("CCM_Lorenz5D_MV_valid.csv")

    ccm   = round(  df.iloc[:,1:], 2 )
    valid = round( dfv.iloc[:,1:], 2 )
    assert ccm.equals( valid )

#------------------------------------------------------------
def test_ccm3():
    '''CCM nan'''
    data = EDM.sampleData["circle"]
    dfn = data.copy()
    dfn.iloc[ [5,6,12], 1 ] = nan
    dfn.iloc[ [10,11,17], 2 ] = nan
    
    kwargs = CCMArgs.copy()
    kwargs.update( dict(columns  = 'x',
                        target   = 'y',
                        libSizes = [10,190,10],
                        sample   = 100,
                        E        = 2,
                        Tp       = 5,
                        seed     = 777) )

    df  = EDM.CCM(data, **kwargs)
    dfv = ValidData("CCM_nan_valid.csv")

    ccm   = round(  df.iloc[:,1:], 2 )
    valid = round( dfv.iloc[:,1:], 2 )
    assert ccm.equals( valid )

#------------------------------------------------------------
def test_ccm4():
    '''CCM Negative Tp'''
    data = EDM.sampleData["circle"]
    kwargs = CCMArgs.copy()
    kwargs.update( dict(columns  = 'x',
                        target   = 'y',
                        libSizes = [20, 200, 50],
                        sample   = 30,
                        E        = 2,
                        Tp       = -5,
                        seed     = 777) )

    df  = EDM.CCM(data, **kwargs)
    dfv = ValidData("CCM_NegativeTp.csv")

    ccm   = round(  df.iloc[:,1:], 2 )
    valid = round( dfv.iloc[:,1:], 2 )
    assert ccm.equals( valid )

#------------------------------------------------------------
def test_ccm5():
    '''CCM exclusionRadius'''
    data = EDM.sampleData["Lorenz5D"]
    kwargs = CCMArgs.copy()
    kwargs.update( dict(columns         = 'V1',
                        target          = 'V5',
                        libSizes        = [50, 1000, 50],
                        sample          = 30,
                        E               = 5,
                        Tp              = 10,
                        tau             = -5,
                        exclusionRadius = 20,
                        seed            = 777) )

    df  = EDM.CCM(data, **kwargs)
    dfv = ValidData("CCM_exclusionRadius.csv")

    ccm   = round(  df.iloc[:,1:], 2 )
    valid = round( dfv.iloc[:,1:], 2 )
    assert ccm.equals( valid )

#------------------------------------------------------------
def test_ccm6():
    '''CCM positive tau'''
    data = EDM.sampleData["circle"]
    kwargs = CCMArgs.copy()
    kwargs.update( dict(columns  = 'x',
                        target   = 'y',
                        libSizes = [20, 120, 10],
                        sample   = 100,
                        E        = 2,
                        tau      = 3,
                        seed     = 777) )

    df  = EDM.CCM(data, **kwargs)
    dfv = ValidData("CCM_positiveTau.csv")

    ccm   = round(  df.iloc[:,1:], 2 )
    valid = round( dfv.iloc[:,1:], 2 )
    assert ccm.equals( valid )
