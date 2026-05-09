
'''Validation tests for dimx MDE'''
import pytest

try:
    from pyEDM import sampleData
except ImportError:
    raise( "test_MDE(): pyEDM sampleData not imported" )

try:
    import dimx as dx
except ImportError:
    raise( "test_MDE(): MDE dimx package not imported" )

from conftest import MDEArgs, ValidData, MDE_FlyData

#------------------------------------------------------------
def test_mde1():
    '''   '''
    data = sampleData["Lorenz5D"]
    kwargs = MDEArgs.copy()
    kwargs.update( dict(removeTime      = True,
                        removeColumns   = ['V5'],
                        D               = 4,
                        target          = 'V5',
                        tau             = -5,
                        exclusionRadius = 10,
                        crossMapRhoMin  = 0.2,
                        embedDimRhoMin  = 0.2,
                        firstEMax       = True) )

    mde = dx.MDE(data, **kwargs)
    mde.Run()

    df  = mde.MDEOut
    dfv = ValidData("MDE_test1_valid.csv")

    mdeOut = round(  df.iloc[:,1:], 3 )
    valid  = round( dfv.iloc[:,1:], 3 )
    assert mdeOut.equals( valid )

#------------------------------------------------------------
def test_mde2():
    '''   '''
    data = MDE_FlyData()

    kwargs = MDEArgs.copy()
    kwargs.update( dict(removeTime      = True,
                        removeColumns   = ['FWD','Left_Right'],
                        D               = 7,
                        target          = 'FWD',
                        tau             = -5,
                        crossMapRhoMin  = 0.2,
                        embedDimRhoMin  = 0.2,
                        firstEMax       = True) )

    mde = dx.MDE(data, **kwargs)
    mde.Run()

    df  = mde.MDEOut
    dfv = ValidData("MDE_test2_valid.csv")

    mdeOut = round(  df.iloc[:,1:], 2 )
    valid  = round( dfv.iloc[:,1:], 2 )
    assert mdeOut.equals( valid )

