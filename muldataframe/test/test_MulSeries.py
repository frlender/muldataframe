
from ..MulSeries import MulSeries
import pandas as pd
from .lib import eq
import pytest

def test_loc():
    index = pd.DataFrame([['a','b','c'],
                        [ 'g','b','f'],
                        [ 'b','g','h']],
                        columns=['x','y','y'],
                        index=['k','l','m'])
    name = pd.Series(['a','b'],index=['e','f'],name='cc')
    ms = MulSeries([1,2,3],index=index,name=name)

    assert ms.loc['k'] == 1
    assert eq(ms.loc[['m','l']].values, [3,2])

    ms.index.index = ['a','b','b']
    print('================')
    assert eq(ms.loc['b'].values, [2,3])
    print('88***************')
    assert eq(ms.index.index.values,['a','b','b'])
    
    ms.loc['b'] = 5
    assert eq(ms.values, [1,5,5])

    ms.loc['b'] = [5,6]
    assert eq(ms.values, [1,5,6])

    ms.loc[['a','b']] = [1,2,3]
    assert eq(ms.values, [1,2,3])
    assert ms.values[-1] == 3



def test_mloc_get():
    index = pd.DataFrame([['a','b','c'],
                        [ 'g','b','f'],
                        [ 'b','g','h']],
                        columns=['x','y','y'])
    name = name = pd.Series(['a','b'],index=['e','f'],name='cc')
    ms = MulSeries([1,2,3],index=index,name=name)
    print(type(ms.mloc[[None,'b']].values))

    assert eq(ms.mloc[[None,'b']].values,
                      [1,2])
    assert eq(ms.mloc[[None,None,['h','f']]].values,[3,2])

    assert ms.mloc[['g',None,['h','f']]] == 2

    with pytest.raises(KeyError):
        ms.mloc[['a',None,['h','f']]]
    with pytest.raises(KeyError):
        ms.mloc[[['a','g'],None,['h','f']]]
    with pytest.raises(IndexError):
        ms.mloc[[['a','g'],None,['h','f'],'kk']]
    
    assert ms.mloc[['b']] == 3
    with pytest.raises(KeyError):
        ms.mloc[{'y':'b'}]

    assert eq(ms.mloc[{'y':['c','h']}].values,[1,3])
    assert ms.mloc[{'x':'b'}] == 3

    assert eq(ms.mloc[[['a','g'],None,['f','c']]].values,[2,1])
    assert eq(ms.mloc[[['g','a'],None,['c','f']]].values,[1,2])

    assert eq(ms.mloc[{'y':['c','h'],'x':['b','a']}].values,[3,1])
    assert eq(ms.mloc[{'y':['h','c'],'x':['a','b']}].values,[1,3])




