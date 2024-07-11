
from ..MulSeries import MulSeries
import pandas as pd
from .lib import eq
import pytest
import numpy as np

def test_init():
    index = pd.DataFrame([['a','b'],['c','d']])
    with pytest.raises(IndexError):
        MulSeries([1,2,3],index=index)
    
    index = pd.DataFrame([['a','b'],['c','d']],
                         index=['e','f'])
    assert eq(MulSeries(pd.Series([1,2],index=['f','e']),
                        index=index).values,[2,1])
    assert eq(MulSeries({'f':1,'e':2},index=index).values,[2,1])
    assert eq(MulSeries({'f':1,'e':2,'g':3},index=index).values,[2,1])

    with pytest.raises(IndexError):
        MulSeries(pd.Series([1,2,3],index=['f','e','f']),
                        index=index)
    
    index = pd.DataFrame([['a','b'],['c','d'],
                          ['g','h']],
                         index=['e','f','f'])
    assert eq(MulSeries({'f':1,'e':2},index=index).values,[2,1,1])

def test_getattr():
    index = pd.DataFrame([['a','b'],['c','d']],
                         index=['e','f'])
    ms = MulSeries(pd.Series([1,2],index=['e','f']), 
                        index=index)

    ss = ms.ss
    ss.index = ['k','l']
    assert eq(ms.index.index.values,['e','f'])
    ss.iloc[0] = 5
    assert ms.iloc[0] == 1

    ss = ms.ds
    ss.index = ['k','l']
    assert eq(ms.index.index.values,['e','f'])
    ss.iloc[0] = 5
    assert ms.iloc[0] == 5

    assert ms.shape == (2,)

    assert ms.mindex.equals(ms.index)
    assert ms.mname.equals(ms.name)

    index = pd.DataFrame([['a','b'],['c','d'],
                          ['g','h']],
                         index=['e','f','f'])
    ms = MulSeries(pd.Series([1,2,3],index=['e','f','f']), 
                        index=index)
    ms2 = ms['f']
    index = pd.DataFrame([['c','d'],
                          ['g','h']],
                         index=['f','f'])
    mse = ms = MulSeries(pd.Series([2,3],index=['f','f']), 
                        index=index)
    assert ms2 == mse



def test_setattr():
    index = pd.DataFrame([['a','b'],['c','d']],
                         index=['e','f'])
    ms = MulSeries(pd.Series([1,2],index=['e','f']), 
                        index=index)
    index = pd.DataFrame([['a','b'],
                          ['c','d'],
                          ['e','f']],
                          index=['g','h','i'])
    with pytest.raises(AttributeError):
        ms.ss = pd.Series([1,5])
    with pytest.raises(IndexError):
        ms.index = index
    

def test_ops_get_set_items():
    index = pd.DataFrame([['a','b'],['c','d']],
                         index=['e','f'])
    ms = MulSeries(pd.Series([1,3],index=['e','f']), 
                        index=index)
    ms2 = MulSeries(pd.Series([1,3],index=['e','f']), 
                        index=index)
    assert ms.equals(ms2)
    assert ms == ms2
    assert ms != pd.Series([1,3],index=['e','f'])

    def foo(ss):
        return ss.iloc[::-1]
    msf = ms.call(foo)
    assert eq(msf.values,[3,1])

    def foo2(ss):
        return ss.iloc[[0,1,1]]
    with pytest.raises(NotImplementedError):
        msf = ms.call(foo2)

    def foo3(ss):
        return pd.Series([1,3],index=['a','b'])
    with pytest.raises(NotImplementedError):
        msf = ms.call(foo3)

    ms2['e'] = 5
    assert ms != ms2
    assert ms2.ss.iloc[0] == 5
    assert ms2['e'] == 5

    ms3 = ms+ms2
    assert eq(ms3.values,[6,6])

    mse = MulSeries(pd.Series([1,3],index=['e','f']), 
                        index=index)
    ms2e = MulSeries(pd.Series([5,3],index=['e','f']), 
                        index=index)
    assert mse == ms
    assert ms2e == ms2

    msa = ms*2
    msb = 2*ms
    assert eq(msa.values,[2,6])
    assert msa == msb

    ms2 = MulSeries(pd.Series([1,3],index=['e','f']), 
                        index=index)
    ms2.index.index = ['e','k']

    assert not ms2.index.equals(index)
    with pytest.raises(NotImplementedError):
        ms + ms2
    
    index = pd.DataFrame([['a','b'],['c','d']],
                         index=['e','f'])
    ms = MulSeries(pd.Series([1,3],index=['e','f']), 
                        index=index,
                        index_copy=False)
    ms.index.index = ['e','k']
    assert eq(index.index.values,['e','k'])

    ms.index.index = ['e','f']
    msc = ms.call(np.power,2)
    assert eq(msc.values,[1,9])

    msc = ms.call(np.sum)
    assert msc == 4

    msc = ms.call(np.add,[5,6])
    # print(msc)
    assert eq(msc.values,[6,9])

    index = pd.DataFrame([['a','b'],['c','d'],
                          ['e','f']],
                         index=['a','b','b'])
    ms = MulSeries(pd.Series([1,3,5],
                    index=['a','b','b']), 
                        index=index,
                        name='abc')
    msp = ms+pd.Series([1,3,5],index=['a','b','b'])
    assert eq(msp.values,[2,6,10])
    assert msp.name.name is None

    with pytest.raises(NotImplementedError):
        ms+pd.Series([1,3,5],index=['b','a','b'])

    with pytest.raises(NotImplementedError):
        ms.call(foo)
    
    def to_df(ss):
        return pd.DataFrame([])
    
    with pytest.raises(NotImplementedError):
        ms.call(to_df)

    index = pd.DataFrame([['a','b'],['c','d']],
                         index=['e','f'])
    ms = MulSeries(pd.Series([1,3],index=['e','f']), 
                        index=index)
    assert ms.mean() == 2

    ms2 = ms.multiply(2)
    assert isinstance(ms2,MulSeries)
    assert eq(ms2.values,[2,6])

    ms2 = ms.log2()
    assert ms2.iloc[0] == 0


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
    assert eq(ms.loc['b'].values, [2,3])
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
    # print(type(ms.mloc[[None,'b']].values))
    assert eq(ms.mloc[:].values,[1,2,3])

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

    index = pd.DataFrame([['a','b','c'],
                        [ 'g','b','f'],
                        [ 'b','g','h']],
                        columns=['x','y','z'])
    name = name = pd.Series(['a','b'],index=['e','f'],name='cc')
    ms = MulSeries([1,2,3],index=index,name=name)
    assert eq(ms.mloc[[None,'b']].values,[1,2])
    
    res = ms.mloc[{'y':[True, False, True]}]
    assert eq(res.values,[1,3])

def test_mloc_set():
    index = pd.DataFrame([['a','b','c'],
                        [ 'g','b','f'],
                        [ 'b','g','h']],
                        columns=['x','y','y'])
    name = name = pd.Series(['a','b'],index=['e','f'],name='cc')
    ms = MulSeries([1,2,3],index=index,name=name)
    ms.mloc[[None,'g']] = 5
    assert eq(ms.values,[1,2,5])

    ms.iloc[:] = [1,2,3]
    assert eq(ms.values,[1,2,3])
    ms.mloc[:] = [1,2,3]
    assert eq(ms.values,[1,2,3])

    ms.mloc[{'y':['c','f']}] = [5,6]
    assert eq(ms.values,[5,6,3])

    index = pd.DataFrame([['a','b','c'],
                        [ 'g','b','f'],
                        [ 'b','g','h']],
                        columns=['x','y','z'])
    name = name = pd.Series(['a','b'],index=['e','f'],name='cc')
    ms = MulSeries([1,2,3],index=index,name=name)
    ms.mloc[{'y':'b'}] = [5,6]
    assert eq(ms.values,[5,6,3])



def test_groupby():
    index = pd.DataFrame([['a','b','c'],
                        [ 'g','b','f'],
                        [ 'b','g','h']],
                        columns=['x','y','z'])
    name = name = pd.Series(['a','b'],index=['e','f'],name='cc')
    ms = MulSeries([1,2,3],index=index,name=name)

    assert eq(ms.groupby('y').sum().values,[3,3])
    ms2 = ms.copy()
    ms2.index.set_index('x',inplace=True)
    res = ms2.groupby('y').mean()
    assert eq(res.values,[1.5,3])
    assert eq(res.index.index.values,[0,1])
    assert eq(res.index.columns.values,['y'])

    res = ms2.groupby('y',agg_mode='array').mean()
    assert eq(res.index.columns.values,['y','z'])
    assert eq(res.index['z'].values[0],['c','f'])


    res = ms2.groupby('y',keep_primary=True,
                      agg_mode='array').mean()
    assert eq(res.index.index.values,[0,1])
    assert eq(res.index.columns.values,['x','y','z'])
    assert eq(res.index['z'].values[0],['c','f'])
    assert eq(res.index['x'].values[0],['a','g'])


    def gpAdd(gp):
        if gp.mindex['y'].iloc[0] == 'b':
            return gp+5
        else:
            return gp
    msa = ms.groupby('y').call(gpAdd,use_mul=True)
    assert eq(msa.values,[6,7,3])