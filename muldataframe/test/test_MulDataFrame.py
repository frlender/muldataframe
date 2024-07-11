from ..MulSeries import MulSeries
from ..MulDataFrame import MulDataFrame
import pandas as pd
from .lib import eq
import pytest
import numpy as np

def get_data():
    index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
    columns = pd.DataFrame([[5,7],[3,6]],
                        index=['c','d'],
                        columns=['f','g'])
    md = MulDataFrame([[1,2],[8,9],[8,10]],index=index,
                    columns=columns)
    return md,index,columns

def test_init():
    index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
    columns = pd.DataFrame([[5,7],[3,6]],
                        index=['c','d'],
                        columns=['f','g'])
    md = MulDataFrame([[1,2],[8,9],[8,10]],index=index,
                    columns=columns)
    
    index.iloc[0,0] = 7
    assert md.mindex.iloc[0,0] == 1
    md.mindex.iloc[2,0] = 8
    assert index.iloc[2,0] == 5
    columns.iloc[0,0] = 7
    assert md.mcolumns.iloc[0,0] == 5

    index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
    columns = pd.DataFrame([[5,7],[3,6]],
                        index=['c','d'],
                        columns=['f','g'])
    md = MulDataFrame([[1,2],[8,9],[8,10]],
                      index=index,
                    columns=columns,
                    both_copy=False)
    index.iloc[0,0] = 7
    assert md.mindex.iloc[0,0] == 7
    md.mindex.iloc[2,0] = 8
    assert index.iloc[2,0] == 8
    columns.iloc[0,0] = 7
    assert md.mcolumns.iloc[0,0] == 7

    index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
    md = MulDataFrame(index)
    assert md.mindex.shape == (3,0)
    assert md.mcolumns.shape == (2,0)

    # print('===================')
    md = MulDataFrame(index,index=index)
    index2 = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['b','a','b'],
                     columns=['x','y'])
    with pytest.raises(IndexError):
        MulDataFrame(index,index=index2)
    md = MulDataFrame(index.values,index=index2)
    assert eq(md.mindex.index.values,['b','a','b'])
    md = MulDataFrame(index.values,index=index2,
                      index_init='override')
    assert eq(md.mindex.index.values,['b','a','b'])


    columns2 = pd.DataFrame([[5,7],[3,6]],
                        index=['d','c'],
                        columns=['f','g'])
    md = MulDataFrame(columns,index=columns2)
    assert eq(md.mindex.index.values,['d','c'])


def test_get_set_attr():
    md,index,columns = get_data()
    assert eq(md.values,md.df.values)
    assert eq(md.values,md.ds.values)

    values = md.values
    dvals = md.df.values
    svals = md.ds.values
    values[0,0] = 5
    assert md.values[0,0] == 5
    assert svals[0,0] == 5
    assert dvals[0,0] == 1

    assert md.shape == (3,2)

    index2 = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['b','a','b'],
                     columns=['x','y'])
    md.index = index2
    assert md.df.index.equals(index2.index)
    md.mindex = index
    assert md.df.index.equals(index.index)

    # md.index = columns
    with pytest.raises(IndexError):
        md.index = columns


def test_get_setitem():
    md,index,columns = get_data()
    ss = md['c']
    assert ss.index.equals(md.mindex)
    assert isinstance(ss,MulSeries)
    assert eq(ss.values,[1,8,8])

    md.columns.index = ['c','c']
    md2 = md['c']
    assert md2 == md

    md,index,columns = get_data()
    md['c'] = [5,5,5]
    assert eq(md.loc[:,'c'].values,[5,5,5])


def test_op_call():
    md,index,columns = get_data()
    assert md != md.df

    md2 = md*2
    md3 = 2*md
    assert md2 == md3
    assert eq(md2.iloc[:,0],[2,16,16])

    md2 = md.sum(axis=0)
    assert eq(md2.values,[17,21])

    md2 = md.power(2)
    assert md2.iloc[1,0] == 64

    md.iloc[1,0] = 6
    assert md.mean() == 6

    md,index,columns = get_data()
    
    md2 = md + np.array([[1,1],[1,1],[1,1]])
    assert isinstance(md,MulDataFrame)
    assert md2.iloc[0,0] == 2
    assert md2.iloc[2,1] == 11

    md2 = md + pd.DataFrame([[1,1],[1,1],[1,1]],index=['a','b','b'],
                  columns=['c','d'])
    assert isinstance(md,MulDataFrame)
    assert md2.iloc[0,0] == 2
    assert md2.iloc[2,1] == 11

    md2 = md * pd.Series([2,2],index=['c','d'])
    assert md2.iloc[0,0] == 2
    assert md2.iloc[2,1] == 20

    md2 = md * MulSeries(pd.Series([2,2],index=['c','d']))
    assert md2.iloc[0,0] == 2
    assert md2.iloc[2,1] == 20

    md2 = md.mean(axis=1)
    assert eq(md2.values,[1.5,8.5,9.0])

    def func(df):
        return df.iloc[:2]
    with pytest.raises(NotImplementedError):
        md.call(func)
    
    def func(df):
        return df.iloc[:,[1,0]]
    md2 = md.call(func)
    assert eq(md2.iloc[0].values,[2,1])
    

def test_mloc():
    md,index,columns = get_data()
    md2 = md.mloc[[None,6]]
    assert eq(md2.values,[[8,9],[8,10]])
    md2 = md.mloc[[[1,3],6]]
    assert eq(md2.values,[8,9])

    md2 = md.mloc[[[1,3],[6]]]
    assert eq(md2.values,[[8,9]])

    md2 = md.mloc[[3,6]]
    assert eq(md2.values,[8,9])

    with pytest.raises(KeyError):
        md.mloc[[1,6]]

    md2 = md.mloc[{'y':6,'x':3}]
    assert eq(md2.values,[8,9])
    md2 = md.mloc[{'y':6,'x':[3]}]
    assert eq(md2.values,[[8,9]])
    with pytest.raises(KeyError):
        md.mloc[{'y':6,'x':[1,3]}]

    md2 = md.mloc[:,[None,7]]
    assert eq(md2.values,[1,8,8])
    md2 = md.mloc[[None,6],[None,7]]
    assert eq(md2.values,[8,8])

    md2 = md.mloc[[None,2],{'g':6}]
    assert md2 == 2

    md.mloc[[None,2],{'g':6}] = 3
    assert md.iloc[0,1] == 3

    md.mloc[[None,6],{'f':5}] = [5,5]
    assert eq(md.iloc[1:,0].values, [5,5])

    md.mcols.iloc[:,0] = [5,5]
    md.mloc[[1],[5]] = [3,3]
    assert eq(md.iloc[0,:].values, [3,3])

def test_set_index():
    md,index,columns = get_data()
    md2 = md.set_index('c')
    assert md2.shape == (3,1)
    assert md2.mindex.shape == (3,3)
    assert eq(md2.mindex['c'], [1,8,8])

    md.set_index('c',inplace=True,drop=False)
    assert md.shape == (3,2)
    assert md.mindex.shape == (3,3)
    assert eq(md.mindex['c'], [1,8,8])

    md,index,columns = get_data()
    md.set_index('c',inplace=True,drop=True)
    assert md.shape == (3,1)
    assert md.mindex.shape == (3,3)
    assert eq(md.mindex['c'], [1,8,8])

    md,index,columns = get_data()
    md.mcols.iloc[:,0] = [5,5]
    md2 = md.set_index(mloc=[5])
    # print(md2)
    assert md2.shape == (3,0)
    assert md2.mindex.shape == (3,4)
    assert eq(md2.mindex['c'], [1,8,8])

    with pytest.raises(ValueError):
        md.set_index()


def test_drop_duplicates():
    md,index,columns = get_data()
    md2 = md.drop_duplicates('c')
    assert eq(md2.values,[[1,2],[8,9]])

    md2 = md.drop_duplicates(['c','d'])
    assert md2.equals(md)

    md2 = md.drop_duplicates(mloc=[3])
    assert md2.equals(md)

    md2 = md.drop_duplicates(mloc={'g':7})
    assert eq(md2.values,[[1,2],[8,9]])

    md.drop_duplicates('c',inplace=True)
    assert eq(md.values,[[1,2],[8,9]])

    with pytest.raises(ValueError):
        md.drop_duplicates()

def test_iterrows():
    md,index,columns = get_data()
    for i, (k,row) in enumerate(md.iterrows()):
        assert isinstance(row,MulSeries)
        if i == 0:
            assert k.name == 'a'
        else:
            assert k.name == 'b'


def test_groupby():
    md, index, columns = get_data()
    for i, (k,gp) in md.groupby('y'):
        if i==0:
            assert k == 2
            assert gp.shape == (1,2)
            assert isinstance(gp,MulDataFrame)
        else:
            assert k == 6
            assert gp.shape == (2,2)


def test_query():
    index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
    columns = pd.DataFrame([[5,7],[3,6]],
                        index=['c','d'],
                        columns=['f','g'])
    md = MulDataFrame([[1,2],[8,9],[8,10]],index=index,
                    columns=columns)
    
    md2 = md.query('c == 8')
    assert md2.shape == (2,2)
    assert md2.iloc[0,0] == 8

    md3 = md.query(index='y > 3')
    assert md3 == md2

    md4 = md.query(index='y > 3',
                   columns='f>3')
    assert md4.shape == (2,1)
    assert eq(md4.values,[[8],[8]])
    assert eq(md4.mindex.values,[[3,6],[5,6]])

    md5 = md.query('c==8',index='x<=3')
    assert md5.shape == (1,2)
    assert eq(md5.values,[[8,9]])

    md6 = md.query('c==8',index='x<=3',
                   columns='f>3')
    assert md6.shape == (1,1)
    assert eq(md6.values,[[8]])
    assert md6.mcols.shape == (1,2)

    md7 = md.copy()
    md7.query('c==8',index='x<=3',
                   columns='f>3',
                   inplace=True)
    assert md6 == md7

    md8 = md.query(columns='g < 7')
    assert md8.shape == (3,1)




def test_melt():
    index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
    columns = pd.DataFrame([[5,7],[3,6]],
                        index=['c','d'],
                        columns=['f','g'])
    md = MulDataFrame([[1,2],[8,9],[8,10]],index=index,
                    columns=columns)
    
    mf = md.melt()
    # print(mf)
    assert mf.shape == (md.shape[0]*md.shape[1],7)
    assert eq(mf['value'].values,np.ravel(md.values))
    assert eq(mf.columns.tolist(),
              ['index','x','y','index','f','g','value'])
    
    mf2 = md.melt(prefix=True,value_name='num')
    assert mf2.shape == (md.shape[0]*md.shape[1],7)
    assert mf2.columns[-1] == 'num'
    assert mf2.columns[0] == 'x_index'
    assert eq(mf2['x_index'].values,
              ['a','a','b','b','b','b'])
    assert eq(mf2['y_index'].values,
              ['c','d','c','d','c','d'])
    
    md.mcols.columns = ['x','g']
    def prefix_func(indexType,label):
        if indexType == 'index':
            label = f'row_{label}'
        else:
            label = f'col_{label}'
        return label
    
    mf3 = md.melt(prefix=prefix_func,
                  ignore_primary_columns=True,
                  ignore_primary_index=True)
    assert mf3.shape == (md.shape[0]*md.shape[1],5)
    assert eq(mf3.columns.tolist(),
              ['row_x','y','col_x','g','value'])


