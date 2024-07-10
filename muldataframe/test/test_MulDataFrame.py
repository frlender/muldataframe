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


