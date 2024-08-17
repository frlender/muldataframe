
import pandas as pd
from warnings import warn
from typing import Literal, TypeVar, Generic
# import muldataframe.MulSeries as MulSeries
import muldataframe as md
# import muldataframe.util as util
import numpy as np

# MulSeries = MulSeriesModule.MulSeries

IndexType = Literal['index'] | Literal['columns']
MIndexType = Literal['mindex'] | Literal['mcolumns']
IndexInit = Literal['override'] | Literal['align']
IndexAgg = Literal['same_only'] | Literal['list'] | Literal['tuple']


class Accessor:
    def __init__(self,getter,setter,max_idx=1):
        self.getter = getter
        self.setter = setter
        self.max_idx = max_idx
    
    def __getitem__(self,key):
        if isinstance(key,tuple) and len(key) > self.max_idx:
            raise IndexError(f"Too many indices. Only {self.max_idx} {'indices are' if self.max_idx > 1 else 'index is'} allowed")
        return self.getter(key)
    
    def __setitem__(self,key,value):
        if isinstance(key,tuple) and len(key) > self.max_idx:
            raise IndexError(f"Too many indices. Only {self.max_idx} {'indices are' if self.max_idx > 1 else 'index is'} allowed")
        # print(key,value,self.setter)
        self.setter(key,value)
    


def _mloc_idx_each(nx,col_nx,idx):
    if isinstance(col_nx,pd.Series):
        col_nx_reverse = pd.Series(col_nx.index.values,
                                   index=col_nx.values)
        col_idx = col_nx_reverse.loc[idx]
        return col_idx.values if isinstance(col_idx,pd.Series) \
              else col_idx
    else:
        err_no_overlap = 'The input indices to the multi-index dataframe do not overlap.'
        if isinstance(idx,list):
            if col_nx in idx:
                return nx
            else:
                raise KeyError(err_no_overlap)
        else:
            if col_nx == idx:
                return nx
            else:
                raise KeyError(err_no_overlap)


def _mloc_idx(key,df):
    nx = list(range(df.shape[0]))
    if isinstance(key,list):
        if len(key) > df.shape[0]:
            raise IndexError(f'Too many indices. There should be at most {df.shape[0]} indices.')
        for i, idx in enumerate(key):
            if idx is not ...:
                ss = df.iloc[:,i]
                ss.index = list(range(df.shape[0]))
                col_nx = ss.loc[nx]
                nx = _mloc_idx_each(nx,col_nx,idx)
    else:
        for k, idx in key.items():
            colx = df.loc[:,k]
            if isinstance(colx, pd.DataFrame):
                colx = colx.iloc[:,-1]
                msg = f'There are more multiple {k} columns in index dataframe. Use only the last {k} column.'
                warn(msg)
            colx.index = list(range(df.shape[0]))
            col_nx = colx.loc[nx]
            nx = _mloc_idx_each(nx,col_nx,idx)
    return nx


def _nloc_idx(key:dict,df):
    nx = list(range(df.shape[0]))
    for k, idx in key.items():
        colx = df.iloc[:,k]
        colx.index = list(range(df.shape[0]))
        col_nx = colx.loc[nx]
        nx = _mloc_idx_each(nx,col_nx,idx)
    return nx


# def _n2m_key(df,key:dict):
#     key2 = {}
#     max_pos = 0
#     for k,v in key.items():
#         if not isinstance(k,int):
#             raise ValueError(f"The keys of a dict in nloc indexing can only be integers.{k} found.")
#         pos = df.shape[1]+k if k<0 else k
#         key2[pos] = v
#         if max_pos < pos:
#             max_pos = pos
#     if max_pos >= df.shape[1]:
#         raise ValueError("The position index must be smaller than the index dataframe's columns' length")
#     key3 = [...]*(max_pos+1)
#     for k,v in key2.items():
#         key3[k] = v
#     return key3

def checkAlign(label,ss_shape,index_shape):
    if ss_shape > index_shape:
        raise IndexError(f'Index Align: The {label} of the values Series/dataframe is not unique.')
    
def checkOverride(label,ss_shape,index_shape):
    if ss_shape != index_shape:
        raise IndexError(f'Index Overriding: The index length of the {label} dataframe is not the same as the {label} length of the values dataframe.')


def setMulIndex(dx:pd.Series|pd.DataFrame,indexType:IndexType,
                index:pd.DataFrame|None, index_init:IndexInit,
                index_copy:bool):
    if index is None:
        index = pd.DataFrame([],index=getattr(dx,indexType))
    else:
        if index_copy:
            index = index.copy()
        if index_init == 'override':
            checkOverride(indexType,getattr(dx,indexType).shape[0],
                          index.shape[0])
            # setattr(dx,indexType,index.index)
        else:
            # print(index,getattr(dx,indexType))
            if not index.index.equals(getattr(dx,indexType)):
                if indexType == 'index':
                    dx = dx.loc[index.index]
                else:
                    dx = dx.loc[:,index.index]
                checkAlign(indexType,getattr(dx,indexType).shape[0],index.shape[0])
    return dx, index


def checkSetIdxValue(self,name,value):
    className = self.__class__.__name__
    shapeIdx = 0 if name == 'index' else 1
    if not self._hasVal():
            return
    if not isinstance(value,pd.DataFrame) or \
        value.shape[0] != self.shape[shapeIdx]:
        raise IndexError(f"The assigned value must be a dataframe with its index length being the same as the {className}'s {name} length.")

def test_idx_eq(mindex,idx,indexType='index',
                copy=True,
                err1=KeyError('Failed to index the dataframe "mindex" using "idx"'),
                err2=IndexError('The indexed new dataframe does not have the same shape as the original "mindex" dataframe. Possibly there are duplicate values in the index of the "mindex" dataframe.')):
    # equal index even if order is not the same.
    if getattr(mindex,indexType).equals(idx):
        return mindex.copy() if copy else mindex
    try:
        if indexType == 'index':
            new_idx = mindex.loc[idx]
            if new_idx.shape[0] == mindex.shape[0]:
                return new_idx
            else:
                raise err2
        else:
            new_idx = mindex.loc[:,idx]
            if new_idx.shape[1] == mindex.shape[1]:
                return new_idx
            else:
                raise err2
    except:
        raise err1

def align_index_in_call(idx,self,indexType:IndexType):
    return test_idx_eq(getattr(self,indexType),idx,
                       err1=NotImplementedError,
                       err2=NotImplementedError)


def get_index_name(indexType,arr):
    for i in range(1000):
        name = f'primary_{indexType}' if i==0 else f'primary_{indexType}_{i}'
        if name not in arr:
            return name

def is_pandas_method(self,name):
        if isinstance(self,md.MulSeries):
            pdClass = pd.Series
        else:
            pdClass = pd.DataFrame
        return hasattr(pdClass,name) and hasattr(getattr(pdClass,name),'__call__')
    
def is_numpy_function(name):
    return hasattr(np,name) and hasattr(getattr(np,name),'__call__')


def groupby(self,indexType:IndexType|MIndexType,by=None,keep_primary=False,
            agg_mode:IndexAgg='same_only',**kwargs):
        index_agg = agg_mode
        if isinstance(self,md.MulSeries):
            M = md.MulSeries
        else:
            M = md.MulDataFrame
        if by is None or (isinstance(by,list) and None in by) or keep_primary:
            ms = self.loc[:]
            if getattr(self,indexType).index.name is None:
                getattr(ms,indexType).index.name = \
                    get_index_name('index',getattr(self,indexType).columns)
            primary_name = getattr(ms,indexType).index.name
            setattr(ms,indexType,getattr(ms,indexType).reset_index())
            # ms.index = self.index.reset_index()
            if by is None:
                by = primary_name
            elif isinstance(by,list) and None in by:
                by = [primary_name if b is None else b for b in by]
            groupBy = getattr(ms,indexType).groupby(by,**kwargs)
            return MulGroupBy[M](ms,indexType,by,groupBy,index_agg)
        else:
            groupBy = getattr(self,indexType).groupby(by)
            return MulGroupBy[M](self,indexType,by,groupBy,index_agg)

# G = TypeVar('G')
M = TypeVar('M')

class MulGroupBy(Generic[M]):
    def __init__(self,parent:M,indexType:IndexType|MIndexType,
                 by, groupBy:pd.core.groupby.DataFrameGroupBy,
                 index_agg:IndexAgg):
        self.groupBy = groupBy
        '''
        A `pandas.api.typing.DataFrameGroupBy <https://pandas.pydata.org/docs/reference/groupby.html>`_ object.
        '''
        self.parent = parent
        '''
        The parent MulSeries or MulDataFrame that calls the grouby method.
        '''
        self.by = by
        '''
        Same as the ``by`` argument in the parent's groupby method
        '''
        self.index_agg = index_agg
        '''
        Same as the by ``agg_mode`` argument in the parent's groupby method
        '''
        self.indexType = indexType
        '''
        The index dataframe used to group by the parent.

        It must be ``'index'`` if parent is a MulSeries. It can be ``'index'`` or ``'columns'`` if parent is a MulDataFrame.
        '''
    
    def __iter__(self):
        '''
        Make the MulGroupBy object iterable.
        '''
        if self.indexType in ['index','mindex','midx']:
            for k,v in self.groupBy.indices.items():
                yield k, self.parent.iloc[v]
        else:
            for k,v in self.groupBy.indices.items():
                yield k, self.parent.iloc[:,v]
    

    def __getattr__(self,name):
        if hasattr(np,name) and hasattr(getattr(np,name),'__call__'):
            def func(*args,**kwargs):
                return self.call(getattr(np,name),*args,**kwargs)
            return func
        
    def call(self,func,*args,**kwargs):
        '''
        Call a function on the MulGroupBy object.

        Similar to `MulSeries.call <../mulseries/call>`_ and `MulDataFrame.call <../muldataframe/call>`_, it applies a function to the MulSeries or MulDataFrame in each group and concateate the results into a final MulSeries or MulDataFrame.

        Parameters:
        ------------
        func: function
            A function applied to the MulSeries or MulDataFrame in each group.
        use_mul: bool, default False
            An optional argument to determine how :code:`func` is applied. If False, :code:`MulSeries.call(func)` or :code:`MulDataFrame.call(func)` is used to compute the results in each group. The object passed to :code:`func` will be the MulSeries or the MulDataframe's values Series or DataFrame in each group. If True, :code:`func(MulSeries)` or :code:`func(MulDataFrame)` are used to compute the results.
        
        Returns
        --------
        MulSeries or MulDataFrame
            The return value is a MulSeries if the MulGroupBy object's parent is a MulSeries. Otherwise, it is a MulDataFrame
        '''
        # print('******',G,M,type(G),)
        M_class = self.__orig_class__.__args__[0]
        res = None
        use_mul = False
        arr = []
        if 'use_mul' in kwargs:
            use_mul = kwargs['use_mul']
            del kwargs['use_mul']
        for i,(k,gp) in enumerate(self):
            if use_mul:
                val = func(gp,*args,**kwargs)
            else:
                val = gp.call(func,*args,**kwargs)
            if isinstance(val,M_class): # MulSeries
               arr.append(val)
                # return NotImplemented
            else:
                index = gp.index
                index = md.aggregate_index(i,index,self.index_agg)
                if not isinstance(val,md.MulDataFrame) and \
                    not isinstance(val,md.MulSeries):
                    if isinstance(self.parent,md.MulSeries):
                        name = self.parent.name.copy()
                    else:
                        name = func.__name__
                    ms = md.MulSeries([val],index=index,
                                name=name)
                elif isinstance(self.parent,md.MulDataFrame) and \
                    isinstance(val,md.MulSeries):
                    if self.indexType == 'index':
                         ms = md.MulDataFrame([val.values],
                            index=index,
                            columns=val.index,both_copy=False)
                        #  print(val,ms)
                    else:
                        ms = md.MulDataFrame([[x] for x in val.values],
                            index=val.index,
                            columns=index,both_copy=False)
                else:
                    raise NotImplementedError('The function applied to a group MulDataFrame can only produce a A MulDataFrame, a MulSeries or a scalar. The function applied to a group MulSeries can only produce a MulSeries or a scalar.')


                arr.append(ms)
        axis = 0 if self.indexType == 'index' else 1
        res = md.concat(arr,axis=axis)
        return res

def fmtSeries(ss:pd.Series):
    if ss.shape[0] == 0:
        return str(ss)
    else:
        df = pd.DataFrame(ss)
        if ss.name is None:
            df.columns = [None]
        return df


def fmtColStr(df:pd.DataFrame,transpose=True):
    if df.shape[0] == 0 or df.shape[1] == 0:
        return str(df)
    else:
        if transpose:
            df = df.transpose()
        xs = str(df)
        lines = xs.split('\n')
        lines = lines[1:][::-1]+[lines[0]]
        return '\n'.join(lines)


# funcs = ['mean','median','std','var','sum','prod','count','first','last','mad']
# funcs = ['mean','median','std','var','sum','prod']
# for func_name in funcs:
#     def call_func_factory(func_name):
#         def call_func(self,*args,**kwargs):
#             func = getattr(np,func_name)
#             # print(op_attr,func)
#             return self.call(func,*args,**kwargs)
#         return call_func
#     setattr(MulGroupBy,func_name,call_func_factory(func_name))


# def concat(ss1:MulSeries,ss2:MulSeries):
#     ss_new = ss1.ss.concat(ss2.ss)
#     index_new = pd.concat([ss1.index,ss2.index],join='inner')
#     return MulSeries(ss_new,index=index_new,
#                      name=ss1.name.copy)
