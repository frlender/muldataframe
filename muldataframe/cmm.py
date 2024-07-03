
import pandas as pd
from warnings import warn
from typing import Literal
# import muldataframe.MulSeries as MulSeries
import muldataframe as md
import muldataframe.util as util
import numpy as np
# MulSeries = MulSeriesModule.MulSeries

IndexType = Literal['index'] | Literal['columns']
MIndexType = Literal['mindex'] | Literal['mcolumns']
IndexInit = Literal['override'] | Literal['align']
IndexAgg = Literal['same_only'] | Literal['array'] | Literal['tuple']


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
            if idx is not None:
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
    

def checkAlign(label,ss_shape,index_shape):
    if ss_shape > index_shape:
        raise IndexError(f'Index Align: The {label} of the values Series/dataframe is not unique.')
    
def checkOverride(label,ss_shape,index_shape):
    if ss_shape != index_shape:
        raise IndexError(f'Index Overriding: The index length of the {label} dataframe is not the same as the {label} length of the values dataframe.')


def setMulIndex(dx:pd.Series|pd.DataFrame,indexType:IndexType,
                index:pd.DataFrame|None, index_init:IndexInit):
    if index is None:
        index = pd.DataFrame([],index=getattr(dx,indexType))
    else:
        if index_init == 'override':
            checkOverride(indexType,getattr(dx,indexType).shape[0],
                          index.shape[0])
            # setattr(dx,indexType,index.index)
        else:
            if indexType == 'index':
                dx = dx.loc[index.index]
            else:
                dx = dx.loc[:,index.index]
            checkAlign(indexType,getattr(dx,indexType).shape[0],index.shape[0])
    return dx, index


def groupby(self,indexType:IndexType|MIndexType,by=None,keep_primary=False,
            agg_mode:IndexAgg='same_only',**kwargs):
        index_agg = agg_mode
        if isinstance(self,md.MulSeries):
            G = pd.core.groupby.SeriesGroupBy
            M = md.MulSeries
        else:
            G = pd.core.groupby.DataFrameGroupBy
            M = md.MulDataFrame
        if by is None or (isinstance(by,list) and None in by) or keep_primary:
            ms = self.loc[:]
            if getattr(self,indexType).index.name is None:
                for i in range(1000):
                    name = f'primary_index' if i==0 else f'primary_index_{i}'
                    if name not in getattr(self,indexType).columns:
                        getattr(ms,indexType).index.name = name
                        break
            primary_name = getattr(ms,indexType).index.name
            setattr(ms,indexType,getattr(ms,indexType).reset_index())
            # ms.index = self.index.reset_index()
            if by is None:
                by = primary_name
            elif isinstance(by,list) and None in by:
                by = [primary_name if b is None else b for b in by]
            groupBy = getattr(ms,indexType).groupby(by,**kwargs)
            return MulGroupBy[G,M](ms,indexType,by,groupBy,index_agg)
        else:
            groupBy = getattr(self,indexType).groupby(by)
            return MulGroupBy[G,M](self,indexType,by,groupBy,index_agg)


class MulGroupBy[G,M]():
    def __init__(self,parent,indexType:IndexType|MIndexType,
                 by, groupBy:G, #pd.core.groupby.SeriesGroupBy,
                 index_agg:IndexAgg):
        self.groupBy = groupBy
        self.parent = parent
        self.by = by
        self.index_agg = index_agg
        self.indexType = indexType
    
    def __iter__(self):
        if self.indexType in ['index','mindex']:
            for k,v in self.groupBy.indices.items():
                yield k, self.parent.iloc[v]
        else:
            for k,v in self.groupBy.indices.items():
                yield k, self.parent.iloc[:,v]
    
    def call(self,func,*args,**kwargs):
        res = None
        for i,(k,gp) in enumerate(self):
            val = gp.call(func,*args,**kwargs)
            if isinstance(val,M): # MulSeries
                return NotImplemented
            index = gp.index
            index = util.aggregate_index(i,index,self.index_agg)
            if isinstance(self,md.MulDataFrame) and \
                isinstance(val,md.MulSeries):
                ms = M([val.values],index=index,
                       columns=val.columns.copy())
            else:
                if isinstance(self.parent,md.MulSeries):
                    name = self.parent.name.copy()
                else:
                    name = func.__name__
                ms = M([val],index=index,
                            name=name)
            if i == 0:
                res = ms
            else:
                res = util.concat(res,ms)
        return res


# funcs = ['mean','median','std','var','sum','prod','count','first','last','mad']
funcs = ['mean','median','std','var','sum','prod']
for func_name in funcs:
    def call_func_factory(func_name):
        def call_func(self,*args,**kwargs):
            func = getattr(np,func_name)
            # print(op_attr,func)
            return self.call(func,*args,**kwargs)
        return call_func
    setattr(MulGroupBy,func_name,call_func_factory(func_name))


# def concat(ss1:MulSeries,ss2:MulSeries):
#     ss_new = ss1.ss.concat(ss2.ss)
#     index_new = pd.concat([ss1.index,ss2.index],join='inner')
#     return MulSeries(ss_new,index=index_new,
#                      name=ss1.name.copy)
