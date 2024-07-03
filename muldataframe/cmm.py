
import pandas as pd
from warnings import warn
from typing import Literal
# import muldataframe.MulSeries as MulSeries
import muldataframe as md

# MulSeries = MulSeriesModule.MulSeries

IndexType = Literal['index'] | Literal['columns']
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

def concat(ss1,ss2):
        ss_new = pd.concat([ss1.ss,ss2.ss])
        index_new = pd.concat([ss1.index,ss2.index],join='inner')
        return md.MulSeries(ss_new,index=index_new,
                        name=ss1.name.copy())
# def concat(ss1:MulSeries,ss2:MulSeries):
#     ss_new = ss1.ss.concat(ss2.ss)
#     index_new = pd.concat([ss1.index,ss2.index],join='inner')
#     return MulSeries(ss_new,index=index_new,
#                      name=ss1.name.copy)
