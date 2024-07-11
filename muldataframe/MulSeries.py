import pandas as pd
import muldataframe.cmm as cmm
# import muldataframe.util as util
from typing import Any
import numpy as np
import muldataframe as md
# import muldataframe.ValFrameBase as vfb
import muldataframe.ValFrameBase as vfb

#TODO: query for mulseries and muldataframe


class MulSeries:
    # force pandas to return NotImplemented when using ops like +, * 
    # in the case of pd.Series + MulSeries.
    __pandas_priority__ = 10000
    def __init__(self,ss,index:pd.DataFrame=None,
                 name:pd.Series|str=None,
                 index_init:cmm.IndexInit=None,
                 index_copy=True,name_copy=True):
        
        if isinstance(ss,dict):
            ss = pd.Series(ss)

        if not isinstance(name,pd.Series):
            name = pd.Series([],name=name)
        else:
            name = name.copy() if name_copy else name

        if isinstance(ss,pd.Series):
            index_init = 'align' if index_init is None else index_init
        else:
            index_init = 'override' if index_init is None else index_init
            ss = pd.Series(ss)

        ss, index = cmm.setMulIndex(ss,'index',index,index_init,index_copy)

        self.index = index
        self.name = name
        # print(hasattr(self, 'index'))
        self.__ss = ValSeries(self,ss) # private

        self.iloc = cmm.Accessor(self._xloc_get_factory('iloc'),
                             self._xloc_set_factory('iloc'))
        self.loc = cmm.Accessor(self._xloc_get_factory('loc'),
                            self._xloc_set_factory('loc'))
        self.mloc = cmm.Accessor(self._mloc_get,
                             self._mloc_set)

    def __repr__(self):
        return 'ss:\n'+self.__ss.__repr__()+'\n\nindex:\n'+\
                self.index.__repr__()+'\n\nname:\n'+\
                self.name.__repr__()
    
    def __getattr__(self,name):
        if name == 'values':
            return self.__ss.values
        elif name == 'ss':
            return pd.Series(self.values.copy(),
                             index=self.index.index.copy(),
                             name=self.name.name)
        elif name in ['mindex','mname']:
            return getattr(self,name.lstrip('m'))
        elif name == 'shape':
            return self.__ss.shape
        elif name == 'ds':
            # values are not copied version
            return pd.Series(self.values,
                             index=self.index.index.copy(),
                             name=self.name.name,
                             copy=False)
        elif hasattr(np,name) and hasattr(getattr(np,name),'__call__'):
            def func(*args,**kwargs):
                return self.call(getattr(np,name),*args,**kwargs)
            return func

        
    def _hasVal(self):
        return self.__ss is not None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'ss':
            raise AttributeError(f"ss is a read-only attribute.")
        if name in ['index','mindex','midx']:
            cmm.checkSetIdxValue(self,'index',value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __len__(self):
        return self.shape[0]

    def __eq__(self,other):
        return self.equals(other)
    
    def equals(self,other):
        if not isinstance(other,MulSeries):
            return False
        else:
            return self.ds.equals(other.ds) and self.index.equals(other.index) and self.name.equals(other.name)

    def copy(self):
        return MulSeries(self.__ss.copy().values,
                         index=self.index,
                         name=self.name.copy())
    
    def __getitem__(self,key):
        new_ss = self.__ss[key]
        if(isinstance(new_ss,pd.Series)):
            idx_new = self.mindex.loc[key]
                # print('ok')
            ms = MulSeries(new_ss,
                                index=idx_new,
                                name=self.name,
                                index_init='override')
            return ms
        else:
            return new_ss
    
    def __setitem__(self,key, values):
        self.__ss[key] = values
    
    def _xloc_get_factory(self,attr):
        def _xloc_get(key):
            new_ss = getattr(self.__ss,attr)[key]
            if(isinstance(new_ss,pd.Series)):
                idx_new = getattr(self.mindex,attr)[key]
                ms = MulSeries(new_ss,
                                    index=idx_new,
                                    name=self.name,
                                    index_init='override')
                return ms
            else:
                return new_ss
        return _xloc_get
    
    def _xloc_set_factory(self,attr):
        def _xloc_set(key,values):
            getattr(self.__ss,attr)[key] = values
        return _xloc_set

    def _mloc(self,key):
        nx = cmm._mloc_idx(key,self.index)
        return nx
    
    def _mloc_get(self,key):
        if key == slice(None):
            return self.iloc[:]
        else:
            nx = self._mloc(key)
            return self.iloc[nx]
    
    def _mloc_set(self,key,values):
        if key == slice(None):
            self.iloc[:] = values
        else:
            nx = self._mloc(key)
            self.iloc[nx] = values

    # def __add__(self,other):
    #     return self.call(pd.Series.__add__,other)

    def call(self,func,*args,**kwargs):
        # if 'unsafe' in kwargs and kwargs['unsafe']:
        # consider to change to self.ss to improve safety?
        args = list(args)
        if len(args) > 0 and (isinstance(args[0],MulSeries) or \
            isinstance(args[0],md.MulDataFrame)):
            args[0] = args[0].ds
        if len(args) > 0 and hasattr(md,'__pandas_priority__') \
            and args[0].__pandas_priority__ > self.__pandas_priority__:
            return NotImplemented
        # print(func,self,args)
        self.__ss._update_super_index()
        res = func(self.__ss,*args,**kwargs)
        errMsg = f'Currently, {self.__class__} only supports operators or functions that return a scalar value or a pandas series with the same primary index (order can be different if there are no duplicate values) in its .call() method.'
        # else:
        #     res = func(self.ss,*args,**kwargs)
        # print(res)
        if isinstance(res,pd.DataFrame):
            # print('dataframe',res)
            # have to raise error here because pandas does not support muldataframe in its __radd__ like functions.
            raise NotImplementedError(errMsg)
            # return NotImplemented
        elif isinstance(res,pd.Series):
            if res.shape[0] == self.shape[0]:
                if not res.index.equals(self.index.index):
                    try:
                        # res2 = self.loc[res.index]
                        new_idx = cmm.align_index_in_call(res.index,self,
                                                          'index')
                        if new_idx.shape[0] == self.shape[0]:
                            return MulSeries(res.values,
                                             index=new_idx,
                                             name=self.name,
                                             index_copy=False)
                        # if res2.shape[0] == self.shape[0]:
                        #     return res2
                        else:
                            raise NotImplementedError(errMsg)
                    except:
                        raise NotImplementedError(errMsg)
                else:
                    if res.name == self.name.name:
                        name = self.name.copy()
                    else:
                        name = pd.Series([],name=res.name)
                    return MulSeries(res,
                                    index=self.index,
                                    name=name,
                                    index_init='override')
            else:
                # print('shape',res.shape,self.shape)
                raise NotImplementedError(errMsg)
        else:
            return res

    def groupby(self,by=None,keep_primary=False,agg_mode:cmm.IndexAgg='same_only'):
        return cmm.groupby(self,'index',by=by,
                           keep_primary=keep_primary,agg_mode=agg_mode)
    
    
    def drop_duplicates(self,keep='first', inplace=False):
        bidx = self.__ss.duplicated(keep=keep)
        bidx_keep = ~bidx
        new_ss = self.__ss.loc[bidx]

        if inplace:
            # primary_index = self.index.index
            # primary_columns = self.columns.index
            self.__ss = None
            self.index = self.index.loc[bidx_keep]
            self.__ss = ValSeries(self,new_ss)
        else:
            return MulSeries(new_ss.values,
                        index=self.index.loc[bidx_keep],
                        name=self.name)



ops = ['add','sub','mul','div','truediv','floordiv','mod','pow']
for op in ops:
    op_attr = '__'+op+'__'
    def call_op_factory(op_attr):
        def call_op(self,other):
            func = getattr(pd.Series,op_attr)
            # print(op_attr,func)
            return self.call(func,other)
        return call_op
    setattr(MulSeries,op_attr,call_op_factory(op_attr))
    r_op_attr = '__r'+op+'__'
    setattr(MulSeries,r_op_attr,call_op_factory(r_op_attr))


# ops = []
# for op in ops:
#     op_attr = '__'+op+'__'
#     def call_op_factory(op_attr):
#         def call_op(self,other):
#             func = getattr(pd.Series,op_attr)
#             # print(op_attr,func)
#             return self.call(func,other)
#         return call_op
#     setattr(MulSeries,op_attr,call_op_factory(op_attr))


ValSeries = vfb.ValFrameBase_factory(pd.Series)
def _update_super_index(self):
    self.index = self.parent.mindex.index
    self.name = self.parent.name.name
ValSeries._update_super_index = _update_super_index

