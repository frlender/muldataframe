import pandas as pd
import muldataframe.cmm as cmm
# import muldataframe.util as util
from typing import Any
import numpy as np
# import muldataframe.ValFrameBase as vfb
import muldataframe.ValFrameBase as vfb





class MulSeries:
    def __init__(self,ss,index:pd.DataFrame=None,
                 name:pd.Series|str=None,
                 index_init:cmm.IndexInit=None):
        
        if isinstance(ss,dict):
            ss = pd.Series(ss)

        if not isinstance(name,pd.Series):
            name = pd.Series([],name=name)

        if isinstance(ss,pd.Series):
            index_init = 'align' if index_init is None else index_init
        else:
            index_init = 'override' if index_init is None else index_init
            ss = pd.Series(ss)

        ss, index = cmm.setMulIndex(ss,'index',index,index_init)

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
            return pd.Series(self.values[:],
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
        

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'ss':
            raise AttributeError(f"ss is a read-only attribute.")
        if name == 'index':
            if self.__ss is not None:
                if value.shape[0] != self.__ss.shape[0]:
                    raise IndexError(f"index shape {value.shape[0]} is not consistent with series shape {self.__ss.shape[0]}.")
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)
    
    def __eq__(self,other):
        return self.equals(other)
    
    def equals(self,other):
        if not isinstance(other,MulSeries):
            return False
        else:
            return self.ss.equals(other.ss) and self.index.equals(other.index) and self.name.equals(other.name)

    
    def _xloc_get_factory(self,attr):
        def _xloc_get(key):
            new_ss = getattr(self.__ss,attr)[key]
            # print(new_ss)
            if(isinstance(new_ss,pd.Series)):
                idx_new = getattr(self.mindex,attr)[key]
                # print('ok')
                ms = MulSeries(new_ss,
                                    index=idx_new,
                                    name=self.name.name,
                                    index_init='override')
                # print(ms,'dddd')
                return ms
            else:
                return new_ss
        return _xloc_get
    
    def _xloc_set_factory(self,attr):
        def _xloc_set(key,values):
            getattr(self.__ss,attr)[key] = values
        return _xloc_set

    def _mloc(self,key):
        if key == slice(None):
            return self.iloc[:]
        nx = cmm._mloc_idx(key,self.index)
        return nx
    
    def _mloc_get(self,key):
        nx = self._mloc(key)
        return self.iloc[nx]
    
    def _mloc_set(self,key,values):
        nx = self._mloc(key)
        self.iloc[nx] = values

    # def __add__(self,other):
    #     return self.call(pd.Series.__add__,other)

    def call(self,func,*args,**kwargs):
        # if 'unsafe' in kwargs and kwargs['unsafe']:
        # consider to change to self.ss to improve safety?
        res = func(self.__ss,*args,**kwargs)
        # else:
        #     res = func(self.ss,*args,**kwargs)
        # print(res)
        if isinstance(res,pd.DataFrame):
            # print('dataframe',res)
            return NotImplemented
        elif isinstance(res,pd.Series):
            try:
                res = res.loc[self.index.index]
            except:
                return NotImplemented
            if res.shape[0] == self.shape[0]:
                if res.name == self.name.name:
                    name = self.name.copy()
                else:
                    name = pd.Series([],name=res.name)
                return MulSeries(res,
                                index=self.index.copy(),
                                name=name,
                                index_init='override')
            else:
                # print('shape',res.shape,self.shape)
                return NotImplemented
        else:
            return res

    def groupby(self,by=None,keep_primary=False,agg_mode:cmm.IndexAgg='same_only'):
        return cmm.groupby(self,'index',by=by,
                           keep_primary=keep_primary,agg_mode=agg_mode)
        # index_agg = agg_mode
        # if by is None or (isinstance(by,list) and None in by) or keep_primary:
        #     ms = self.loc[:]
        #     if self.index.index.name is None:
        #         for i in range(1000):
        #             name = f'primary_index' if i==0 else f'primary_index_{i}'
        #             if name not in self.index.columns:
        #                 ms.index.index.name = name
        #                 break
        #     primary_name = ms.index.index.name
        #     ms.index = self.index.reset_index()
        #     if by is None:
        #         by = primary_name
        #     elif isinstance(by,list) and None in by:
        #         by = [primary_name if b is None else b for b in by]
        #     groupBy = ms.index.groupby(by)
        #     return MulSeriesGroupBy(ms,by,groupBy,index_agg)
        # else:
        #     groupBy = self.index.groupby(by)
        #     return MulSeriesGroupBy(self,by,groupBy,index_agg)

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



# class MulSeriesGroupBy():
#     def __init__(self,parent:MulSeries,by,
#                  groupBy:pd.core.groupby.SeriesGroupBy,
#                  index_agg):
#         self.groupBy = groupBy
#         self.parent = parent
#         self.by = by
#         self.index_agg = index_agg
    
#     def __iter__(self):
#         for k,v in self.groupBy.indices.items():
#             yield k, self.parent.iloc[v]
    
#     def call(self,func,*args,**kwargs):
#         res = None
#         for i,(k,gp) in enumerate(self):
#             val = gp.call(func,*args,**kwargs)
#             if isinstance(val,MulSeries):
#                 return NotImplemented
#             index = gp.index
#             index = util.aggregate_index(i,index,self.index_agg)
#             ms = MulSeries([val],index=index,
#                            name=self.parent.name.copy())
#             if i == 0:
#                 res = ms
#             else:
#                 res = util.concat(res,ms)
#         return res


# funcs = ['mean','median','std','var','sum','prod','count','first','last','mad']
# funcs = ['mean','median','std','var','sum','prod']
# for func_name in funcs:
#     def call_func_factory(func_name):
#         def call_func(self,*args,**kwargs):
#             func = getattr(np,func_name)
#             # print(op_attr,func)
#             return self.call(func,*args,**kwargs)
#         return call_func
#     setattr(MulSeriesGroupBy,func_name,call_func_factory(func_name))


ValSeries = vfb.ValFrameBase_factory(pd.Series)
def _update_super_index(self):
    self.index = self.parent.mindex.index
    self.name = self.parent.name.name
ValSeries._update_super_index = _update_super_index

# class ValSeries(pd.Series,metaclass=vfb.ValFrameBaseMeta):
#     def _update_super_index(self):
#         self.index = self.parent.mindex.index
#         self.name = self.parent.name.name


# class ValSeries(pd.Series):
#     def __init__(self,parent:MulSeries,ss):
#         super().__init__(ss)
#         self.parent = parent

#     def __getattribute__(self, name:str):
#         if name == 'index':
#             # print(super().__getattribute__('index'),
#             #       super().__getattribute__('parent').index.index)
#             return super().__getattribute__('parent').index.index
#         elif name == 'name':
#             return super().__getattribute__('parent').name.name
#         else:
#             return super().__getattribute__(name)
   

# ops = ['add','sub','mul','div','truediv','floordiv','mod','pow']
# for op in ops:
#     op_attr = '__'+op+'__'
#     def call_op_factory(op_attr):
#         def call_op(self,other):
#             func = getattr(pd.Series,op_attr)
#             # print(op_attr,func)
#             return self.call(func,other)
#         return call_op
#     setattr(MulSeries,op_attr,call_op_factory(op_attr))
