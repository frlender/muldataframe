import pandas as pd
from .cmm import Accessor, _mloc_idx, setMulIndex, IndexInit
from typing import Any



class MulSeries:
    def __init__(self,ss,index:pd.DataFrame=None,name:pd.Series=None,index_init:IndexInit=None):
        self.iloc = Accessor(self._xloc_get_factory('iloc'),
                             self._xloc_set_factory('iloc'))
        self.loc = Accessor(self._xloc_get_factory('loc'),
                            self._xloc_set_factory('loc'))
        self.mloc = Accessor(self._mloc_get,
                             self._mloc_set)

        if isinstance(ss,pd.Series):
            name = pd.Series([],name=ss.name) if name is None else name
            index_init = 'align' if index_init is None else index_init
        else:
            name = pd.Series([]) if name is None else name
            index_init = 'override' if index_init is None else index_init
            ss = pd.Series(ss)

        ss, index = setMulIndex(ss,'index',index,index_init)

        self.index = index
        self.name = name
        # print(hasattr(self, 'index'))
        self.__ss = ValSeries(self,ss) # private

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
        

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'ss':
            raise AttributeError(f"ss is a read-only attribute.")
        if name in ['__add__']:
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)
    
    def _xloc_get_factory(self,attr):
        def _xloc_get(key):
            new_ss = getattr(self.__ss,attr)[key]
            idx_new = getattr(self.mindex,attr)[key]
            if(isinstance(new_ss,pd.Series)):
                return MulSeries(new_ss,
                                    index=idx_new,
                                    name=self.name.name)
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
        nx = _mloc_idx(key,self.index)
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
            print('dataframe',res)
            return NotImplemented
        elif isinstance(res,pd.Series):
            # may add more checks here
            if res.shape[0] == self.shape[0]:
                return MulSeries(res,
                                index=self.index.copy(),
                                name=self.name.copy())
            else:
                print('shape',res.shape,self.shape)
                return NotImplemented
        else:
            return res



class ValSeries(pd.Series):
    def __init__(self,parent:MulSeries,ss):
        super().__init__(ss)
        self.parent = parent

    def __getattribute__(self, name:str):
        if name == 'index':
            # print(super().__getattribute__('index'),
            #       super().__getattribute__('parent').index.index)
            return super().__getattribute__('parent').index.index
        elif name == 'name':
            return super().__getattribute__('parent').name.name
        else:
            return super().__getattribute__(name)
   

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
