import pandas as pd
from cmm import Accessor   
from typing import Any


class MulSeries:
    def __init__(self,data,index=None,name=None,index_init=None):
        self.index = index
        self.name = name
        print(hasattr(self, 'index'))
        self._ss = ValSeries(self,data)
        self.iloc = Accessor(self._iloc)
        self.mloc = Accessor(self._mloc)
        self.dloc = Accessor(self._dloc)

    def __repr__(self):
        return 'ss:\n'+self._ss.__repr__()+'\n\nindex:\n'+\
                self.index.__repr__()+'\n\nname:\n'+\
                self.name.__repr__()
    
    def __getattr__(self,name):
        if name == 'values':
            return self._ss.values
        elif name == 'ss':
            return pd.Series(self.values[:],
                             index=self.index.index.copy(),
                             name=self.name.name)
        elif name in ['mindex','mname']:
            return getattr(self,name.lstrip('m'))
        
    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'ss':
            raise AttributeError(f"ss is a read-only attribute.")
        else:
            super().__setattr__(name, value)
    
    def _iloc(self,key):
        new_ss = self._ss.iloc[key]
        idx_new = self.mindex.iloc[key]
        if(isinstance(new_ss,pd.Series)):
            return MulSeries(new_ss,
                             index=idx_new,
                             name=self.name.name)
        else:
            return new_ss
            
    def _mloc(self,key):
        pass

    def _dloc(self,key):
        pass
    # def __getitem__(self,key):
    #     print(key)
    #     return ['a']
        # pass


class ValSeries(pd.Series):
    def __init__(self,parent:MulSeries,ss:pd.Series):
        super().__init__(ss)
        self.parent = parent

    def __getattribute__(self, name:str):
        if name == 'index':
            return super().__getattribute__('parent').index.index
        elif name == 'name':
            return super().__getattribute__('parent').name.name
        else:
            return super().__getattribute__(name)
   