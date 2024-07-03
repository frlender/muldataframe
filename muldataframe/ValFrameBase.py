
import muldataframe.cmm as cmm
# import pandas as pd
# from abc import ABC,abstractmethod


# class ValFrameBase(ABC):
class ValFrameBase():
    def __init__(self,parent,df):
        super().__init__(df)
        self.parent = parent
        self._iloc_accessor = cmm.Accessor(self._iloc,
                                 self._set_iloc,len(self.shape))
        self._loc_accessor = cmm.Accessor(self._loc,
                                 self._set_loc,len(self.shape))

    # @abstractmethod
    def _update_super_index(self):
        self.index = self.parent.mindex.index
        self.columns = self.parent.mcolumns.index

    def __getitem__(self,key):
        # # Without calling self._update_super_index(),
        # # the expression below uses self.parent.mindex.index
        # # to index but returns a dataframe or series with
        # # super(ValDataFrame,self).index.
        # res = super().__getitem__(key)
        self._update_super_index()
        return super().__getitem__(key)
        
    def _iloc(self,key):
        self._update_super_index()
        return super().iloc[key]
    
    def _set_iloc(self,key,value):
        self._update_super_index()
        super().iloc[key] = value
    
    def _loc(self,key):
        self._update_super_index()
        return super().loc[key]
    
    def _set_loc(self,key,value):
        self._update_super_index()
        super().loc[key] = value


    def __getattribute__(self, name:str):
        print(name)
        if name == 'index':
            return super().__getattribute__('parent').index.index
        elif name == 'columns' and hasattr(self,'columns'):
            return super().__getattribute__('parent').columns.index
        elif name == 'name' and hasattr(self,'name'):
            return super().__getattribute__('parent').name.name
        elif name == 'iloc':
            return super().__getattribute__('_iloc_accessor')
        elif name == 'loc':
            return super().__getattribute__('_loc_accessor')
        else:
            return super().__getattribute__(name)
        

class ValFrameBaseMeta(type):
    def __new__(cls, clsname, superclasses, attributedict):
        for k,v in dict(ValFrameBase.__dict__).items():
            if k not in attributedict:
                attributedict[k] = v
        return type(clsname, superclasses, attributedict)