import pandas as pd
import muldataframe.cmm as cmm

def ValFrameBase_factory(baseClass:pd.DataFrame|pd.Series):
    class ValFrameBase(baseClass):
        def __init__(self,parent,df):
            super().__init__(df)
            self.parent = parent

        # @abstractmethod
        def _update_super_index(self):
            self.index = super().__getattribute__('parent').mindex.index
            self.columns = super().__getattribute__('parent').mcolumns.index

        def __getitem__(self,key):
            # # Without calling self._update_super_index(),
            # # the expression below uses self.parent.mindex.index
            # # to index but returns a dataframe or series with
            # # super(ValDataFrame,self).index.
            # res = super().__getitem__(key)
            self._update_super_index()
            return super().__getitem__(key)
            


        def __getattribute__(self, name:str):
            if name == 'index':
                return super().__getattribute__('parent').mindex.index
            elif name == 'columns':
                return super().__getattribute__('parent').mcolumns.index
            elif name == 'name':
                return super().__getattribute__('parent').name.name
            elif name == 'iloc':
                self._update_super_index
                return super().__getattribute__('iloc')
            elif name == 'loc':
                self._update_super_index
                return super().__getattribute__('loc')
            else:
                return super().__getattribute__(name)
    return ValFrameBase