import pandas as pd
from cmm import Accessor   
from MulSeries import MulSeries
from typing import Any


class MulDataFrame:
    def __init__(self, data, index=None, columns=None,
        index_init=None, columns_init=None):
        self._df = ValDataFrame(self,data)
        self.index = index
        self.columns = columns
        self.iloc = Accessor(self._iloc,2)
    
    def __repr__(self):
        return 'df:\n'+self._df.__repr__()+'\n\nindex:\n'+\
                self.index.__repr__()+'\n\ncolumns:\n'+\
                self.columns.__repr__()

    def _iloc(self,key):
        if isinstance(key,tuple):
            idx,col = key
        else:
            idx = key
            col = slice(None)
        new_df = self._df.iloc[idx,col]

        if isinstance(new_df,pd.DataFrame) or \
            isinstance(new_df,pd.Series):
            index = self.index.iloc[idx] 
            columns = self.columns.iloc[col]

            if isinstance(new_df,pd.DataFrame):
                return MulDataFrame(new_df,
                        index=index,
                        columns=columns)
            else:
                if isinstance(columns,pd.DataFrame):
                    index,columns = columns,index
                return MulSeries(new_df,
                            index=index,
                            name=columns)
        else:
            return new_df



class ValDataFrame(pd.DataFrame):
    def __init__(self,parent:MulDataFrame,df:pd.DataFrame):
        super().__init__(df)
        self.parent = parent

    def __getattribute__(self, name:str):
        if name == 'index':
            return super().__getattribute__('parent').index.index
        elif name == 'columns':
            return super().__getattribute__('parent').columns.index
        else:
            return super().__getattribute__(name)
   