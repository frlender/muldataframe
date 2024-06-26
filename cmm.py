
import pandas as pd

class Accessor:
    def __init__(self,func,max_idx=1):
        self.func = func
        self.max_idx = max_idx
        pass
    
    def __getitem__(self,key):
        if isinstance(key,tuple) and len(key) > self.max_idx:
            raise ValueError(f"Too many indices. Only {self.max_idx} {'indices are' if self.max_idx > 1 else 'index is'} allowed")
        return self.func(key)
    



def _mloc_idx(key,df):
    if not isinstance(key,tuple):
        key = (key,)
    for i, idx in enumerate(key):
        col_idx = pd.Series(range(df.shape[0]),
                            index=df.iloc[:,i].values)
    