import pandas as pd
import muldataframe as md
from typing import Any
import muldataframe.cmm as cmm
import muldataframe.ValFrameBase as vfb
# import muldataframe.util as util

class MulDataFrame:
    def __init__(self, data, index=None, columns=None,
        index_init:cmm.IndexInit=None, 
        columns_init:cmm.IndexInit=None,
        both_init:cmm.IndexInit=None):

        if index_init is None and columns_init is None:
            index_init = both_init
            columns_init = both_init

        if isinstance(data,pd.DataFrame) or \
            isinstance(data,dict):
            columns_init = 'align' if columns_init is None else columns_init
        else:
            columns_init = 'override' if columns_init is None else columns_init
        
        if isinstance(data,pd.DataFrame) or \
            isinstance(data,pd.Series):
            index_init = 'align' if index_init is None else index_init
        else:
            index_init = 'override' if index_init is None else index_init

        if not isinstance(data,pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
        
        df, index = cmm.setMulIndex(df,'index',index,index_init)
        df, columns = cmm.setMulIndex(df,'columns',columns,columns_init)


        self.index = index
        self.columns = columns
        self.__df = ValDataFrame(self,df)

        # super(ValDataFrame,self.__df).index and self.__df.index
        # are not guaranteed to be the same. Users should always assume that
        # self.index and self.__df.index are different.
        # But they are synchronized when ValDataFrame.iloc/loc/[] are called.

        self.iloc = cmm.Accessor(self._xloc_get_factory('iloc'),
                             self._xloc_set_factory('iloc'),2)
        self.loc = cmm.Accessor(self._xloc_get_factory('loc'),
                             self._xloc_set_factory('loc'),2)
        self.mloc = cmm.Accessor(self._mloc_get,
                             self._mloc_set,2)
        
    
    def __repr__(self):
        return 'df:\n'+self.__df.__repr__()+'\n\nindex:\n'+\
                self.index.__repr__()+'\n\ncolumns:\n'+\
                self.columns.__repr__()
    
    def __getattr__(self,name):
        if name == 'values':
            return self.__df.values
        elif name == 'df':
            return pd.DataFrame(self.__df.copy().values,
                             index=self.index.index.copy(),
                             columns=self.columns.index.copy())
        elif name in ['mindex','mcolumns']:
            return getattr(self,name.lstrip('m'))
        elif name == 'shape':
            return self.__df.shape
        elif name == 'ds':
            # values are not copied version
            return pd.DataFrame(self.values,
                             index=self.index.index.copy(),
                             columns=self.columns.index.copy(),
                             copy=False)
        
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ['mindex','mcolumns']:
            super().__setattr__(name.lstrip('m'), value)
        else:
            super().__setattr__(name, value)

    
    def __eq__(self,other):
        return self.equals(other)
    
    def equals(self,other):
        if not isinstance(other,MulDataFrame):
            return False
        else:
            return self.ds.equals(other.ds) and \
                self.index.equals(other.index) and \
                self.columns.equals(other.columns)

    def copy(self):
        return MulDataFrame(self.__df.copy().values,
                             index=self.index.copy(),
                             name=self.columns.copy())
    
    def _xloc_get_factory(self,attr):
        def _xloc_get(key):
            if isinstance(key,tuple):
                idx,col = key
            else:
                idx = key
                col = slice(None)
            new_df = getattr(self.__df,attr)[idx,col]

            if isinstance(new_df,pd.DataFrame) or \
                isinstance(new_df,pd.Series):
                index = getattr(self.index,attr)[idx] 
                columns = getattr(self.columns,attr)[col]

                if isinstance(new_df,pd.DataFrame):
                    return MulDataFrame(new_df,
                            index=index,
                            columns=columns,
                            both_init='override')
                else:
                    if isinstance(columns,pd.DataFrame):
                        index,columns = columns,index
                    return md.MulSeries(new_df,
                                index=index,
                                name=columns,
                                both_init='override')
            else:
                return new_df
        return _xloc_get
    
    def _get_indices(self,key):
        if isinstance(key,tuple):
                idx,col = key
        else:
            idx = key
            col = slice(None)
        return idx, col

    def _xloc_set_factory(self,attr):
        def _xloc_set(self,key,values):
            idx, col = self._get_indices(key)
            getattr(self.__df,attr)[idx,col] = values
        return _xloc_set
    
    def _mloc(self,key):
        idx, col = self._get_indices(key)
        if idx == slice(None):
            nx_idx = idx
        else:
            nx_idx = cmm._mloc_idx(idx,self.mindex)
        if col == slice(None):
            nx_col = col
        else:
            nx_col = cmm._mloc_idx(col,self.mcolumns)
        return nx_idx, nx_col
    
    def _mloc_get(self,key):
        nx_idx, nx_col = self._mloc(key)
        return self.iloc[nx_idx,nx_col]
    
    def _mloc_set(self,key,value):
        nx_idx, nx_col = self._mloc(key)
        self.iloc[nx_idx,nx_col] = value

    
    @classmethod
    def _mloc_to_primary(cls,key,mindex):
        nx = cmm._mloc_idx(key,mindex)
        subset = mindex.index[nx]
        return subset
        

    def set_index(self,keys,mloc=False,drop=False,inplace=False):
        if mloc:
            keys = self._mloc_to_primary(keys,self.mcolumns)
        sub_df = self.__df[keys]
        # print(self.mindex,sub_df)
        new_mindex =  pd.concat([self.mindex,sub_df],axis=1)
        if inplace:
            self.mindex = new_mindex
            if drop:
                self.__df.drop(keys,axis=1,inplace=True)
                self.mcolumns.drop(keys,axis=0,inplace=True)
        else:
            new_df = self.df
            new_mcolumns = self.mcolumns.copy()
            if drop:
                new_df.drop(keys,axis=1,inplace=True)
                new_mcolumns = new_mcolumns.drop(keys,axis=0)
            return MulDataFrame(new_df.values,
                        index=new_mindex,
                        columns=new_mcolumns)


    def drop_duplicates(self,subset=None,mloc=False,
                        keep='first',inplace=False):
        if mloc:
            subset = self._mloc_to_primary(subset,self.mcolumns)

        # print(super(ValDataFrame,self.__df).index)
        # self.__df.index = list(range(self.shape[0]))
        # self.__df.columns = list(range(self.shape[1]))
        # self.__df.index = self.__df.index
        # self.__df.columns = self.__df.columns
        bidx = self.__df.duplicated(subset=subset,keep=keep)
        bidx_keep = ~bidx
        new_df = self.__df.loc[bidx_keep]

        if inplace:
            # primary_index = self.index.index
            # primary_columns = self.columns.index
            self.index = self.index.loc[bidx_keep]
            self.__df = ValDataFrame(self,new_df)
        else:
            return MulDataFrame(new_df.values,
                        index=self.index.loc[bidx_keep],
                        columns=self.columns.copy())
    
    def iterrows(self):
        for i in range(self.shape[0]):
            yield self.mindex.iloc[i], self.iloc[i]
    
    def call(self,func,*args,**kwargs):
        new_df = func(self.__df,*args,**kwargs)
        func_name = func.__name__
        if isinstance(new_df,pd.DataFrame):
            if new_df.index.equals(self.mindex.index) and \
                new_df.columns.equals(self.mcolumns.index):
                return MulDataFrame(new_df.values,
                        index=self.mindex.copy(),
                        columns=self.mcolumns.copy())
            else:
                return NotImplemented
        elif isinstance(new_df,pd.Series):
            if new_df.index.equals(self.mindex.index) and \
                not new_df.columns.equals(self.mcolumns.index):
                return md.MulSeries(new_df.values,index=self.mindex.copy(),name=func_name)
            elif new_df.columns.equals(self.mcolumns.index) and \
                not new_df.index.equals(self.mindex.index):
                return md.MulSeries(new_df.values,index=self.mcolumns.copy(),name=func_name)
            elif new_df.index.equals(self.mindex.index) and \
                ('axis' in kwargs and kwargs['axis'] == 1):
                return md.MulSeries(new_df.values,index=self.mindex.copy(),name=func_name)
            elif new_df.columns.equals(self.mcolumns.index) and \
                ('axis' in kwargs and kwargs['axis'] == 0):
                return md.MulSeries(new_df.values,index=self.mcolumns.copy(),name=func_name)
            else:
                return NotImplemented
        else:
            return new_df
        
    def groupby(self,by=None,axis=0,agg_mode:cmm.IndexAgg='same_only',
                keep_primary=False,sort=True):
        indexType = 'index' if axis == 0 else 'columns'
        return cmm.groupby(self,indexType,by=by,
                           keep_primary=keep_primary,agg_mode=agg_mode,
                           sort=sort)

ops = ['add','sub','mul','div','truediv','floordiv','mod','pow']
for op in ops:
    op_attr = '__'+op+'__'
    def call_op_factory(op_attr):
        def call_op(self,other):
            func = getattr(pd.DataFrame,op_attr)
            # print(op_attr,func)
            return self.call(func,other)
        return call_op
    setattr(MulDataFrame,op_attr,call_op_factory(op_attr))

ValDataFrame = vfb.ValFrameBase_factory(pd.DataFrame)

# class MulDataFrameGroupby[G,M](cmm.MulGroupBy):
#     def call(self,func,*args,**kwargs):
#         res = None
#         for i,(k,gp) in enumerate(self):
#             val = gp.call(func,*args,**kwargs)
#             if isinstance(val,M):
#                 return NotImplemented
#             index = util.aggregate_index(i,gp.index,self.index_agg)



# class ValDataFrame(pd.DataFrame):
    # def __init__(self,parent:MulDataFrame,df:pd.DataFrame):
    #     super().__init__(df)
    #     self.parent = parent
    #     self._iloc_accessor = cmm.Accessor(self._iloc,
    #                              self._set_iloc,len(self.shape))
    #     self._loc_accessor = cmm.Accessor(self._loc,
    #                              self._set_loc,len(self.shape))


    # def _update_super_index(self):
    #     self.index = self.parent.mindex.index
    #     self.columns = self.parent.mcolumns.index

    # def __getitem__(self,key):
    #     # # Without calling self._update_super_index(),
    #     # # the expression below uses self.parent.mindex.index
    #     # # to index but returns a dataframe or series with
    #     # # super(ValDataFrame,self).index.
    #     # res = super().__getitem__(key)
    #     self._update_super_index()
    #     return super().__getitem__(key)
        
    # def _iloc(self,key):
    #     self._update_super_index()
    #     return super().iloc[key]
    
    # def _set_iloc(self,key,value):
    #     self._update_super_index()
    #     super().iloc[key] = value
    
    # def _loc(self,key):
    #     self._update_super_index()
    #     return super().loc[key]
    
    # def _set_loc(self,key,value):
    #     self._update_super_index()
    #     super().loc[key] = value


    # def __getattribute__(self, name:str):
    #     if name == 'index':
    #         return super().__getattribute__('parent').index.index
    #     elif name == 'columns':
    #         return super().__getattribute__('parent').columns.index
    #     elif name == 'iloc':
    #         return super().__getattribute__('_iloc_accessor')
    #     elif name == 'loc':
    #         return super().__getattribute__('_loc_accessor')
    #     else:
    #         return super().__getattribute__(name)
   