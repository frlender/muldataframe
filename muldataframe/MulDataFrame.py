import pandas as pd
import muldataframe as md
from typing import Any
import muldataframe.cmm as cmm
import muldataframe.ValFrameBase as vfb
import numpy as np
# import muldataframe.util as util

class MulDataFrame:
    __pandas_priority__ = 10000
    def __init__(self, data, index=None, columns=None,
        index_init:cmm.IndexInit=None, 
        columns_init:cmm.IndexInit=None,
        both_init:cmm.IndexInit=None,
        index_copy=True,
        columns_copy=True,
        both_copy=True):

        if index_init is None and columns_init is None:
            index_init = both_init
            columns_init = both_init
        
        if both_copy:
            index_copy = True
            columns_copy = True
        else:
            index_copy = False
            columns_copy = False

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
        
        # print('-----',df,index,index_init)
        df, index = cmm.setMulIndex(df,'index',index,index_init,index_copy)
        df, columns = cmm.setMulIndex(df,'columns',columns,columns_init,columns_copy)


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
        
    def _hasVal(self):
        return self.__df is not None

    def __repr__(self):
        return 'df:\n'+self.__df.__repr__()+'\n\nindex:\n'+\
                self.index.__repr__()+'\n\ncolumns:\n'+\
                self.columns.__repr__()
    
    def __getattr__(self,name):
        if name == 'values':
            return self.__df.values
        elif name == 'df':
            return pd.DataFrame(self.__df.copy().values,
                             index=self.index.index,
                             columns=self.columns.index)
        elif name in ['mindex','midx']:
            return self.index
        elif name in ['mcolumns','mcols']:
            return self.columns
        elif name == 'shape':
            return self.__df.shape
        elif name == 'ds':
            # values are not copied version
            return pd.DataFrame(self.values,
                             index=self.index.index,
                             columns=self.columns.index,
                             copy=False)
        elif hasattr(np,name) and hasattr(getattr(np,name),'__call__'):
            def func(*args,**kwargs):
                return self.call(getattr(np,name),*args,**kwargs)
            return func
        
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ['index','mindex','midx']:
            name = 'index'
            cmm.checkSetIdxValue(self,name,value)
            super().__setattr__(name, value)
        elif name in ['columns','mcolumns','mcols']:
            name = 'columns'
            cmm.checkSetIdxValue(self,name,value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __len__(self):
        return self.shape[0]
    
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
                             index=self.index,
                             columns=self.columns)
    def _get_indices(self,key):
        if isinstance(key,tuple):
                idx,col = key
        else:
            idx = key
            col = slice(None)
        return idx, col
    
    def __getitem__(self,key):
        # print('--get-item',md)
        new_df = self.__df[key]
        if  isinstance(new_df,pd.DataFrame):
            new_mcols = self.mcolumns.loc[key]
            mx = MulDataFrame(new_df.values,
                                index=self.index,
                                columns=new_mcols,
                                columns_copy=False)
            return mx
        elif isinstance(new_df,pd.Series):
            new_mcols = self.mcolumns.loc[key]
                # print('ok')
            ms = md.MulSeries(new_df.values,
                                index=self.index,
                                name=new_mcols,
                                name_copy=False)
            return ms
        else:
            raise ValueError
        
    
    def __setitem__(self,key, values):
        self.__df[key] = values

    def _xloc_get_factory(self,attr):
        def _xloc_get(key):
            idx, col = self._get_indices(key)
            new_df = getattr(self.__df,attr)[idx,col]

            if isinstance(new_df,pd.DataFrame) or \
                isinstance(new_df,pd.Series):
                index = getattr(self.index,attr)[idx] 
                columns = getattr(self.columns,attr)[col]

                if isinstance(new_df,pd.DataFrame):
                    return MulDataFrame(new_df,
                            index=index,
                            columns=columns,
                            both_init='override',
                            both_copy=False)
                else:
                    if isinstance(columns,pd.DataFrame):
                        index,columns = columns,index
                    return md.MulSeries(new_df,
                                index=index,
                                name=columns,
                                index_init='override',
                                index_copy=False,
                                name_copy=False)
            else:
                return new_df
        return _xloc_get
    
    
    def _xloc_set_factory(self,attr):
        def _xloc_set(key,values):
            # print('===dddddd===',key,values)
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
        

    def set_index(self,keys=None,mloc=None,drop=True,inplace=False):
        if keys is None and mloc is None:
            raise ValueError('one of the keys or the mloc argument must be set.')
        if keys is None and mloc is not None:
            keys = self._mloc_to_primary(mloc,self.mcolumns)
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

    def reset_index():
        pass

    def drop_duplicates(self,subset=None,mloc=None,
                        keep='first',inplace=False):
        if subset is None and mloc is None:
            raise ValueError('one of the subset or the mloc argument must be set.')
        if mloc:
            subset = self._mloc_to_primary(mloc,self.mcolumns)

        # print(super(ValDataFrame,self.__df).index)
        # self.__df.index = list(range(self.shape[0]))
        # self.__df.columns = list(range(self.shape[1]))
        # self.__df.index = self.__df.index
        # self.__df.columns = self.__df.columns
        bidx = self.__df.duplicated(subset=subset,keep=keep)
        bidx_keep = ~bidx
        new_df = self.__df.loc[bidx_keep]

        if inplace:
            # Run "self.__df = ValDataFrame(self,new_df)"
            # before "self.index = ..." reports error
            # I don't know why. Possibly due to 
            # some mechanisms in the pandas library
            # that forces index to be consistent.
            self.__df = None
            self.index = self.index.loc[bidx_keep]
            self.__df = ValDataFrame(self,new_df)
        else:
            return MulDataFrame(new_df.values,
                        index=self.index.loc[bidx_keep],
                        columns=self.columns,
                        index_copy=False)
    
    def iterrows(self):
        for i in range(self.shape[0]):
            yield (self.mindex.iloc[i], self.iloc[i])
    
    def call(self,func,*args,**kwargs):
        args = list(args)
        if len(args)>0 and (isinstance(args[0],md.MulSeries) or \
            isinstance(args[0],MulDataFrame)):
            args[0] = args[0].ds
        
        if len(args) > 0 and hasattr(md,'__pandas_priority__') \
            and args[0].__pandas_priority__ > self.__pandas_priority__:
            return NotImplemented

        self.__df._update_super_index()
        new_df = func(self.__df,*args,**kwargs)
        func_name = func.__name__
        if isinstance(new_df,pd.DataFrame):
            if new_df.shape[0] != self.shape[0] or \
                new_df.shape[1] != self.shape[1]:
                raise NotImplementedError
            
            new_mindex = cmm.align_index_in_call(new_df.index,self,'index')
            new_mcols = cmm.align_index_in_call(new_df.columns,self,'columns')
            return  MulDataFrame(new_df.values,
                        index=new_mindex,columns=new_mcols,
                        index_copy=False,columns_copy=False)
            
            # if new_df.index.equals(self.mindex.index) and \
            #     new_df.columns.equals(self.mcolumns.index):
            #     return MulDataFrame(new_df.values,
            #             index=self.mindex.copy(),
            #             columns=self.mcolumns.copy())
            # else:
            #     return NotImplemented
        elif isinstance(new_df,pd.Series):
            if new_df.shape[0] == self.shape[0] and (
                new_df.shape[0] != self.shape[1] or 
                ('axis' in kwargs and kwargs['axis'] == 1) ):
                new_idx = cmm.align_index_in_call(new_df.index,self,'index')
                return md.MulSeries(new_df.values,index=new_idx,name=func_name,
                                    index_copy=False)
            elif new_df.shape[0] == self.shape[1] and  (
                 new_df.shape[0] != self.shape[0] or 
                ('axis' in kwargs and kwargs['axis'] == 0) ):
                new_idx = cmm.align_index_in_call(new_df.index,self,'columns')
                return md.MulSeries(new_df.values,index=new_idx,name=func_name,
                                    index_copy=False)
            else:
                raise NotImplementedError
        else:
            return new_df
        

    def groupby(self,by=None,axis=0,agg_mode:cmm.IndexAgg='same_only',
                keep_primary=False,sort=True):
        indexType = 'index' if axis == 0 else 'columns'
        return cmm.groupby(self,indexType,by=by,
                           keep_primary=keep_primary,agg_mode=agg_mode,
                           sort=sort)
    

    def __query_index(self,df,expr,**kwargs):
        col = '__@$&idx'
        df[col] = list(range(df.shape[0]))
        df2 = df.query(expr,**kwargs)
        return df2[col].tolist()

    def query(self,values=None,index=None,columns=None,
              **kwargs):
        if 'inplace' in kwargs:
            inplace = kwargs['inplace']
        else:
            inplace = False
        kwargs['inplace']=False
        if values is not None:
            valIdx = self.__query_index(self.df,values,**kwargs)
        if index is not None:
            idxIdx = self.__query_index(self.mindex.copy(),index,**kwargs)
        if columns is not None:
            colIdx = self.__query_index(self.mcolumns.copy(),columns,**kwargs)
        
        if values is not None and index is not None:
            rowIdx = []
            for i in range(self.shape[0]):
                if i in valIdx and i in idxIdx:
                    rowIdx.append(i)
        elif values is not None and index is None:
            rowIdx = valIdx
        elif index is not None and values is None:
            rowIdx = idxIdx
        else:
            rowIdx = slice(None)

        colIdx = colIdx if columns is not None else slice(None)
        if not inplace:
            return self.iloc[rowIdx,colIdx]
        else:
            self.__df = self.__df.iloc[rowIdx,colIdx]
            self.index = self.index.iloc[rowIdx]
            self.columns = self.columns.iloc[colIdx]

    def __melt_prefix(self,mindex,mcolumns,prefix):
        cmm_labels = list(set(mindex.columns).intersection(mcolumns.columns))
        if len(cmm_labels) > 0:
            if prefix is True:
                mindex.columns = \
                    [f'x_{label}' if label in cmm_labels else label 
                        for label in mindex.columns]
                mcolumns.columns = \
                    [f'y_{label}' if label in cmm_labels else label 
                        for label in mcolumns.columns]
            else:
                mindex.columns = \
                    [prefix('index',label) if label in cmm_labels else label 
                        for label in mindex.columns]
                mcolumns.columns = \
                    [prefix('columns',label) if label in cmm_labels else label 
                        for label in mcolumns.columns]
        return mindex,mcolumns
                    
    def melt(self,prefix=None,value_name='value',
             ignore_primary_index=False,
             ignore_primary_columns=False):
        if ignore_primary_index:
            mindex = self.index.copy()
            mindex.index = list(range(mindex.shape[0]))
        else:
            mindex = self.index.reset_index()
        
        if ignore_primary_columns:
            mcolumns = self.columns.copy()
            mcolumns.index = list(range(mcolumns.shape[0]))
        else:
            mcolumns = self.columns.reset_index()

        
        if prefix is not None:
            mindex,mcolumns = self.__melt_prefix(mindex,mcolumns,prefix)

        numIdx = list(range(self.shape[0]*self.shape[1]))
        df = pd.DataFrame(index=numIdx,
                          columns=mindex.columns.tolist()+
                          mcolumns.columns.tolist()+[value_name])
        col_len = self.shape[1]
        for i, (_,row) in enumerate(self.__df.iterrows()):
            mindex_sub = mindex.iloc[[i]*col_len]
            rstart = i*col_len
            rend = i*col_len + col_len
            cRowMeta = range(mindex.shape[1])
            cColMeta = range(mindex.shape[1],
                             mindex.shape[1]+mcolumns.shape[1])
            df.iloc[rstart:rend,cRowMeta] = mindex_sub.values
            df.iloc[rstart:rend,cColMeta] = \
                mcolumns.values
            df.iloc[rstart:rend,-1] = row.values
        
        return df


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
    r_op_attr = '__r'+op+'__'
    setattr(MulDataFrame,r_op_attr,call_op_factory(r_op_attr))

ValDataFrame = vfb.ValFrameBase_factory(pd.DataFrame)

