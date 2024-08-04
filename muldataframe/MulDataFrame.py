import pandas as pd
import muldataframe as md
from typing import Any
import muldataframe.cmm as cmm
import muldataframe.ValFrameBase as vfb
import numpy as np
import tabulate
tabulate.PRESERVE_WHITESPACE = True
# import muldataframe.util as util

class MulDataFrame:
    '''
    A multi-index dataframe with the index and the columns being pandas dataframes. It also has an underlying values dataframe that is not directly accessible. Its values are the same as the values of the values dataframe.

    Parameters
    -----------
    data : pandas.DataFrame, ndarray (structured or homogeneous), Iterable, dict
        either a pandas DataFrame or the same kind of data argument as required in the `pandas.DataFrame constructor <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame>`_. The values dataframe is constructed from the data argument.
    index : pandas.DataFrame
        If index is None, construct an empty index dataframe using the index of the values dataframe as its index.
    columns : pandas.DataFrame
        If columns is None, construct an empty columns dataframe using the columns of the values dataframe as its index.
    index_init : Literal['override'] | Literal['align']
        The option determins how to align the index of the index dataframe to the index of the values dataframe. In the override mode, the index of the index dataframe overrides the index of the values dataframe. This mode requires both indices' lengths to be the same. The align mode is only effective if the data argumnet implies an index and the index argument is not None. In this mode, the index of the index dataframe is used to index the values dataframe constructed from the data argument. The resulting dataframe is used as the final values dataframe. It requires the index of the values dataframe being uinque and the labels of the index dataframe's index exist in the index of the values dataframe. By default, the constructor prioritizes the align mode if possible.
    columns_init : Literal['override'] | Literal['align']
        The option determins how to align the index of the columns dataframe to the columns of the values dataframe. In the override mode, the index of the columns dataframe overrides the columns of the values dataframe. This mode requires both indices' lengths to be the same. The align mode is only effective if the data argumnet implies a columns index and the columns argument is not None. In this mode, the index of the columns dataframe is used to index the columns of the values dataframe constructed from the data argument. The resulting dataframe is used as the final values dataframe. It requires the columns of the values dataframe being uinque and the labels of the columns dataframe's index exist in the columns of the values dataframe. By default, the constructor prioritizes the align mode if possible.
    both_init : Literal['override'] | Literal['align']
        It overrides index_init and columns_init with the same value.
    index_copy : bool
        whether to create a copy of the index argument.
    columns_copy : bool
        whether to create a copy of the columns argument.
    both_copy : bool
        It overrides index_copy and columns_copy with the same value.

    Examples
    ----------
    Construct a muldataframe. Notice that the index of the index muldataframe and the index of the values muldataframe are the same and the index of the columns dataframe and the columns of the values dataframe are the same.

    >>> import pandas as pd
    >>> import muldataframe as md
    >>> index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
    >>> columns = pd.DataFrame([[5,7],[3,6]],
                        index=['c','d'],
                        columns=['f','g'])
    >>> md = MulDataFrame([[1,2],[8,9],[8,10]],index=index,columns=columns)
    >>> md
    (3, 2)    g  7   6
              f  5   3
                 c   d
    --------  ---------
       x  y      c   d
    a  1  2   a  1   2
    b  3  6   b  8   9
    b  5  6   b  8  10
    '''
    __pandas_priority__ = 10000
    def __init__(self, data, index=None, columns=None,
        index_init:cmm.IndexInit=None, 
        columns_init:cmm.IndexInit=None,
        both_init:cmm.IndexInit=None,
        index_copy=True,
        columns_copy=True,
        both_copy=None):

        if both_init is not None:
            index_init = both_init
            columns_init = both_init
        
        if both_copy is not None:
            index_copy = both_copy
            columns_copy = both_copy

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
        return tabulate.tabulate(
                [[str(self.index),str(self.ds)]],
               headers=[self.shape,
                        cmm.fmtColStr(self.mcols)])

    def __iter__(self):
        '''
        Iterate over info axis of the values dataframe.

        Use `DataFrame.__iter__ <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.__iter__.html>`_ of the values dataframe under the hood. 
        '''
        return self.__df.__iter__()
    
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
    

    def transpose(self,inplace=False):
        if inplace:
            __df = self.__df
            self.__df = None
            self.index, self.columns = self.columns, self.index
            self.__df = ValDataFrame(self,__df.values.T)
        else:
            return MulDataFrame(self.values.copy().T,
                index=self.columns,columns=self.index)

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
        
    def __fill_cols(self,col_fill,cols,inplace):
        if isinstance(col_fill,pd.Series) or \
            isinstance(col_fill,pd.DataFrame):
            if isinstance(col_fill,pd.Series):
                col_fill = pd.DataFrame(col_fill).transpose()
            col_fill = cmm.test_idx_eq(col_fill,cols,copy=False)
            col_fill = cmm.test_idx_eq(col_fill,self.columns.columns,
                    indexType='columns',copy=False)
            if inplace:
                self.mcols = mcols
            else:
                mcols = pd.concat([col_fill,self.mcols],axis=0)
                return mcols
        else:
            mcols = self.mcols.transpose()
            for i, col in enumerate(cols):
                mcols.insert(i,col,col_fill,allow_duplicates=True)
            if inplace:
                self.mcols = mcols.transpose()
            else:
                return mcols.transpose()

    def reset_index(self,columns=None, drop=False, 
                    inplace=False, col_fill=''):
        if columns is None:
            if self.mindex.index.name is None:
                indexName = cmm.get_index_name('index',
                                               self.mindex.columns)
            else:
                indexName = self.mindex.index.name
            mselect = pd.DataFrame(self.mindex.index,
                                   index=self.mindex.index,
                                   columns=[indexName])
        else:
            mselect = self.mindex[columns]
            if isinstance(mselect,pd.Series):
                mselect = pd.DataFrame(mselect)
        if inplace:
            if columns is not None:
                self.mindex.drop(columns,axis=1,inplace=True)
            if not drop:
                ds = self.ds
                self.__df = None
                self.__fill_cols(col_fill,mselect.columns,True)
                ds = pd.concat([mselect,ds],axis=1,copy=False)
                self.__df = ValDataFrame(self,ds)
            if columns is None:
                self.mindex.index = range(self.shape[0])

        else:
            if columns is not None:
                mkeep = self.mindex.drop(columns,axis=1)
            else:
                mkeep = self.mindex.copy()
                mkeep.index = range(self.shape[0])
            if not drop:
                mcols = self.__fill_cols(col_fill,mselect.columns,
                                         False)
                df = pd.concat([mselect,self.ds],axis=1)
            else:
                mcols = self.mcols.copy()
                df = self.df
            mf = MulDataFrame(df.values,index=mkeep,columns=mcols,
                              index_copy=False,columns_copy=False)
            return mf
            

    def drop_duplicates(self,subset=None,mloc=None,
                        keep='first',inplace=False):
        if subset is None and mloc is None:
            raise ValueError('one of the subset or the mloc argument must be set.')
        if mloc:
            subset = self._mloc_to_primary(mloc,self.mcolumns)

       
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

