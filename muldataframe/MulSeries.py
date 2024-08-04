import pandas as pd
import muldataframe.cmm as cmm
# import muldataframe.util as util
from typing import Any
import numpy as np
import muldataframe as md
# import muldataframe.ValFrameBase as vfb
import muldataframe.ValFrameBase as vfb
import tabulate
tabulate.PRESERVE_WHITESPACE = True

#TODO: query for mulseries and muldataframe


class MulSeries:
    '''
    A multi-index series with the index being a pandas dataframe and the name a pandas series. It also has an underlying values series that is not directly accessible. Its values are the same as the values of the values series.

    Parameters
    -----------
    data: pandas.Series, array-like, Iterable, dict, or scalar value
        either a pandas Series or the same kind of data argument as required in the `pandas Series constructor <https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series>`_. The values series is constructed from the data argument.
    index: pandas.DataFrame
        If index is None, construct an empty index dataframe using the index of the values series as its index.
    name: pandas.Series, str
        If name is of str type, construct an empty name series using name as its name. If name is None, construct an empty name series using the name of the values series as its name.
    index_init: Literal['override'] | Literal['align']
        The option determins how to align the index of the index dataframe to the index of the values series. In the override mode, the index of the index dataframe overrides the index of the values series. This mode requires both indices' lengths to be the same. The align mode is only effective if the data argumnet implies an index and the index argument is not None. In this mode, the index of the index dataframe is used to index the values series constructed from the data argument. The resulting series is used as the final values series. It requires the index of the values series being uinque and the labels of the index dataframe's index exist in the index of the values series. By default, the constructor prioritizes the align mode if possible.
    index_copy: bool
        whether to create a copy of the index argument.
    name_copy: bool
        whether to create a copy of the name argument.

    Examples:
    ----------
    Construct a mulseries. Notice that the index of the dataframe and the index of the values series are the same and the name of the name series and the name of the values series are the same.
    
    >>> import pandas as pd
    >>> import muldataframe as md
    >>> index = pd.DataFrame([[1,2],[3,5],[3,6]],
                            index=['a','b','b'],
                            columns=['x','y'])
    >>> name = pd.Series(['g','h'],index=['e','f'], name='cc')
    >>> ms = md.MulSeries([1,2,3],index=index,name=name)
    >>> ms
    (3,)     f   h
             e   g
                cc
    -------  ------
       x  y     cc
    a  1  2  a   1
    b  3  5  b   2
    b  3  6  b   3
    '''
    # force pandas to return NotImplemented when using ops like +, * 
    # in the case of pd.Series + MulSeries.
    __pandas_priority__ = 10000
    def __init__(self,data,index:pd.DataFrame=None,
                 name:pd.Series|str|None=None,
                 index_init:cmm.IndexInit=None,
                 index_copy=True,name_copy=True):
       
        ss = data
        
        if isinstance(ss,dict):
            ss = pd.Series(ss)

        if isinstance(ss,pd.Series):
            index_init = 'align' if index_init is None else index_init
        else:
            index_init = 'override' if index_init is None else index_init
            ss = pd.Series(ss)

        if not isinstance(name,pd.Series):
            if isinstance(name,str):
                name = pd.Series([],name=name)
            else:
                name = pd.Series([],name=ss.name)
        else:
            name = name.copy() if name_copy else name

        ss, index = cmm.setMulIndex(ss,'index',index,index_init,index_copy)

        
        self.index = index
        '''
        The index dataframe. Use :doc:`MulSeries.mindex <mindex>` as an alias for this attribute.
        '''
        self.name = name
        '''
        The name series. Use :doc:`MulSeries.mname <mname>` as an alias for this attribute.
        '''
        # print(hasattr(self, 'index'))
        self.__ss = ValSeries(self,ss) # private

        self.iloc = cmm.Accessor(self._xloc_get_factory('iloc'),
                             self._xloc_set_factory('iloc'))
        '''
        Position-based indexing. It is the same as the `Series.iloc <https://pandas.pydata.org/docs/reference/api/pandas.Series.iloc.html>`_ of the values series except that it returns a MulSeries with the index dataframe properly sliced. If the return value is a scalar, it returns the scalar.
        '''
        self.loc = cmm.Accessor(self._xloc_get_factory('loc'),
                            self._xloc_set_factory('loc'))
        '''
        Label-based indexing. It is the same as `Series.loc <https://pandas.pydata.org/docs/reference/api/pandas.Series.loc.html>`_ of the values series except that it returns a MulSeries with the index dataframe properly sliced. If the return value is a scalar, it returns the scalar.
        '''
        self.mloc = cmm.Accessor(self._mloc_get,
                             self._mloc_set)
        '''
        Flexible hierachical indexing on the index dataframe. The slicer can be an array or a dict. Check introduction to mloc ??? for detailed usage.
        
        If an array is used, its length should be less than or equal to the columns length of the index dataframe. The hierarchical indexing order is from the leftmost column to the rightmost. Use ``None`` as ``:`` in the array to select all elements in a column.

        If a dict is used, its keys should be the column names of the index dataframe and its values the slicers on the columns. The hierachical indexing order is the insertion order of the keys in the dict. Although Python does not guanrantee the insertion order, it is generally preserved in most cases. Use the `OrderedDict <https://docs.python.org/3/library/collections.html#collections.OrderedDict>`_ class if you are really concerned about it.

        Examples
        ---------
        Array indexing:

        >>> import muldataframe as md
        >>> index = pd.DataFrame([['a','b','c'],
                                  [ 'g','b','f'],
                                  [ 'b','g','h']],
                           columns=['x','y','y'])
        >>> name = pd.Series(['a','b'],index=['e','f'],name='cc')
        >>> ms = md.MulSeries([1,2,3],index=index,name=name)
        >>> ms.mloc[[None,'b']]
        (2,)        e   a
                    f   b
                       cc
        ----------  ------
           x  y  y     cc
        0  a  b  c  0   1
        1  g  b  f  1   2
        >>> ms.mloc[['g',None,['h','f']]]
        2

        Dictionary indexing:

        >>> ms.mloc[{'y':['c','h'],'x':['b','a']}]
        (2,)        e   a
                    f   b
                       cc
        ----------  ------
           x  y  y     cc
        2  b  g  h  2   3
        0  a  b  c  0   1

        Note in the above example that if the index dataframe's columns have duplicate names, use the **last** column for indexing.

        '''

    def __repr__(self):
        cols = cmm.fmtSeries(self.name)
        vals = cmm.fmtSeries(self.ds)
        return tabulate.tabulate(
                [[str(self.index),str(vals)]],
               headers=[self.shape,
                        cmm.fmtColStr(cols,False)])
        # return 'ss:\n'+self.__ss.__repr__()+'\n\nindex:\n'+\
        #         self.index.__repr__()+'\n\nname:\n'+\
        #         self.name.__repr__()
    
    def __getattr__(self,name):
        if name == 'values':
            return self.__ss.values
        elif name == 'ss':
            return pd.Series(self.values.copy(),
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
        elif hasattr(np,name) and hasattr(getattr(np,name),'__call__'):
            def func(*args,**kwargs):
                return self.call(getattr(np,name),*args,**kwargs)
            return func

        
    def _hasVal(self):
        return self.__ss is not None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'ss':
            raise AttributeError(f"ss is a read-only attribute.")
        if name in ['index','mindex','midx']:
            cmm.checkSetIdxValue(self,'index',value)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __len__(self):
        return self.shape[0]

    def __eq__(self,other):
        return self.equals(other)
    
    def equals(self,other):
        '''
        Test whether two MulSeries are the same elements.

        Two MulSeries are equal only if their index dataframes, name series and value dataframes are equal. Use `Series.equals <https://pandas.pydata.org/docs/reference/api/pandas.Series.equals.html>`_ and `DataFrame.equals <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.equals.html#>`_ under the hood.

        Parameters
        ------------
        other : object
            The other object to be compared with the MulSeries. If the other is not a MulSeries, returns False.

        Returns
        ----------
        bool
            True for equality.
        '''
        if not isinstance(other,MulSeries):
            return False
        else:
            return self.ds.equals(other.ds) and self.index.equals(other.index) and self.name.equals(other.name)

    def copy(self):
        '''
        Create a deep copy of the mulseries.
        '''
        return MulSeries(self.__ss.copy().values,
                         index=self.index,
                         name=self.name.copy())
    
    def __iter__(self):
        '''
        Return an iterator of the values.

        Use `Series.__iter__ <https://pandas.pydata.org/docs/reference/api/pandas.Series.__iter__.html#pandas.Series.__iter__>`_ of the values series under the hood. 
        '''
        return self.__ss.__iter__()

    def __getitem__(self,key):
        new_ss = self.__ss[key]
        if(isinstance(new_ss,pd.Series)):
            idx_new = self.mindex.loc[key]
                # print('ok')
            ms = MulSeries(new_ss,
                                index=idx_new,
                                name=self.name,
                                index_init='override')
            return ms
        else:
            return new_ss
    
    def __setitem__(self,key, values):
        self.__ss[key] = values
    
    def _xloc_get_factory(self,attr):
        def _xloc_get(key):
            new_ss = getattr(self.__ss,attr)[key]
            if(isinstance(new_ss,pd.Series)):
                idx_new = getattr(self.mindex,attr)[key]
                ms = MulSeries(new_ss,
                                    index=idx_new,
                                    name=self.name,
                                    index_init='override')
                return ms
            else:
                return new_ss
        return _xloc_get
    
    def _xloc_set_factory(self,attr):
        def _xloc_set(key,values):
            getattr(self.__ss,attr)[key] = values
        return _xloc_set

    def _mloc(self,key):
        nx = cmm._mloc_idx(key,self.index)
        return nx
    
    def _mloc_get(self,key):
        if key == slice(None):
            return self.iloc[:]
        else:
            nx = self._mloc(key)
            return self.iloc[nx]
    
    def _mloc_set(self,key,values):
        if key == slice(None):
            self.iloc[:] = values
        else:
            nx = self._mloc(key)
            self.iloc[nx] = values

    def __fill_cols(self,col_fill,cols,inplace):
        if isinstance(col_fill,pd.Series) or \
            isinstance(col_fill,pd.DataFrame):
            if isinstance(col_fill,pd.Series):
                col_fill = pd.DataFrame(col_fill).transpose()
            col_fill = cmm.test_idx_eq(col_fill,cols,copy=False)

            col_fill = cmm.test_idx_eq(col_fill,self.name.index,indexType='columns',copy=False)
            
            mcols = pd.DataFrame(self.name).transpose()
            mcols = pd.concat([col_fill,mcols],axis=0)
            return mcols
        else:
            mcols = pd.DataFrame(self.name)
            for i, col in enumerate(cols):
                mcols.insert(i,col,col_fill,allow_duplicates=True)
            return mcols.transpose()
            
    def reset_index(self,columns=None, drop=False, 
                    inplace=False, col_fill=''):
        '''
        Reset the columns of the index dataframe as the columns of the mulseries.

        Parameters
        ----------
        columns : column name(s) of the index dataframe.
            If this argument is None, reset the index of the index dataframe. If the name of this index is None, it will be named as "primary_index". If "primary_index" exists in the primary columns, it will be named as "primary_index_1" and so on.
        drop : bool, default False
            Just reset the index, without inserting index dataframe's column(s) as column(s) in the new MulDataFrame.
        inplace : bool, default False
            Modify the MulSeries in place (do not create a new object).
        col_fill : object, default ''
            A scalar, a pandas Series or a pandas DataFrame to fill in the columns dataframe of the new MulDataFrame for the inserted values. If the argument is a Series or a DataFrame, its index should align with the index of the mulseries' name (which is a pandas series) in the same way as the align mode in the :doc:`constructor <mulseries>`.

        Returns
        --------
        MulSeries, MulDataFrame or None
            The return value behaves similarly to `Series.reset_index <https://pandas.pydata.org/docs/reference/api/pandas.Series.reset_index.html>`_.
        
        Examples
        ---------
        >>> import muldataframe as md
        >>> index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
        >>> name = pd.Series([5,7],
                        index=['f','g'],
                        name='c')
        >>> ms = MulSeries([1,8,9],index=index,name=name)
        >>> ms.reset_index()
        (3, 2)    g                7
                  f                5
                    primary_index  c
        --------  --------------------
           x  y     primary_index  c
        0  1  2   0             a  1
        1  3  6   1             b  8
        2  5  6   2             b  9

        Add a col_fill:

        >>> ss_fill = pd.Series([8,9],index=['g','f'],name='primary_index'))
        >>> ms.reset_index(col_fill=ss_fill)
        (3, 2)    g             8  7
                  f             9  5
                    primary_index  c
        --------  ---------------------
           x  y     primary_index  c
        0  1  2   0             a  1
        1  3  6   1             b  8
        2  5  6   2             b  9
        '''
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
                raise TypeError('Cannot reset_index inplace on a MulSeries to create a MulDataFrame')
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
                # print(df,mcols)
                return md.MulDataFrame(df.values,index=mkeep,columns=mcols,
                              index_copy=False,columns_copy=False)
            else:
                self2 = self.copy()
                self2.index = mkeep
                return self2

    def call(self,func,*args,**kwargs):
        '''
        Apply a function to the values series and returns the result as a scalar or a MulSeries with the index dataframe properly sliced.

        Parameters:
        -------------
        func : function
            A function applied to the values series of the MulSeries. Currently, the method only supports functions that return a scalar value or a pandas series with the same primary index (order can be different if there are no duplicate values in the primary index).
        \*args : positional arguments to the function
            The MulSeries is the 1st positional argument to the function. \*args are from the 2nd positional argument onwards.
        \*\*kwargs : keyword arguments to the function
            keyword arguments to the function.

        Returns
        -----------
        scalar or MulSeries
            If the return value is a MulSeries, it should have the same index dataframe as the caller.


        Examples
        ----------
        >>> import muldataframe as md
        >>> import numpy as np
        >>> index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
        >>> name = pd.Series([5,7],
                        index=['f','g'],
                        name='c')
        >>> ms = MulSeries([1,8,9],index=index,name=name)
        >>> ms.call(np.power,2)
        (3,)      g  7
                  f  5
                     c
        -------  ------
           x  y      c
        a  1  2  a   1
        b  3  6  b  64
        b  5  6  b  81
        '''
        # if 'unsafe' in kwargs and kwargs['unsafe']:
        # consider to change to self.ss to improve safety?
        args = list(args)
        if len(args) > 0 and (isinstance(args[0],MulSeries) or \
            isinstance(args[0],md.MulDataFrame)):
            args[0] = args[0].ds
        if len(args) > 0 and hasattr(md,'__pandas_priority__') \
            and args[0].__pandas_priority__ > self.__pandas_priority__:
            return NotImplemented
        # print(func,self,args)
        self.__ss._update_super_index()
        res = func(self.__ss,*args,**kwargs)
        errMsg = f'Currently, {self.__class__} only supports operators or functions that return a scalar value or a pandas series with the same primary index (order can be different if there are no duplicate values) in its .call() method.'
        # else:
        #     res = func(self.ss,*args,**kwargs)
        # print(res)
        if isinstance(res,pd.DataFrame):
            # print('dataframe',res)
            # have to raise error here because pandas does not support muldataframe in its __radd__ like functions.
            raise NotImplementedError(errMsg)
            # return NotImplemented
        elif isinstance(res,pd.Series):
            if res.shape[0] == self.shape[0]:
                if not res.index.equals(self.index.index):
                    try:
                        # res2 = self.loc[res.index]
                        new_idx = cmm.align_index_in_call(res.index,self,
                                                          'index')
                        if new_idx.shape[0] == self.shape[0]:
                            return MulSeries(res.values,
                                             index=new_idx,
                                             name=self.name,
                                             index_copy=False)
                        # if res2.shape[0] == self.shape[0]:
                        #     return res2
                        else:
                            raise NotImplementedError(errMsg)
                    except:
                        raise NotImplementedError(errMsg)
                else:
                    if res.name == self.name.name:
                        name = self.name.copy()
                    else:
                        name = pd.Series([],name=res.name)
                    return MulSeries(res,
                                    index=self.index,
                                    name=name,
                                    index_init='override')
            else:
                # print('shape',res.shape,self.shape)
                raise NotImplementedError(errMsg)
        else:
            return res

    def groupby(self,by=None,keep_primary=False,agg_mode:cmm.IndexAgg='same_only'):
        '''
        Group MulSeries by its index dataframe using a mapper or the index dataframe's columns.

        The function uses the `DataFrame.groupby(axis=0) <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby>`_ method of the index dataframe to create groups under the hood. The values of the MulSeries are grouped accordingly. It returns a :doc:`MulGroupBy <../groupby/indices>` object that contains information about the groups.

        Parameters
        ------------
        by : None, mapping, function, label, pd.Grouper or list of such
            Please refers to `DataFrame.groupby <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby>`_ for detailed information on this argument. The difference to the :code:`by` argument in DataFrame.groupby is that if it is None, uses the primary index to group the MulSeries.
        keep_primary : bool, default False
            Whether to keep primary index in the grouped index dataframes. If True, the primary index will be reset as a column and kept in the grouped dataframes.
        agg_mode : 'same_only', 'list','tuple'
            Determine how to aggregate column values in the index dataframe that are not the same in each group when calls numpy functions on or using the :doc:`call <../groupby/indices>` method of the MulGroupBy object.'same_only': only keep columns that have the same values within each group. 'list': put columns that do not have the same values within a group into a list. 'tuple': similar to 'list', but put them into a tuple.

        Returns
        -----------
        MulGroupBy
            A `MulGroupBy <../groupby/call>`_ object that contains information about the groups.
                

        Examples
        ------------
        >>> import muldataframe as md
        >>> import pandas as pd
        >>> index = pd.DataFrame([['a','b','c'],
                                  ['g','b','f'],
                                  ['b','g','h']],
                        columns=['x','y','z'])
        >>> name = pd.Series(['a','b'],index=['e','f'],name='cc')
        >>> ms = MulSeries([1,2,3],index=index,name=name)
        >>> for key, group in ms.groupby('y'):
        ...     print(key,'\\n',group)
        ...     break
        b
        (2,)        f  b
                    e  a
                       cc
        ----------  ------
           x  y  z     cc
        0  a  b  c  0   1
        1  g  b  f  1   2
        >>> ms.groupby('y').sum()
        (2,)    f   b
                e   a
                   cc
        ------  ------
           y       cc
        0  b    0   3
        1  g    1   3
        >>> ms.groupby('y',agg_mode='list').sum()
        (2,)                  f   b
                              e   a
                                 cc
        --------------------  ------
                x  y       z     cc
        0  [a, g]  b  [c, f]  0   3
        1       b  g       h  1   3
        '''
        return cmm.groupby(self,'index',by=by,
                           keep_primary=keep_primary,agg_mode=agg_mode)
    
    
    def drop_duplicates(self,keep='first', inplace=False):
        '''
        Return MulSeries with duplicate values removed. 
        
        It is similar to `Series.drop_duplciates <https://pandas.pydata.org/docs/reference/api/pandas.Series.drop_duplicates.html>`_ except it returns a MulSeries with the index dataframe properly sliced.

        Parameters
        -----------
        keep: {'first', 'last', False}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.
        inplace: bool, default False
            If True, performs operation inplace and returns None.
        
        Returns:
        ----------
        MulSeries or None
            If inplace=True, returns None. Otherwise, returns a MulSeries. The MulSeries' index dataframe is properly sliced according to removed values.

        '''
        bidx = self.__ss.duplicated(keep=keep)
        bidx_keep = ~bidx
        new_ss = self.__ss.loc[bidx_keep]

        if inplace:
            # primary_index = self.index.index
            # primary_columns = self.columns.index
            self.__ss = None
            self.index = self.index.loc[bidx_keep]
            self.__ss = ValSeries(self,new_ss)
        else:
            return MulSeries(new_ss.values,
                        index=self.index.loc[bidx_keep],
                        name=self.name)



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
    r_op_attr = '__r'+op+'__'
    setattr(MulSeries,r_op_attr,call_op_factory(r_op_attr))


# ops = []
# for op in ops:
#     op_attr = '__'+op+'__'
#     def call_op_factory(op_attr):
#         def call_op(self,other):
#             func = getattr(pd.Series,op_attr)
#             # print(op_attr,func)
#             return self.call(func,other)
#         return call_op
#     setattr(MulSeries,op_attr,call_op_factory(op_attr))


ValSeries = vfb.ValFrameBase_factory(pd.Series)
def _update_super_index(self):
    self.index = self.parent.mindex.index
    self.name = self.parent.name.name
ValSeries._update_super_index = _update_super_index

