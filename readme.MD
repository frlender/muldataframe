# MulDataFrame
## Towards a more intuitive multi-index DataFrame

*"A multi-index is just a DataFrame, period."*

Have you found the multi-index in pandas difficult to use? With unexpected behaviors? Do you want to get rid of the long and hard-to-remember methods like get_level_values() and pd.IndexSlice()? Have you wondered why the multi-index is so similar to a dataframe but is not one? Have you been confused with the difference between levels and columns?

If you answered yes to any of these questions, then MulDataFrame might be right for you. MulDataFrame uses pandas dataframes as index and columns, which means you can manipulate them with all the familiar methods of the DataFrame class and no more. 



A MulDataFrame object consists of three pandas dataframes: an index dataframe, a columns dataframe and a values dataframe. The index of the index dataframe and the index of the columns dataframe are guaranteed to be the same as the index and the columns of the values dataframe. I'll call them the primary index and the primary columns.

Because of the primary index and columns, you can use `__getitem__`, `iloc` and `loc` on a muldataframe exactly as on its values dataframe, except that the return value is a muldataframe with its index and columns properly sliced. You can easily change the primary index to another column in the index dataframe by calling the `set_index()` method of the index dataframe like `md.index.set_index(col_name, inplace=True)`, where `md` is a muldataframe. 

`md.mindex`/`md.midx` and `md.mcolumns`/`md.mcols` are implemented as alias for for `md.index` and `md.columns` to help distinguish between a multi-index and a regular index.


