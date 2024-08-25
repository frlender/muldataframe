advantages:
1. mloc, flexible hierchical indexing using dictionary.
2. ['a',None,['b','c']] returns a MulSeries or raises error, pd.MultiIndex returns a dataframe
3. md.mean(axis=) returns a MulSeries with the name being the funciton name 'mean'.
4. df.set_index does not work for df with both index and columns multi-index
5. df.reset_index puts the level names to the first row in columns Multi-index
6. With primary index, operations like add and div between MulDataFrames can be aligned like pandas.DataFrame. (TODO: compare to operations between pandas.DataFrame with multi-index???)
7. ... syntax
8. mloc dict syntax allows the change of the order of flexible hierachical indexing


An alternative multi-index DataFrame for Python and Pandas