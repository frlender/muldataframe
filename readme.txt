advantages:
1. mloc, flexible hierchical indexing using dictionary.
2. ['a',None,['b','c']] returns a MulSeries or raises error, pd.MultiIndex returns a dataframe
3. md.mean(axis=) returns a MulSeries with the name being the funciton name 'mean'.
4. df.set_index does not work for df with both index and columns multi-index
5. df.reset_index puts the level names to the first row in columns Multi-index