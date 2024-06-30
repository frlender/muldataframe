index = pd.DataFrame([[1,2],[3,4],[5,6]],
                     index=['a','b','c'],
                     columns=['x','y'])
columns = pd.DataFrame([[5,7],[3,6]],
                     index=['c','d'],
                     columns=['f','g'])
md = MulDataFrame([[1,2],[3,5],[8,10]],index=index,
                  columns=columns)