import muldataframe as md
import pandas as pd
# from muldataframe import MulSeries

# print(MulSeries)

# from circular.b import b

# import circular.a
# import circular.b as cb
# print(circular.a.A,cb.b,cb.z)

# def f2(a,*args,k=5,**kwargs):
#     print(a,args,k,kwargs)

# def f(a,*args,**kwargs):
#     f2(a,*args,**kwargs)

# print(f(8,k=9))

# index = pd.DataFrame([['a','b','c'],
#                                   ['g','b','f'],
#                                   ['b','g','h']],
#                         columns=['x','y','z'])
# name = pd.Series(['a','b'],index=['e','f'],name='cc')
# ms = md.MulSeries([1,2,3],index=index,name=name)

# print(ms.groupby('y').sum())
# print(ms.groupby('y',agg_mode='list').sum())



index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
columns = pd.DataFrame([[5,7],[3,6]],
                     index=['c','d'],
                     columns=['f','g'])
mf = md.MulDataFrame([[1,2],[8,9],[8,7]],index=index,
                  columns=columns)

# print(mf.groupby('y',axis=0).sum())
# print(mf)
print(mf.groupby('y',axis=0).sum(axis=1))


print(mf.groupby('f',axis=1).sum(axis=0))

# index = pd.DataFrame([[1,2,'d'],
#                           [3,6,'d'],
#                           [5,6,'e']],
#                      index=['a','b','b'],
#                      columns=['x','y','y'])
# columns = pd.DataFrame([[5,7],[3,6]],
#                     index=['c','d'],
#                     columns=['f','g'])
# mf = md.MulDataFrame([[1,2],[8,9],[8,10]],
#     index=index,columns=columns)

# mf.index = mf.index.iloc[:,:-1]

# print(mf.groupby('y',agg_mode='list').mean())
# print(mf.groupby('y',agg_mode='list',keep_primary=True).mean())
# print(mf.groupby('y',agg_mode='list').mean())
