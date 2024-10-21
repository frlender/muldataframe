import muldataframe as md
import pandas as pd
import numpy as np
# from muldataframe import MulSeries

def get_data():
    index = pd.DataFrame([[1,2],[3,6],[5,6]],
                     index=['a','b','b'],
                     columns=['x','y'])
    columns = pd.DataFrame([[5,7],[3,6]],
                        index=['c','d'],
                        columns=['f','g'])
    mf = md.MulDataFrame([[1,2],[8,9],[8,7]],index=index,
                    columns=columns)
    return mf,index,columns

mf,index,columns = get_data()
mf.index.iloc[2] = [3,6]
print('\n',mf.groupby(['x','y']).mean())


# df = mf.df
# df.index = pd.MultiIndex.from_frame(mf.index)
# print(df)
# ix = pd.IndexSlice
# print(df.loc[ix[[3],[2,6]],:])
# mf.mloc[[[3], [2,6]]]
print('\n',mf)
print(mf.call('min'))
print(mf.min())
print(mf.call(np.min))
# md2 = mf.mloc[[3,6]]
print('\n',mf.groupby(['x','y']).mean())

# mf,index,columns = get_data()
# columns = pd.DataFrame([[5,7],[3,6]],
#                         index=['c','d'],
#                         columns=['f','g'])
# ms = md.MulSeries([3,2],index=columns,
#                   name=pd.Series([5,2],
#                     index=['x','y'],name='c'))
# mf.loc['c'] = ms
# print(mf)
# print('\n',mf.groupby('y').mean())


# index = pd.DataFrame([['a','b','c'],
#                             [ 'g','b','f'],
#                             [ 'b','g','h']],
#                             columns=['x','y','y'])
# name = pd.Series(['a','b'],index=['e','f'],name='cc')
# ms = md.MulSeries([1,2,3],index=index,name=name)

# # ms.mloc[[..., 'g']] = 5
# print('\n',ms)
# ms.mloc[{'x':'b'}]
# print('\n',ms.mloc[[...,'g']])
# # # ms.mloc[{'y':['c','f']}] = [5,6]

# # print(ms.mloc[['g', ..., ['h','f']]])
# print(ms.mloc[[..., 'b', ['h','f']]])
# print(ms.mloc[['a']])


# ss = pd.Series([1,2,3],index=pd.MultiIndex.from_frame(index))
# ix = pd.IndexSlice
# print('\n',ss)
# res = ss.loc[ix[['g','b'],:,['c','f','h']]]
# print('\n',res)

# res = ss.loc[ix[['g'],['h','f']]]
# print('\n',res)
# res = ss.loc[ix['a']]
# print('\n',res)


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



# index = pd.DataFrame([[1,2],[3,6],[5,6]],
#                      index=['a','b','b'],
#                      columns=['x','y'])
# columns = pd.DataFrame([[5,7],[3,6]],
#                      index=['c','d'],
#                      columns=['f','g'])
# mf = md.MulDataFrame([[1,2],[8,9],[8,7]],index=index,
#                   columns=columns)

# # print(mf.groupby('y',axis=0).sum())
# # print(mf)
# print(mf.groupby('y',axis=0).sum(axis=1))


# print(mf.groupby('f',axis=1).sum(axis=0))

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
