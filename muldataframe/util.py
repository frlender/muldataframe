import pandas as pd
import muldataframe.cmm as cmm
import muldataframe as md
# import muldataframe.cmm as cmm


def aggregate_index(i:int,index:pd.DataFrame,index_agg:cmm.IndexAgg) -> pd.DataFrame:
    # agg_mode: 'same_only', 'array'
    # print(key)
    final_index = pd.Index([i])
    if index_agg == 'same_only':
        # print(index[index.columns[0]].unique())
        index_same = index.loc[:,[len(index[col].unique())==1 for col in index.columns]]
        index_one = index_same.iloc[[0]]
        index_one.index = final_index
        # print('====',index,'\n',index_same,'\n',index_one)
        return index_one
    elif index_agg in ['list','tuple']:
        index_vals = []
        for col in index.columns:
            vals = index[col].unique()
            if len(vals) == 1:
                index_vals.append(vals[0])
            else:
                vals = tuple(vals) if index_agg == 'tuple' else vals
                index_vals.append(vals)
        return pd.DataFrame([index_vals],columns=index.columns,index=final_index)

        

def concat(arr:list[md.MulSeries|md.MulDataFrame],axis=0):
    ds_new = pd.concat([x.ds for x in arr],join='inner',axis=axis)
    # print('ddddddd',mds1.ds,mds2.ds,ds_new)
    if isinstance(ds_new,pd.Series):
        mindex_new = pd.concat([x.index for x in arr],join='inner')
        return md.MulSeries(ds_new.values,index=mindex_new,
                    name=arr[0].name,index_copy=False)
    else:
        if axis == 0:
            mindex_new = pd.concat([x.index for x in arr],join='inner')
            return md.MulDataFrame(ds_new.values,index=mindex_new,
                    columns=arr[0].columns,index_copy=False)
        else:
            mcols_new = pd.concat([x.columns 
                if hasattr(x,'columns') else x.name 
                for x in arr],join='inner')
            return md.MulDataFrame(ds_new.values,index=arr[0].index,
                    columns=mcols_new,columns_copy=False)
    

def pivot_table(*args,**kwargs):
    df = pd.pivot_table(*args,**kwargs)
    # print(df)
    if isinstance(df.index,pd.MultiIndex):
        new_idx = df.index.to_frame(index=False)
    else:
        new_idx = pd.DataFrame(index=df.index)
    if isinstance(df.columns,pd.MultiIndex):
        new_cols = df.columns.to_frame(index=False)
    else:
        new_cols = pd.DataFrame(index=df.columns)
    
    return md.MulDataFrame(df.values,
                           index=new_idx,
                           columns=new_cols,
                           both_copy=False)