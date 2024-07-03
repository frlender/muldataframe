import pandas as pd
import muldataframe.cmm as cmm
import muldataframe as md

def aggregate_index(i:int,index:pd.DataFrame,index_agg:cmm.IndexAgg):
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
    elif index_agg in ['array','tuple']:
        index_vals = []
        for col in index.columns:
            vals = index[col].unique()
            if len(vals) == 1:
                index_vals.append(vals[0])
            else:
                vals = tuple(vals) if index_agg == 'tuple' else vals
                index_vals.append(vals)
        return pd.DataFrame([index_vals],columns=index.columns,index=final_index)



def concat(ss1,ss2):
    ss_new = pd.concat([ss1.ss,ss2.ss])
    index_new = pd.concat([ss1.index,ss2.index],join='inner')
    return md.MulSeries(ss_new,index=index_new,
                    name=ss1.name.copy())