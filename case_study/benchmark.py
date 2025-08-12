import os
import time
import numpy as np
import pandas as pd
import muldataframe as md
import argparse

MulDataFrame = md.MulDataFrame

data_path = os.path.join('.','data')
if not os.path.exists(data_path):
    os.mkdir(data_path)

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--both", action="store_true", help="run both 1e6 and 1e7 examples. Otherwise, only 1e6 is run.")
args = parser.parse_args()

if args.both:
    nums = [1e6,1e7]
else:
    nums = [1e6]

# print(nums)

iter_count = 100
xf = pd.DataFrame(columns=['muldataframe','pandas','pandas_build_idx']).astype(float)
for rlen in nums:
    rlen = int(rlen)
    print(f'Row length:{rlen}')
    xf.loc[rlen] = 0
    np.random.seed(123)
    df = pd.DataFrame(np.random.random((rlen,3)))
    for i in range(iter_count):
        if (i+1) % 10 == 0:
            print(i+1)
        rds = np.random.randint(0,rlen,3)
        rd1,rd3,rd2 = sorted(rds,reverse=True)
        # print(rd1,rd2,rd3)
        index = pd.DataFrame({'x':['a']*rd1+['b']*(rlen-rd1),
                            'y':['a']*rd2+['b']*(rlen-rd2),
                            'z':['a']*rd3+['b']*(rlen-rd3)})
        mf = MulDataFrame(df,index=index)
        t0 = time.time()
        res = mf.mloc[['a','b','a']]
        t1 = time.time()
        xf.loc[rlen,'muldataframe'] += t1-t0

        df2 = df.copy()
        t0 = time.time()
        idx = pd.IndexSlice
        df2.index = pd.MultiIndex.from_frame(index)
        # df3 = df2.reorder_levels(order=[1,2,0])
        res = df2.loc[idx['a','b','a']]
        t1 = time.time()
        xf.loc[rlen,'pandas_build_idx'] += t1-t0

        t0 = time.time()
        res = df2.loc[idx['a','b','a']]
        t1 = time.time()
        xf.loc[rlen,'pandas'] += t1-t0

        df3 = df2.reset_index()
        df3.columns = [str(x) for x in df3.columns]
        dirPath = os.path.join('.','data',str(rlen))
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        ftPath = os.path.join(dirPath,f'{i+1}.ft')
        df3.to_feather(ftPath)
    
print(xf/iter_count)
# print(xf['muldataframe']/xf['pandas_build_idx'])



