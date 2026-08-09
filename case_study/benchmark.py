import os
import time
import numpy as np
import pandas as pd
import muldataframe as md
import argparse
from collections import defaultdict
import tracemalloc

MulDataFrame = md.MulDataFrame

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--both", action="store_true", help="run both 1e6 and 1e7 examples. Otherwise, only 1e6 is run.")
args = parser.parse_args()

if args.both:
    nums = [1e6,1e7]
else:
    nums = [1e7]

# print(nums)

iter_count = 10
xf = defaultdict(list)
for rlen in nums:
    rlen = int(rlen)
    print(f'Row length:{rlen}')
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
        tracemalloc.start()
        mf = MulDataFrame(df,index=index)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Peak memory: {peak / 1024**2:.2f} MB")

        t0 = time.time()
        res = mf.mloc[['a','b','a']]
        t1 = time.time()
        xf[(rlen,'muldataframe', 'multi-indexing')].append(t1-t0)

        df2 = df.copy()
        t0 = time.time()
        idx = pd.IndexSlice
        df2.index = pd.MultiIndex.from_frame(index)
        # df3 = df2.reorder_levels(order=[1,2,0])
        res = df2.loc[idx['a','b','a']]
        t1 = time.time()
        xf[(rlen,'pandas_build_idx', 'multi-indexing')].append(t1-t0)
        # xf.loc[rlen,'pandas_build_idx'] += t1-t0

        tracemalloc.start()
        idx = pd.IndexSlice
        df2.index = pd.MultiIndex.from_frame(index)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Peak memory: {peak / 1024**2:.2f} MB")

        
        t0 = time.time()
        mf.index.loc[100,'x'] = 'c'
        t1 = time.time()
        xf[(rlen,'muldataframe', 'metadata editting')].append(t1-t0)

        t0 = time.time()
        index_frame = df2.index.to_frame().reset_index(drop=True)
        # print(index_frame)
        index_frame.loc[100,'x'] = 'c'
        df2.index= pd.MultiIndex.from_frame(index_frame)
        t1 = time.time()
        xf[(rlen,'pandas_build_idx', 'metadata editting')].append(t1-t0)



        # t0 = time.time()
        # res = df2.loc[idx['a','b','a']]
        # t1 = time.time()
        # xf.loc[rlen,'pandas'] += t1-t0

# print(xf)
    
for rlen in nums:
    rlen = int(rlen)
    for col in ['muldataframe','pandas_build_idx']:
        for item in ['multi-indexing', 'metadata editting']:
            avg = np.mean(xf[(rlen, col, item)])
            std = np.std(xf[(rlen, col, item)], ddof=1)
            print((rlen, col, item), avg, std)

# print(xf['muldataframe']/xf['pandas_build_idx'])



