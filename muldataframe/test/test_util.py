from ..MulDataFrame import MulDataFrame
from ..util import pivot_table
import pandas as pd
from .lib import eq


def test_pivot_table():
    df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
    
    table = pivot_table(df, values='D', index=['A', 'B'],
            columns=['C'], aggfunc="sum")
    # print(table)
    assert table.index.equals(pd.DataFrame(
        [['bar','one'],['bar','two'],
         ['foo','one'],['foo','two']],
         columns=['A','B']
    ))
    assert table.columns.shape == (2,0)
    assert eq(table.columns.index,['large','small'])