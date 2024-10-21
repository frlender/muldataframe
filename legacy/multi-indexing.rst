Multi-indexing
================

mloc
-----
MulDataFrame uses ``.mloc`` to perform multi-indexing. Its input can be a list or a dict. If a list is used, it is similar to the multi-indexing in pandas except that you don't need to create a ``pandas.IndexSlicer`` object. Just input a plain list with ``...`` as placeholders.

>>> mf
(3, 2)    g  7  6
          f  5  3
             c  d
--------  ---------
   x  y      c  d
a  1  2   a  1  2
b  3  6   b  8  9
b  5  6   b  8  7
>>> mf.mloc[[..., 6],[3]]
(2,)     g  6
         f  3
            d
-------  ------
   x  y     d
b  3  6  b  9
b  5  6  b  7

The above example uses the "y" column in the index dataframe to select the 2nd and 3rd rows and the "f" columns in the columns dataframe to select the 2nd column.

If a dict is used, you can change the order of `hierarchical indexing <https://pandas.pydata.org/docs/user_guide/advanced.html>`_:

>>> mf.mloc[{'y':[2,6],'x':[3]}]
(1, 2)    g  7  6
          f  5  3
             c  d
--------  ---------
   x  y      c  d
b  3  6   b  8  9
```

In the above example, the muldataframe is first indexed by the "y" column of the index dataframe and then the "x" column. With a list as input you cannot achieve this. In fact, ``mf.mloc[[[3],[2,6]]]`` will report error. 

When there are duplicate names in the columns of the index or columns dataframe, use the **last** column for dict indexing.

>>> mf2
(3, 2)      g  7  6
            f  5  3
               c  d
----------  ---------
   x  y  y     c  d
a  1  2  8  a  1  2
b  3  6  5  b  8  9
b  5  6  2  b  8  7
>>> mf2.mloc[{'y':[2]}]
(1, 2)      g  7  6
            f  5  3
               c  d
----------  ---------
   x  y  y     c  d
b  5  6  2  b  8  7


You can also mix the use of a dict and a list in ``.mloc``:

>>> mf.mloc[{'y':[2,6],'x':[3]},[..., 7]]
(1,)      g  7
          f  5
             c
--------  ---------
   x  y      c
b  3  6   b  8

``.mloc`` is also implemented for MulSeries:

>>> ms = mf['c']
>>> ms.mloc[[..., 6]]
(2,)     g  7
         f  5
            c
-------  ------
   x  y     c
b  3  6  b  8
b  5  6  b  8




nloc
-------

MulDataFrame and MulSeries also implements ``.nloc`` to enable position-based multi-indexing. If a list is used as input, it behaves exactly the same as ``.mloc``.  If a dict is used, it behaves similarly to ``.mloc`` except that instead of using column names as keys, it uses the numeric positions of the columns as keys.

>>> mf2
(3, 2)      g  7  6
            f  5  3
               c  d
----------  ---------
   x  y  y     c  d
a  1  2  8  a  1  2
b  3  6  5  b  8  9
b  5  6  2  b  8  7
>>> mf2.nloc[{1:6}]
(2, 2)      g  7  6
            f  5  3
               c  d
----------  ---------
   x  y  y     c  d
b  3  6  5  b  8  9
b  5  6  2  b  8  7

Note that with a dict as input to ``.mloc``, you can only select the last "y" column in the index dataframe in the above example. Using ``.nloc`` you can select the first "y" column.

.. _ref-inclusion-indexing:

Inclusive indexing
--------------------------------------------
The multi-indexing in MulDataFrame is implemented differently to that in pandas. MulDataFrame implements inclusive indexing while pandas implements hierarchical indexing. Their difference is illustrated by the two examples below.

.. image:: ../_static/example1.png
  :width: 700
  :alt: Example 1

Example 1 shows that hierarchical indexing requires the indexing value(s) in the next level to be present in all groups selected by the indexing value(s) in the current level. Inclusive indexing, on the other hand, only requires the indexing value(s) in the next column to be present in **any** group selected by the indexing value(s) in the current column. In other words, the rows in the index dataframe selected by the current column should include the indexing values in the next column.

.. image:: ../_static/example2.png
  :width: 700
  :alt: Example 2

As a result, hierarchical indexing fails on the 2nd example but inclusive indexing works. With hierarchical indexing, one cannot use dict objects as indexers because the change of the indexing order breaks the hierarchical structure.

A side effect of hierarchical indexing is that a scalar selection does not always reduce a dataframe to a series or a series to a scalar.



