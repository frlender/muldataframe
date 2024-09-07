Numpy functions
==================
Similar to :ref:`pandas methods <pandas_methods_attr>`, all numpy functions in the main namespace have been added as methods to MulDataFrame and MulSeries using the :doc:`call method <call_method>`. For exmaple, ``np.log1p`` is added but ``numpy.linalg.matmul`` is not. There is some overlap between numpy functions and pandas methods. In this case, pandas methods take precedence over numpy functions.

>>> 