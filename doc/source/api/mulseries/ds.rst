MulSeries.ds
=================

.. currentmodule:: muldataframe

.. attribute:: MulSeries.ds

      A partial copy of the values series. Its difference to the :doc:`MulSeries.ss <ss>` is that its values attribute is not copied but refers to the values attribute of the values series. Its index and columns are deep-copied from the values series. Use this attribute if you want to save some memory or use the same name to refer to the values series/dataframe.
      