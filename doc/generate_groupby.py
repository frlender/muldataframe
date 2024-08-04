from lib import *


attrs = ['parent','by',
         'index_agg','indexType','groupBy']

methods = ['__iter__','call']

fld = 'source/api/groupby'

generate_attr_files('MulGroupBy',
    attrs,{},'source/api/groupby',
    'cmm.MulGroupBy')

generate_method_files('MulGroupBy',
    methods,'source/api/groupby',
    'cmm.MulGroupBy')

data = [['Constructor',['mulgroupby']],
        ['Attributes',attrs],
        ['Methods',methods]]
generate_index('MulGroupBy',data,f'{fld}/indices.rst')