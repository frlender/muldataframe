import pathlib
import sys
# print(pathlib.Path(__file__).parents[2])
fld = pathlib.Path(__file__).parents[1].resolve().as_posix()
sys.path.insert(0,fld)

import muldataframe as md


ss_attrs = ['index','name',
         'values','ss','ds','shape',
         'mindex','mname','iloc','loc','mloc']

# def get_lex(num):
#    # print(num)
#    fold = int(num/9)
#    residue = num%9
#    # print(fold,residue)
#    if fold > 9:
#       return get_lex(fold)+str(residue)
#    else:
#       return str(fold)+str(residue)

# mp={}
# fnames = ['00_mulseries.rst']
# for i, ax in enumerate(ss_attrs):
#    lex = get_lex(i+1)
#    fname = f'{lex}_{ax}.rst'
#    fnames.append(fname)
#    mp[ax] = fname.split('.')[0]

ss_dyn_attrs = {
   'values':
      '''
      The values of the values series.
      ''',
   'ss':
      '''
      A deep copy of the values series.
      ''',
   'ds':
      '''
      A partial copy of the values series. Its difference to the :doc:`MulSeries.ss <ss>` is that its values attribute is not copied but refers to the values attribute of the values series. Its index and columns are deep-copied from the values series. Use this attribute if you want to save some memory or use the same name to refer to the values series/dataframe.
      ''',
   'shape':
      '''
      Same as the shape of the values series.
      ''',
   'mindex':
      '''
      Alias for :doc:`MulSeries.index <index>`.
      ''',
   'mname':
      '''
      Alias for :doc:`MulSeries.name <name>`.
      '''
}

fnames = ['mulseries.rst']
for ax in ss_attrs:
   fname = ax+'.rst'
   fnames.append(fname)
   # print(i,lex,fname)
   
   with open(f'source/api/mulseries/{fname}','w') as ff:
      underlines = '='*(len(ax)+15)
      ff.write(f'MulSeries.{ax}\n{underlines}\n\n')
      if ax not in ss_dyn_attrs:
         ff.write(f'.. autoattribute:: muldataframe.MulSeries.{ax}\n')
      else:
         ff.write('.. currentmodule:: muldataframe\n\n')
         ff.write(f'.. attribute:: MulSeries.{ax}\n')
         ff.write(ss_dyn_attrs[ax])


with open('source/__template/index.rst','r') as rf:
   wstr = rf.read()
   lines = '\n'.join([f'   {x}' for x in fnames])
   wstr = wstr+lines
   with open(f'source/api/mulseries/indices.rst','w') as wf:
      wf.write(wstr)

# print(ss_dyn_attrs)