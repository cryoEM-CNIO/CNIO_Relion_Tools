#!/usr/bin/env python3

"""
Estimate Ice Thickness from CtfFind4
Rafael Fernandez-Leiro & Nayim Gonzalez-Rodriguez 2022
"""

"""
Activate conda environment before opening relion
Execute from relion gui as external job providing the input micrograph_ctf.star and callyng ice.py executable
"""

### Setup
import os
import pandas as pd
# from pandas.core.common import SettingWithCopyWarning
import glob as glob
import numpy as np
import starfile
import pathlib
import sys
import argparse
import warnings
from os.path import exists

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

print('running ...')

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output" , "--o", help = "output folder")
parser.add_argument("-i", "--input", "--in_mics", help = "input micrographs")
args, unknown = parser.parse_known_args()

inargs=args.input
outargs=args.output

print(str('grabbed arguments: ')+ inargs + str(' ') + outargs)

print('calling script...')

### Import

filename = inargs

try:
    dfoptics = starfile.read(filename)['optics']
    df = starfile.read(filename)['micrographs']
except:
    print("No input detected")
    f=open(outargs+"RELION_JOB_EXIT_SUCCESS","w+")
    f.close()
    exit()

if exists(outargs+str('/micrographs_ctf_ice.star')):
    olddf = starfile.read(outargs+str('/micrographs_ctf_ice.star'))['micrographs']
    outdf = df.join(olddf['rlnMicrographIceThickness'])
else:
    outdf = df
    outdf['rlnMicrographIceThickness']=np.nan

### Estimate ice thickness

n = -1
for row in outdf['rlnMicrographIceThickness']:
    n+= 1
    if np.isnan(row):
        txtfile = outdf['rlnCtfImage'][n][:-8]+'_avrot.txt'
        print(txtfile)
        dftxt = (pd.read_csv(txtfile, skiprows=[0,1,2,3,4,7,8,9,10,11,12], header=None, delim_whitespace=True)).transpose()
        ice_ring = dftxt[0].between(0.25,0.28, inclusive = 'both')
        outdf['rlnMicrographIceThickness'][n]=(round((sum(np.abs(dftxt[ice_ring][1]))),6))
    else:
        continue

### Output

dict_output = {'optics' : dfoptics , 'micrographs' : outdf}
starfile.write(dict_output, outargs+'/micrographs_ctf_ice.star', overwrite=True)

### Finish

f=open(outargs+"RELION_OUTPUT_NODES.star","w+")
f.write("data_output_nodes\nloop_\n_rlnPipeLineNodeName #1\n_rlnPipeLineNodeTypeLabel #2\n"+outargs+"/micrographs_ctf_ice.star MicrographsData.star.relion")
f.close()

print('done!')

f=open(outargs+"RELION_JOB_EXIT_SUCCESS","w+")
f.close()
