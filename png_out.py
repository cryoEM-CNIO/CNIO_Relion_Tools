#!/usr/bin/env python

"""
Write PNG files for motioncorr and ctf outputs
Rafael Fernandez-Leiro & Nayim Gonzalez-Rodriguez 2022
"""

"""
Activate conda environment before opening relion
Execute from relion gui as external job providing the input micrograph_ctf.star and callyng png_out.py executable
"""

import sys
import argparse
import os
import starfile

"""VARIABLES >>>"""
print('running ...')


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output" , "--o", help = "output folder")
parser.add_argument("-i", "--in_mics", help = "input CTF micrograph starfile")
args, unknown = parser.parse_known_args()

inargs=args.in_mics
outargs=args.output

# Import

print('parsing STAR file...')

try:
    starf_df_optics = starfile.read(inargs)['optics']
    starf_df_mics = starfile.read(inargs)['micrographs']
    apix = float(starf_df_optics['rlnMicrographPixelSize'])
    micrographs = starf_df_mics
    motion = micrographs['rlnMicrographName']
    ctf = micrographs['rlnCtfImage']
except:
    print("No input detected")
    f=open(outargs+"RELION_JOB_EXIT_SUCCESS","w+")
    f.close()    
    exit()

print('writing pngs...')

# Launching relion_image_handler to produce PNG files

for i in motion:
    os.system(str('if test -f \"'+i[:-3]+'png\"; then continue; else `which relion_image_handler` --i '+i+ ' --o '+i[:-3]+'png --angpix '+str(apix)+' --rescale_angpix '+str(apix*5)+' --sigma_contrast 6 --lowpass 10; fi'))

for i in ctf:
    os.system(str('if test -f \"'+i[:-7]+'png\"; then continue; else `which relion_image_handler` --i '+i+' --o '+i[:-7]+'png; fi'))

print('done!')

### Finish

f=open(outargs+"RELION_JOB_EXIT_SUCCESS","w+")
f.close()
