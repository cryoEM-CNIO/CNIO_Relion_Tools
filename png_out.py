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
from concurrent.futures import ThreadPoolExecutor, as_completed

print('running ...')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", "--o", help="output folder")
parser.add_argument("-i", "--in_mics", help="input CTF micrograph starfile")
parser.add_argument("--j", help="number of threads to be used", default=4, type=int)
args, unknown = parser.parse_known_args()

inargs = args.in_mics
outargs = args.output
threads = args.j

# Import
print('parsing STAR file...')

try:
    starf_df_optics = starfile.read(inargs)['optics']
    starf_df_mics = starfile.read(inargs)['micrographs']
    apix = float(starf_df_optics['rlnMicrographPixelSize'])
    micrographs = starf_df_mics
    motion = micrographs['rlnMicrographName']
    ctf = micrographs['rlnCtfImage']
except Exception as e:
    print("No input detected or error occurred:", e)
    with open(outargs + "RELION_JOB_EXIT_SUCCESS", "w+") as f:
        f.close()
    sys.exit()

print('writing pngs...')

# Process MotCor files
def process_motion(i):
    cmd = (f'if test -f \"{i[:-3]}png\"; then continue; else `which relion_image_handler` '
           f'--i {i} --o {i[:-3]}png --angpix {apix} --rescale_angpix {apix * 5} '
           f'--sigma_contrast 6 --lowpass 10; fi')
    os.system(cmd)

# Process CtfFind files
def process_ctf(i):
    cmd = (f'if test -f \"{i[:-7]}png\"; then continue; else `which relion_image_handler` '
           f'--i {i} --o {i[:-7]}png; fi')
    os.system(cmd)

# Paralellize
max_workers = threads
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for i in motion:
        futures.append(executor.submit(process_motion, i))
    for i in ctf:
        futures.append(executor.submit(process_ctf, i))
    
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            print(f'Generated an exception: {exc}')

with open(outargs + "RELION_JOB_EXIT_SUCCESS", "w+") as f:
    f.close()
    
print('done!')
### Finish
