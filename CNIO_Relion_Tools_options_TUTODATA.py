{
### RELION_IT OPTIONS FILE
### PREP PNG SCHEDULES
### Many parameters are speficif for relion's beta-gal TUTORIAL DATA
### 
##  GENERAL SETTINGS RELION_IT
'do_auto_boxsize' : 'True', 
'do_prep' : 'True', 
'do_proc' : 'False', 
'do_png' : 'True', 
'prep__wait_sec' : '1', # Waiting time in between prep cycles. Keep this low if doing on-the-fly. Make it larger if cycles are faster than data copying speed.
'prep__do_at_most' : '1', # This will tell MotionCorr to only act on 'n' number of movies at a time. Helpful to set this to 5 or 10 during on-the-fly analysis.
'proc__wait_sec' : '45', # Waiting time in between cycles of the proc schedule
'proc__do_3d' : 'True', 
'proc__do_log' : 'False',
'proc__min_nr_parts_3d' : '5000.0',  
'proc__iniref' : 'None', 
'proc__topaz_model' : '', 
###
#   PREP
###
#   import
'prep__importmovies__angpix' : '0.885', 
'prep__importmovies__Cs' : '1.4', 
'prep__importmovies__fn_in_raw' : 'data/*.tiff', 
'prep__importmovies__is_multiframe' : 'True', 
'prep__importmovies__kV' : '200.0', 
#   motioncorr
'prep__motioncorr__other_args' : '--skip_logfile --do_at_most \$\$do_at_most',
'prep__motioncorr__bin_factor' : '1', 
'prep__motioncorr__eer_grouping' : '20',
'prep__motioncorr__dose_per_frame' : '1.277',
'prep__motioncorr__fn_gain_ref' : 'data/gain.mrc',
'prep__motioncorr__do_queue' : 'No',
'prep__motioncorr__nr_mpi' : '1',
'prep__motioncorr__nr_threads' : '24',
'prep__motioncorr__queuename' : 'GPUNODESFAST',
'prep__motioncorr__min_dedicated' : '48',
'prep__motioncorr__qsub' : 'sbatch',
'prep__motioncorr__qsubscript' : '/apps/relion/slurm_4.sh',
'prep__motioncorr__gpu_ids' : '',
'prep__motioncorr__gain_flip' : 'No Flipping (0)',
#   ctffind local
'prep__ctffind__fn_ctffind_exe' : 'ctffind',
'prep__ctffind__fn_gctf_exe' : 'actual_gctf',
'prep__ctffind__other_args' : '--skip_logfile',
'prep__ctffind__do_phaseshift' : 'False', 
}