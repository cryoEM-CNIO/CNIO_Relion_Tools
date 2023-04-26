
# version 30001

data_job

_rlnJobTypeLabel             relion.motioncorr
_rlnJobIsContinue                       0
_rlnJobIsTomo                           0
 

# version 30001

data_joboptions_values

loop_ 
_rlnJobOptionVariable #1 
_rlnJobOptionValue #2 
   bfactor        150 
bin_factor          1 
do_dose_weighting        Yes 
do_float16        Yes 
do_own_motioncor        Yes 
  do_queue         No 
do_save_noDW        Yes 
do_save_ps        Yes 
dose_per_frame      1.277 
eer_grouping         20 
first_frame_sum          1 
 fn_defect         "" 
fn_gain_ref data/gain.mrc 
fn_motioncor2_exe /public/EM/MOTIONCOR2/MotionCor2 
 gain_flip "No flipping (0)" 
  gain_rot "No rotation (0)" 
   gpu_ids         "" 
group_for_ps          4 
group_frames          1 
input_star_mics Schemes/prep/importmovies/movies.star 
last_frame_sum         -1 
min_dedicated         48 
    nr_mpi          1 
nr_threads         24 
other_args "--skip_logfile --do_at_most $$do_at_most" 
other_motioncor2_args         "" 
   patch_x          5 
   patch_y          5 
pre_exposure          0 
      qsub     sbatch 
qsubscript /apps/relion/slurm_4.sh 
 queuename GPUNODESFAST 
 
