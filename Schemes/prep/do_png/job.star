
# version 30001

data_job

_rlnJobTypeLabel             relion.external
_rlnJobIsContinue                       0
_rlnJobIsTomo                           0
 

# version 30001

data_joboptions_values

loop_ 
_rlnJobOptionVariable #1 
_rlnJobOptionValue #2 
  do_queue         No 
    fn_exe png_out.py 
  in_3dref         "" 
 in_coords         "" 
   in_mask         "" 
    in_mic Schemes/prep/ctffind/micrographs_ctf.star
    in_mov         "" 
   in_part         "" 
min_dedicated         48 
nr_threads          1 
other_args         "" 
param10_label         "" 
param10_value         "" 
param1_label         "" 
param1_value         "" 
param2_label         "" 
param2_value         "" 
param3_label         "" 
param3_value         "" 
param4_label         "" 
param4_value         "" 
param5_label         "" 
param5_value         "" 
param6_label         "" 
param6_value         "" 
param7_label         "" 
param7_value         "" 
param8_label         "" 
param8_value         "" 
param9_label         "" 
param9_value         "" 
      qsub     sbatch 
qsubscript /apps/relion/slurm_4dev.sh 
 queuename   GPUNODES 
 
