
# version 30001

data_job

_rlnJobTypeLabel             relion.import.other
_rlnJobIsContinue                       0
_rlnJobIsTomo                           0
 

# version 30001

data_joboptions_values

loop_ 
_rlnJobOptionVariable #1 
_rlnJobOptionValue #2 
        Cs        1.4 
        Q0        0.1 
    angpix        0.885 
beamtilt_x          0 
beamtilt_y          0 
  do_other        Yes 
  do_queue         No 
    do_raw         No 
fn_in_other exported_filtered_micrographs.star 
 fn_in_raw Micrographs/*.tif 
    fn_mtf         "" 
is_multiframe        No 
        kV        300 
min_dedicated         48 
 node_type "Micrographs STAR file (.star)" 
optics_group_name opticsGroup1 
optics_group_particles         "" 
other_args         "" 
      qsub     sbatch 
qsubscript /apps/relion/slurm_4.sh 
 queuename   GPUNODES 
 
