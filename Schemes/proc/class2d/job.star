
# version 30001

data_job

_rlnJobTypeLabel             relion.class2d
_rlnJobIsContinue                       0
_rlnJobIsTomo                           0
 

# version 30001

data_joboptions_values

loop_ 
_rlnJobOptionVariable #1 
_rlnJobOptionValue #2 
allow_coarser         No 
ctf_intact_first_peak         No 
do_bimodal_psi        Yes 
 do_center        Yes 
do_combine_thru_disc         No 
do_ctf_correction        Yes 
     do_em         No 
   do_grad        Yes 
  do_helix         No 
do_parallel_discio        Yes 
do_preread_images         No 
  do_queue        Yes 
do_restrict_xoff        Yes 
do_zero_mask        Yes 
dont_skip_align        Yes 
   fn_cont         "" 
    fn_img Schemes/proc/extract/particles.star 
   gpu_ids         "" 
helical_rise       4.75 
helical_tube_outer_diameter        200 
highres_limit         16 
min_dedicated         48 
nr_classes        100 
nr_iter_em         25 
nr_iter_grad        200 
    nr_mpi          1 
   nr_pool         30 
nr_threads         48 
offset_range          5 
offset_step          1 
other_args         "" 
particle_diameter      198.0 
psi_sampling          6 
      qsub     sbatch 
qsubscript /apps/relion/slurm_4.sh 
 queuename GPUNODESFAST 
 range_psi          6 
scratch_dir   /scratch 
 tau_fudge          2 
   use_gpu        Yes 
 
