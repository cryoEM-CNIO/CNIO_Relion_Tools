
# version 30001

data_scheme_general

_rlnSchemeName                       Schemes/png/
_rlnSchemeCurrentNodeName            WAIT
 

# version 30001

data_scheme_floats

loop_ 
_rlnSchemeFloatVariableName #1 
_rlnSchemeFloatVariableValue #2 
_rlnSchemeFloatVariableResetValue #3 
maxtime_hr    48.000000    48.000000 
wait_sec     1.000000     1.000000  

# version 30001

data_scheme_bools

loop_ 
_rlnSchemeBooleanVariableName #1 
_rlnSchemeBooleanVariableValue #2 
_rlnSchemeBooleanVariableResetValue #3 
do_do_png           0            0 
has_ctffind           0            0 
 
# version 30001

data_scheme_strings

loop_ 
_rlnSchemeStringVariableName #1 
_rlnSchemeStringVariableValue #2 
_rlnSchemeStringVariableResetValue #3 
ctffind_mics Schemes/prep/ctffind/micrographs_ctf.star Schemes/prep/ctffind/micrographs_ctf.star 


# version 30001

data_scheme_operators

loop_ 
_rlnSchemeOperatorName #1 
_rlnSchemeOperatorType #2 
_rlnSchemeOperatorOutput #3 
_rlnSchemeOperatorInput1 #4 
_rlnSchemeOperatorInput2 #5 
EXIT_maxtime exit_maxtime  undefined maxtime_hr  undefined 
HAS_ctffind bool=file_exists has_ctffind ctffind_mics  undefined 
SET_do_do_png   bool=not   do_do_png     do_log  undefined 
WAIT       wait  undefined   wait_sec  undefined 
 

# version 30001

data_scheme_jobs

loop_ 
_rlnSchemeJobNameOriginal #1 
_rlnSchemeJobName #2 
_rlnSchemeJobMode #3 
_rlnSchemeJobHasStarted #4 
  do_png   do_png   continue            0 
 

# version 30001

data_scheme_edges

loop_ 
_rlnSchemeEdgeInputNodeName #1 
_rlnSchemeEdgeOutputNodeName #2 
_rlnSchemeEdgeIsFork #3 
_rlnSchemeEdgeOutputNodeNameIfTrue #4 
_rlnSchemeEdgeBooleanVariable #5 
WAIT EXIT_maxtime            0  undefined  undefined 
EXIT_maxtime HAS_ctffind            0  undefined  undefined 
HAS_ctffind       WAIT            1 SET_do_do_png has_ctffind
SET_do_do_png   do_png            0  undefined  undefined 
do_png       WAIT            0  undefined  undefined 
 
