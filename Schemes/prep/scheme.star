
# version 30001

data_scheme_general

_rlnSchemeName                       Schemes/prep/
_rlnSchemeCurrentNodeName            WAIT
 

# version 30001

data_scheme_floats

loop_ 
_rlnSchemeFloatVariableName #1 
_rlnSchemeFloatVariableValue #2 
_rlnSchemeFloatVariableResetValue #3 
do_at_most     1.000000     1.000000 
maxtime_hr    48.000000    48.000000 
  wait_sec     1.000000     1.000000 
 

# version 30001

data_scheme_operators

loop_ 
_rlnSchemeOperatorName #1 
_rlnSchemeOperatorType #2 
_rlnSchemeOperatorOutput #3 
_rlnSchemeOperatorInput1 #4 
_rlnSchemeOperatorInput2 #5 
EXIT_maxtime exit_maxtime  undefined maxtime_hr  undefined 
      WAIT       wait  undefined   wait_sec  undefined 
 

# version 30001

data_scheme_jobs

loop_ 
_rlnSchemeJobNameOriginal #1 
_rlnSchemeJobName #2 
_rlnSchemeJobMode #3 
_rlnSchemeJobHasStarted #4 
   ctffind    ctffind   continue            0 
    do_png     do_png   continue            0 
   get_ice    get_ice   continue            0 
importmovies importmovies   continue            0 
motioncorr motioncorr   continue            0 
import_mic_sele import_mic_sele   continue            1 
 

# version 30001

data_scheme_edges

loop_ 
_rlnSchemeEdgeInputNodeName #1 
_rlnSchemeEdgeOutputNodeName #2 
_rlnSchemeEdgeIsFork #3 
_rlnSchemeEdgeOutputNodeNameIfTrue #4 
_rlnSchemeEdgeBooleanVariable #5 
      WAIT EXIT_maxtime            0  undefined  undefined 
EXIT_maxtime importmovies            0  undefined  undefined 
importmovies motioncorr            0  undefined  undefined 
motioncorr    ctffind            0  undefined  undefined 
   ctffind     do_png            0  undefined  undefined 
    do_png    get_ice            0  undefined  undefined 
   get_ice import_mic_sele            0  undefined  undefined 
import_mic_sele       WAIT            0  undefined  undefined 
