#!/bin/csh

set STAT = ${argv[1]}

set VERSION = 3.0.8

set NPROC_X = 4
set NPROC_Y = 4
set NPROC_Z = 3

set NPROC_TOTAL = `echo $NPROC_X $NPROC_Y $NPROC_Z | gawk '{printf "%.0f\n",$1*$2*$3;}'`
echo $NPROC_TOTAL

set NX = 300
set NY = 400
set NZ = 200
set HH = 0.100
set NT = 6000
set DT = 0.005

set MAXMEM = 12000

set BFILT = 4
set FLO = 1.0
set SGT_TINC = 1

set ORDER = 2
set ORDER = 4

set RUN_DIR_ROOT = ${HOME}/SGT-Example
set SIM_DIR_ROOT = ${RUN_DIR_ROOT}/SgtCalc/${STAT}

set MODEL_LAT = 33.5000
set MODEL_LON = -116.5000
set MODEL_ROT = -50.0
set XAZIM = `echo $MODEL_ROT | gawk '{print $1+90.0;}'`

set STYPE = cos
set RISE_TIME = 0.1

set SGTCORDS = ${RUN_DIR_ROOT}/SgtInfo/SgtCords/${STAT}-h${HH}.cordfile

set MODEL_STYLE = 0

set VMODDIR = ${RUN_DIR_ROOT}/Model/Mod-1D
set FD_MODFILE = mj-vs500.fd-h0.100

set PERTBFILE = pertb4.0-fz0.70_sv02-h0.025.pertb
set NX_PERTB = $NX
set NY_PERTB = $NY
set NZ_PERTB = $NZ

set VPMIN_PERTB = 1.25
set VPMAX_PERTB = 8.25
set VSMIN_PERTB = 0.25
set VSMAX_PERTB = 5.0
set DEN_SCALE_FACTOR = 0.0

set DUMP_ITINC = 4000

set ENABLE_RESTART = 0
set READ_RESTART = 0
set MAIN_RESTARTDIR = ${SIM_DIR_ROOT}/Restart
set RESTART_ITINC = 20000

set ENABLE_DUMP = 0
set MAIN_DUMP_DIR = ${SIM_DIR_ROOT}/OutBin
set DUMP_ITINC = 4000

set WORK_OUTPUT_DIR = ${SIM_DIR_ROOT}/OutBin
set DELETE_TMPFILES = 1
set DELETE_TMPFILES = 0

if( $#argv > 2) then

set SLON = ${argv[2]}
set SLAT = ${argv[3]}

set RUN_NAME = ${STAT}-${argv[4]}
set XMOM = ${argv[5]}
set YMOM = ${argv[6]}
set ZMOM = ${argv[7]}

set SXY = `echo $SLON $SLAT | ll2xy mlon=$MODEL_LON mlat=$MODEL_LAT xazim=$XAZIM`
set XSRC = `echo $SXY[1] $HH $NX | gawk '{printf "%d\n",int(0.5*$3 + $1/$2 + 0.5);}'`
set YSRC = `echo $SXY[2] $HH $NY | gawk '{printf "%d\n",int(0.5*$3 + $1/$2 + 0.5);}'`
set ZSRC = 1

set DEFAULT_PARFILE = e3d_default.par
set PARFILE = e3d-${argv[4]}.par

\cp $DEFAULT_PARFILE $PARFILE

echo "#"        >> $PARFILE
echo "# Run specific parameters start here"     >> $PARFILE
echo "#"        >> $PARFILE

echo "version=${VERSION}-mpi"           >> $PARFILE
echo "name=${RUN_NAME}"			>> $PARFILE
echo "report=1000"                       >> $PARFILE
echo "reportId=0"                       >> $PARFILE

echo "maxmem=${MAXMEM}"                 >> $PARFILE
echo "order=$ORDER"                           >> $PARFILE

echo "nx=$NX"                           >> $PARFILE
echo "ny=$NY"                           >> $PARFILE
echo "nz=$NZ"                           >> $PARFILE
echo "h=$HH"                            >> $PARFILE
echo "nt=$NT"                           >> $PARFILE
echo "dt=$DT"                           >> $PARFILE

echo "nproc_x=$NPROC_X"                           >> $PARFILE
echo "nproc_y=$NPROC_Y"                           >> $PARFILE
echo "nproc_z=$NPROC_Z"                           >> $PARFILE

echo "bfilt=$BFILT"				>> $PARFILE
echo "flo=$FLO"				>> $PARFILE
echo "fhi=0.0"				>> $PARFILE
echo "dblcpl=0"				>> $PARFILE
echo "ffault=0"                         >> $PARFILE
echo "pointmt=0"                        >> $PARFILE

echo "bforce=1"                         >> $PARFILE
echo "xsrc=$XSRC"                       >> $PARFILE
echo "ysrc=$YSRC"                       >> $PARFILE
echo "zsrc=$ZSRC"                       >> $PARFILE
echo "stype=$STYPE"                     >> $PARFILE
echo "rtime=$RISE_TIME"                 >> $PARFILE
echo "xmom=$XMOM"                         >> $PARFILE
echo "ymom=$YMOM"                         >> $PARFILE
echo "zmom=$ZMOM"                         >> $PARFILE
echo "stfdir=Stf"                     >> $PARFILE

echo "model_style=${MODEL_STYLE}"       >> $PARFILE
echo "vmoddir=${VMODDIR}"               >> $PARFILE

if($MODEL_STYLE == 0 || $MODEL_STYLE == 2) then

echo "model=$FD_MODFILE"                >> $PARFILE

else if($MODEL_STYLE == 1 || $MODEL_STYLE == 3) then

echo "pmodfile=$PMOD"                   >> $PARFILE
echo "smodfile=$SMOD"                   >> $PARFILE
echo "dmodfile=$DMOD"                   >> $PARFILE
echo "qpfrac=100"                        >> $PARFILE
echo "qsfrac=50"                       >> $PARFILE
echo "qpqs_factor=2.0"                  >> $PARFILE

endif

echo "pertbfile=$PERTBFILE"                   >> $PARFILE
echo "nx_pertb=$NX_PERTB"                   >> $PARFILE
echo "ny_pertb=$NY_PERTB"                   >> $PARFILE
echo "nz_pertb=$NZ_PERTB"                   >> $PARFILE

echo "vpmin_pertb=$VPMIN_PERTB"                   >> $PARFILE
echo "vpmax_pertb=$VPMAX_PERTB"                   >> $PARFILE
echo "vsmin_pertb=$VSMIN_PERTB"                   >> $PARFILE
echo "vsmax_pertb=$VSMAX_PERTB"                   >> $PARFILE
echo "den_scale_factor=$DEN_SCALE_FACTOR"          >> $PARFILE

echo "fmax=25.0"				>> $PARFILE
echo "fmin=0.05"			>> $PARFILE

echo "modellon=$MODEL_LON"              >> $PARFILE
echo "modellat=$MODEL_LAT"              >> $PARFILE
echo "modelrot=$MODEL_ROT"              >> $PARFILE

echo "seisdir=${WORK_OUTPUT_DIR}"           >> $PARFILE

echo "enable_output_dump=${ENABLE_DUMP}"        >> $PARFILE
echo "dump_itinc=${DUMP_ITINC}"                 >> $PARFILE
echo "main_dump_dir=${MAIN_DUMP_DIR}"           >> $PARFILE
echo "delete_tmpfiles=${DELETE_TMPFILES}"       >> $PARFILE

echo "nseis=0"				>> $PARFILE
echo "sgtout=1"                         >> $PARFILE
echo "sgt_tinc=$SGT_TINC"                               >> $PARFILE
echo "sgtcords=${SGTCORDS}"           >> $PARFILE

echo "ts_xy=0"				>> $PARFILE

echo "enable_restart=${ENABLE_RESTART}" >> $PARFILE
echo "restartdir=${MAIN_RESTARTDIR}"    >> $PARFILE
echo "restart_itinc=${RESTART_ITINC}"   >> $PARFILE
echo "read_restart=${READ_RESTART}"     >> $PARFILE
echo "restartname=${RUN_NAME}"          >> $PARFILE

echo "vpvs_max_global=3.0"          >> $PARFILE

else

if($argv[2] == "--check") then

echo
echo checking files ...

echo
echo run_dir_root= $RUN_DIR_ROOT
echo sim_dir_root= $SIM_DIR_ROOT

echo
echo work_output_dir= $WORK_OUTPUT_DIR
echo main_dump_dir= $MAIN_DUMP_DIR

echo
if(-e $SGTCORDS ) then

echo sgtcords:
\ls -lt $SGTCORDS

else

echo "**** ERROR ****"
echo "sgtcords= $SGTCORDS does not exist"

endif

if($MODEL_STYLE == 0 || $MODEL_STYLE == 2) then

echo
if(-e $VMODDIR/$FD_MODFILE ) then

echo 1d model:
\ls -lt $VMODDIR/$FD_MODFILE

else

echo "**** ERROR ****"
echo "model= $VMODDIR/$FD_MODFILE does not exist"

endif

else if($MODEL_STYLE == 1 || $MODEL_STYLE == 3) then

echo
if(-e $VMODDIR/$PMOD ) then

echo vp file:
\ls -lt $VMODDIR/$PMOD

else

echo "**** ERROR ****"
echo "model= $VMODDIR/$PMOD does not exist"

endif

echo
if(-e $VMODDIR/$SMOD ) then

echo vs file:
\ls -lt $VMODDIR/$SMOD

else

echo "**** ERROR ****"
echo "model= $VMODDIR/$SMOD does not exist"

endif

echo
if(-e $VMODDIR/$DMOD ) then

echo density file:
\ls -lt $VMODDIR/$DMOD

else

echo "**** ERROR ****"
echo "model= $VMODDIR/$DMOD does not exist"

endif

endif

if($MODEL_STYLE == 2 || $MODEL_STYLE == 3) then

echo
if(-e $VMODDIR/$PERTBFILE ) then

echo pertbfile:
\ls -lt $VMODDIR/$PERTBFILE

else

echo "**** ERROR ****"
echo "model= $VMODDIR/$PERTBFILE does not exist"

endif
endif

endif

endif
