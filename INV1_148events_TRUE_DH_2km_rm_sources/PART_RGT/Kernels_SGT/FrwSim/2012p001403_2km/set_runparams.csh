#!/bin/csh

set EQNAME = ${argv[1]}

set VERSION = 3.0.8

set NPROC_X = 8
set NPROC_Y = 8
set NPROC_Z = 5

set NPROC_TOTAL = `echo $NPROC_X $NPROC_Y $NPROC_Z | gawk '{printf "%.0f\n",$1*$2*$3;}'`
echo $NPROC_TOTAL

set NX = 65
set NY = 65
set NZ = 40
set HH = 2.000
set NT = 2000
set DT = 0.08

set MAXMEM = 12000

set BFILT = 4
set FLO = 0.1

set ORDER = 2
set ORDER = 4

set RUN_DIR_ROOT = /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/INV1_148events_TRUE_DH_2km_rm_sources/PART_RGT/Kernels_SGT
set SIM_DIR_ROOT = ${RUN_DIR_ROOT}/FrwSim/${EQNAME}

set MODEL_LAT = -43.5999
set MODEL_LON = 172.2999
set MODEL_ROT = 0.0
set XAZIM = `echo $MODEL_ROT | gawk '{print $1+90.0;}'`

set STYPE = cos
#set RISE_TIME = 0.1
set RISE_TIME = 1

set RUN_NAME = reg0
set STATCORDS = /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/INV1_148events_TRUE_DH_2km_rm_sources/StatInfo/fd_rt01-h2.000.statcords

#set MODEL_STYLE = 0

#set VMODDIR = /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/INV1_148events_TRUE_DH_2km_rm_sources/PART_RGT/Kernels_SGT/Model/Mod-1D
#set FD_MODFILE = mj-vs500.fd-h2.000
set MODEL_STYLE = 1
#set VMODDIR = /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/INV1_148events_TRUE_DH_2km_rm_sources/Model/Models
set VMODDIR = /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/INV1_148events_TRUE_DH_2km_rm_sources/PART_RGT/Kernels_SGT/Iters/iter1/Model
set PMOD = vp3dfile_opt.p
set SMOD = vs3dfile_opt.s
set DMOD = rho3dfile_opt.d

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

set DEFAULT_PARFILE = e3d_default.par
set PARFILE = e3d.par

set MAXMEM = 12000

set ENABLE_RESTART = 0
set READ_RESTART = 0
set MAIN_RESTARTDIR = ${SIM_DIR_ROOT}/Restart
set RESTART_ITINC = 20000

set ENABLE_DUMP = 0
set MAIN_DUMP_DIR = ${SIM_DIR_ROOT}/OutBin
set DUMP_ITINC = 4000
#set DUMP_ITINC = 0

set WORK_OUTPUT_DIR = ${SIM_DIR_ROOT}/OutBin
set DELETE_TMPFILES = 1
set DELETE_TMPFILES = 0

if( $#argv > 2) then

set ELON = ${argv[2]}
set ELAT = ${argv[3]}
set EDEP = ${argv[4]}

set SXY = `echo $ELON $ELAT | /scale_wlg_persistent/filesets/home/rgraves/Bin/ll2xy mlon=$MODEL_LON mlat=$MODEL_LAT xazim=$XAZIM`
set XSRC = `echo $SXY[1] $HH $NX | gawk '{printf "%d\n",int(0.5*$3 + $1/$2 + 0.5);}'`
set YSRC = `echo $SXY[2] $HH $NY | gawk '{printf "%d\n",int(0.5*$3 + $1/$2 + 0.5);}'`
set ZSRC = `echo $EDEP $HH | gawk '{printf "%d\n",int($1/$2 + 0.5) + 1;}'`

set ESTK = ${argv[5]}
set EDIP = ${argv[6]}
set ERAK = ${argv[7]}
set EMOM = ${argv[8]}

/scale_wlg_persistent/filesets/home/rgraves/Bin/rotate_momten strike=$ESTK dip=$EDIP rake=$ERAK moment=$EMOM >& momten_conversions.txt

set Mnn = `gawk '{if($1 == "Mnn=")print $2;}' momten_conversions.txt`
set Mee = `gawk '{if($1 == "Mee=")print $2;}' momten_conversions.txt`
set Mdd = `gawk '{if($1 == "Mdd=")print $2;}' momten_conversions.txt`
set Mne = `gawk '{if($1 == "Mne=")print $2;}' momten_conversions.txt`
set Mnd = `gawk '{if($1 == "Mnd=")print $2;}' momten_conversions.txt`
set Med = `gawk '{if($1 == "Med=")print $2;}' momten_conversions.txt`

\cp $DEFAULT_PARFILE $PARFILE

echo "#"        >> $PARFILE
echo "# Run specific parameters start here"     >> $PARFILE
echo "#"        >> $PARFILE

echo "version=${VERSION}-mpi"		>> $PARFILE
echo "name=${RUN_NAME}"			>> $PARFILE
echo "report=1000"                       >> $PARFILE
echo "reportId=0"                       >> $PARFILE

echo "maxmem=${MAXMEM}"                 >> $PARFILE
echo "order=$ORDER"          >> $PARFILE

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
echo "bforce=0"				>> $PARFILE
echo "dblcpl=0"				>> $PARFILE
echo "ffault=0"                         >> $PARFILE

echo "pointmt=1"                        >> $PARFILE
echo "xsrc=$XSRC"                       >> $PARFILE
echo "ysrc=$YSRC"                       >> $PARFILE
echo "zsrc=$ZSRC"                       >> $PARFILE
echo "stype=$STYPE"                     >> $PARFILE
echo "rtime=$RISE_TIME"                 >> $PARFILE
echo "Mnn=$Mnn"                         >> $PARFILE
echo "Mee=$Mee"                         >> $PARFILE
echo "Mdd=$Mdd"                         >> $PARFILE
echo "Mne=$Mne"                         >> $PARFILE
echo "Mnd=$Mnd"                         >> $PARFILE
echo "Med=$Med"                         >> $PARFILE

echo "model_style=${MODEL_STYLE}"                    >> $PARFILE
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

echo "fmax=5.0"                        >> $PARFILE
#echo "fmax=0.2"                        >> $PARFILE
echo "fmin=0.05"                        >> $PARFILE

echo "modellon=$MODEL_LON"              >> $PARFILE
echo "modellat=$MODEL_LAT"              >> $PARFILE
echo "modelrot=$MODEL_ROT"              >> $PARFILE

echo "seisdir=${WORK_OUTPUT_DIR}"           >> $PARFILE

echo "enable_output_dump=${ENABLE_DUMP}"	>> $PARFILE
echo "dump_itinc=${DUMP_ITINC}"			>> $PARFILE
echo "main_dump_dir=${MAIN_DUMP_DIR}"    	>> $PARFILE
echo "delete_tmpfiles=${DELETE_TMPFILES}"       >> $PARFILE

echo "nseis=1"				>> $PARFILE
echo "seiscords=${STATCORDS}"           >> $PARFILE

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
if(-e $STATCORDS ) then

echo statcords:
\ls -lt $STATCORDS

else

echo "**** ERROR ****"
echo "statscords= $STATCORDS does not exist"

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
