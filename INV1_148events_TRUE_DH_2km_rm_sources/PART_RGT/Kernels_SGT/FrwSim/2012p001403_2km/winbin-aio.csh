#! /bin/csh

set SEISDIR = OutBin
set RUN = reg0

set FILELIST = fdb.filelist

set TSTRT = -3.0
set TSTRT = -3.05
set SCALE = 1.0

set VELDIR = Vel
\mkdir $VELDIR

set FD_STATLIST = ../../StatInfo/fd_reg0-h0.100.ll
set STATS = `gawk '{print $3;}' $FD_STATLIST `
set STATS = AZ_TRO

echo $STATS

set COMPS = ( 040 130 ver )
set IX = ( 0 1 2 )
set FLIP = ( 1 1 -1 )

ls ${SEISDIR}/${RUN}_seis*.e3d > $FILELIST

set s = 0
foreach stat ( $STATS )
@ s ++

set a = 0
foreach comp ( $COMPS )
@ a ++

fdbin2wcc all_in_one=1 filelist=$FILELIST \
	  ix=$IX[$a] scale=$SCALE flip=$FLIP[$a] tst=$TSTRT \
	  stat=$stat comp=$comp outfile=${stat}.${comp} \
	  nseis_comps=9 swap_bytes=0 tst2zero=0 outbin=0

wcc_header infile=${stat}.${comp} outfile=${stat}.${comp}

end

wcc_rotate filein1=${stat}.$COMPS[1] inbin1=0 \
	   filein2=${stat}.$COMPS[2] inbin2=0 \
	   fileout1=${stat}.000 outbin1=0 \
           fileout2=${stat}.090 outbin2=0 \
	   rot=0.0

\mv ${stat}.000 ${stat}.090 ${stat}.ver $VELDIR
\rm ${stat}.$COMPS[1] ${stat}.$COMPS[2]

end
