#! /bin/csh

# Script to reformat a series of adjoint ascii files into a single binary file for emod3d-mpi

set BPATH = /home/rgraves/Bin

# specify output binary file and directory
set ADJOINT_FILE = my_adjoint_file.e3d
set OUTDIR = ../../../AdjSims/V3.0.7-a2a_xyz/Adj-SourceBin

\mkdir -p $OUTDIR

# required simulation parameters need to be the same as forward runs
set NT = 2000
set DT = 0.08
set HH = 2.0
#set DT = 0.04
#set HH = 1.0
set MODEL_ROT = 0.0

# specification of input files and associated meta data follows

set INDIR = ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii
set STATCORDS = ../../../StatInfo/fd_rt01-h0.400.statcords

# note: station lon,lat info not used currently, but would be best to provide
#       correct locations for possible double-checking
#set STATS = ( A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 C3 C4 D1 D2 D3 D4)
#set SLONS = ( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#set SLATS = ( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

set STATS = ( HALS GOVS DHSS MQZ CSTC KPOC SMHS HUNS LINC AMBC NBLC HORC SPRS REHS KOWC WCSS SHFC SBRC AKSS DORC DSLC GDLC CSHS MNZS DALS LRSC PPHS STKS CACS WSFC CCCC RKAC ROLC OXZ LPCC HHSS OHSS CBGS OPWS SWNC MENS GODS TPLC RHSC PRPC PARS CMHS MTHS STAS SACS MPSS MORS MTPS NNBS LCQC TOKS SMTC ASHS DFHS ADCS SLRC HPSC BWHS MRSS D14C EYRS CRLZ COLD TLED KVSD)
#set STATS = ( HALS GOVS DHSS MQZ CSTC KPOC SMHS HUNS LINC AMBC NBLC HORC SPRS REHS KOWC WCSS SHFC SBRC AKSS DORC DSLC GDLC CSHS DALS LRSC PPHS STKS CACS CCCC RKAC ROLC OXZ LPCC HHSS OHSS CBGS OPWS SWNC MENS GODS SHLC TPLC RHSC PRPC PARS CMHS MTHS STAS SUMS SACS MPSS MORS MTPS NNBS LCQC TOKS SMTC LPOC ASHS DFHS ADCS HVSC SLRC HPSC BWHS KILS MRSS RCBS D14C EYRS CHHC D09C CRLZ D06C D07C COLD TLED KVSD )

set SLONS = ( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 )
set SLATS = ( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 )


# specification of components follows
#
# valid choices for adjoint components are: vx, vy and vz, which are then inserted at
# vx, vy, and vz nodes, respectively in emod3d-mpi
#
# COMPS are the components for the ascii input files and should match the .<comp> values given 
# in the filenames

# example for only 2 components
set ADJOINT_COMPS = ( vx vz )
set COMPS = ( x z )

# example for only 1 components
set ADJOINT_COMPS = ( vy )
set COMPS = ( y )

# example for all 3 components
set ADJOINT_COMPS = ( vx vy vz )
set COMPS = ( x y z )

# Done with input specifications, processing stuff follows

set ADJ_COMPS_IN = $ADJOINT_COMPS[1]
foreach comp ( $ADJOINT_COMPS[2-] )

set ADJ_COMPS_IN = ${ADJ_COMPS_IN},$comp

end

# temporary file for listing inputs, will be created by script
set FILELIST = ../../../AdjSims/V3.0.7-a2a_xyz/a2a_filelist.txt

\rm $FILELIST
set s = 0
foreach stat ( $STATS )
@ s ++

gawk -v ss=$stat -v nt=$NT -v slon=$SLONS[$s] -v slat=$SLATS[$s] '{ \
if(NF>=4 && ss==$4){ix=$1;iy=$2;iz=$3;}} \
END{printf "%5d %5d %5d %6d %.5f %.5f %s",ix,iy,iz,nt,slon,slat,ss;}' $STATCORDS >> $FILELIST

set a = 0
foreach comp ( $ADJOINT_COMPS )
@ a ++

echo -n " ${INDIR}/${stat}.${COMPS[$a]}" >> $FILELIST

end

echo "" >> $FILELIST

end

${BPATH}/ascii2adj filelist=$FILELIST outfile=$OUTDIR/$ADJOINT_FILE \
                   adjoint_comps=${ADJ_COMPS_IN} dt=$DT h=$HH modelrot=$MODEL_ROT >& ../../../AdjSims/V3.0.7-a2a_xyz/a2a.out
