#!/bin/csh

set MODEL_LAT = 33.5000
set MODEL_LON = -116.5000
set MODEL_ROT = -50.0

set HH = 0.100

set XLEN = 30.0
set YLEN = 40.0
set ZLEN = 20.0
set SUFX = _reg0-h${HH}

set CENTER_ORIGIN = 1
set GEOPROJ = 1

set GRIDFILE = gridfile${SUFX}
set GRIDOUT = gridout${SUFX}
set PARAMFILE = model_params${SUFX}
set COORDFILE = model_coords${SUFX}
set BOUNDFILE = model_bounds${SUFX}

set DOCOORDS = 1

echo $XLEN $YLEN $ZLEN $HH | gawk '{xlen=$1;ylen=$2;zlen=$3;h=$4;}  \
END { \
printf "xlen=%f\n",xlen; printf "%10.4f %10.4f %13.6e\n",0.0,xlen,h; \
printf "ylen=%f\n",ylen; printf "%10.4f %10.4f %13.6e\n",0.0,ylen,h; \
printf "zlen=%f\n",zlen; printf "%10.4f %10.4f %13.6e\n",0.0,zlen,h;}' \
> $GRIDFILE

gen_model_cords geoproj=$GEOPROJ gridfile=$GRIDFILE gridout=$GRIDOUT \
                center_origin=$CENTER_ORIGIN do_coords=$DOCOORDS \
		nzout=1 name=$COORDFILE gzip=0 latfirst=0 \
		modellon=$MODEL_LON modellat=$MODEL_LAT \
		modelrot=$MODEL_ROT > $PARAMFILE

if($DOCOORDS == 1) then

set NX = `grep nx= $GRIDOUT | gawk -F"=" '{print $NF;}'`
set NY = `grep ny= $GRIDOUT | gawk -F"=" '{print $NF;}'`

gawk -v nx=$NX -v ny=$NY '{ \
if($3 == 0 || $3 == nx-1 || $4 == 0 || $4 == ny-1)print $0;}' $COORDFILE > $BOUNDFILE

endif
