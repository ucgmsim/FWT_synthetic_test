#!/bin/csh

set NX = 267
set NY = 269
set NZ = 75
set HH = 0.400

set MODEL_LAT = -43.59999583292219
set MODEL_LON = 172.29999979889402
set MODEL_ROT = 0.0

#set DEFAULT_PARFILE = e3d_default.par
#set PARFILE = e3d.par
set PARFILE = ll2xy.par
\rm $PARFILE

set ELON = 172.92310
set ELAT = -43.28590
set EDEP = 20.0000

# use /home/rgraves/Bin/ll2xy to determine xsrc, ysrc & zsrc
set XAZIM = `echo $MODEL_ROT | gawk '{print $1+90.0;}'`
set EXY = `echo $ELON $ELAT | ./ll2xy mlon=$MODEL_LON mlat=$MODEL_LAT xazim=$XAZIM`
#set EXY = `echo $ELON $ELAT | ./home/rgraves/Bin/ll2xy mlon=$MODEL_LON mlat=$MODEL_LAT xazim=$XAZIM`
set XSRC = `echo $EXY[1] $HH $NX | gawk '{printf "%d\n",int(0.5*$3 + $1/$2);}'`
set YSRC = `echo $EXY[2] $HH $NY | gawk '{printf "%d\n",int(0.5*$3 + $1/$2);}'`
set ZSRC = `echo $EDEP $HH | gawk '{printf "%d\n",int($1/$2 + 0.5) + 1;}'`
#set XSRC = 50
#set YSRC = 50
#set ZSRC = 40

#\cp $DEFAULT_PARFILE $PARFILE

echo "xsrc=$XSRC"                       >> $PARFILE
echo "ysrc=$YSRC"                       >> $PARFILE
echo "zsrc=$ZSRC"                       >> $PARFILE


echo "modellon=$MODEL_LON"              >> $PARFILE
echo "modellat=$MODEL_LAT"              >> $PARFILE
echo "modelrot=$MODEL_ROT"              >> $PARFILE


