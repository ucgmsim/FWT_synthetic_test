#!/bin/csh

set HH = 0.100

set VPMIN = 1.8
set VSMIN = 0.5
set DNMIN = 2.0
set QPMIN = 50.0
set QSMIN = 25.0

set INFILE = mj-vs500.vmod
set OUTFILE = mj-vs500.fd-h$HH

gawk -v h=$HH -v vpm=$VPMIN -v vsm=$VSMIN -v dnm=$DNMIN -v qpm=$QPMIN -v qsm=$QSMIN \
'BEGIN{printf "DEF HST\n";dd=0.0;nlay=0;}{ \
if(substr($0,1,1)!="#"){ \
nlay++; \
dd=dd+$1; \
zz[nlay]=dd; \
vp[nlay]=$2;if(vp[nlay]<vpm)vp[nlay]=vpm; \
vs[nlay]=$3;if(vs[nlay]<vsm)vs[nlay]=vsm; \
dn[nlay]=$4;if(dn[nlay]<dnm)dn[nlay]=dnm; \
qp[nlay]=$5;if(qp[nlay]<qpm)qp[nlay]=qpm; \
qs[nlay]=$6;if(qs[nlay]<qsm)qs[nlay]=qsm; \
}} \
END{ \
h2=0.5*h;for(i=1;i<=nlay;i++){zz[i]=int((zz[i]+h2)/h + 0.5)*h - h2;} \
for(i=1;i<nlay;i++){if(zz[i]>0.0 && zz[i]<zz[i+1] && \
(vp[i]!=vp[i+1] || \
vs[i]!=vs[i+1] || \
dn[i]!=dn[i+1] || \
qp[i]!=qp[i+1] || \
qs[i]!=qs[i+1])) \
printf "%8.4f %8.4f %8.4f %10.2f %10.2f %11.5f\n",vp[i],vs[i],dn[i],qp[i],qs[i],zz[i];} \
printf "%8.4f %8.4f %8.4f %10.2f %10.2f %11.5f\n",vp[nlay],vs[nlay],dn[nlay],qp[nlay],qs[nlay],zz[nlay];}' \
$INFILE > $OUTFILE
