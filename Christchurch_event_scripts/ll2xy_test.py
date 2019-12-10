#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:16:19 2019

@author: user
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:56:20 2019

@author: user
"""
#import os
#import commands
import numpy as np
from subprocess import Popen, PIPE
#import matplotlib.pyplot as plt

def ll2xy_conv(MODEL_LON,MODEL_LAT,MODEL_ROT,ELON,ELAT,EDEP,NX,NY,HH):
       
    XAZIM = MODEL_ROT+90.0
    
    cmd = 'echo '+ str(ELON) + " " + str(ELAT) +'|./ll2xy mlon='+str(MODEL_LON)+' mlat='+str(MODEL_LAT)+' xazim='+str(XAZIM)
    stdout = Popen(cmd, shell=True, stdout=PIPE).stdout
    output = stdout.read()    
    EXY = output.split()
   # input('-->')

    XSRC = int(0.5*NX + float(EXY[0])/HH)

    YSRC = int(0.5*NY + float(EXY[1])/HH)

    ZSRC = int(EDEP/HH + 0.5) + 1 
    
    print([XSRC, YSRC, ZSRC])   
    
    return XSRC, YSRC, ZSRC
#########################################
#nSource, sNames = read_list_srf('list_srf.txt')
#S = np.zeros((nSource,3))

MODEL_LON = 172.92310
MODEL_LAT = -43.28590
MODEL_ROT = 0.0

NX = 267
NY = 269
NZ = 75
HH = 0.400

ELON = 172.92310
ELAT = -43.28590
EDEP = 20.0000

XSRC, YSRC, ZSRC = ll2xy_conv(MODEL_LON,MODEL_LAT,MODEL_ROT,ELON,ELAT,EDEP,NX,NY,HH)
