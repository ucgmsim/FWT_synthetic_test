#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:00:22 2019

@author: user
"""
#import obspy.io.sac.sactrace as SACtrace
from pysac.sactrace import SACTrace
import numpy as np

header = {'kstnm': 'ANMO', 'kcmpnm': 'BHZ', 'stla': 40.5, 'stlo': -108.23,
          'evla': -15.123, 'evlo': 123, 'evdp': 50, 'nzyear': 2012,
          'nzjday': 123, 'nzhour': 13, 'nzmin': 43, 'nzsec': 17,
          'nzmsec': 100, 'delta': 1.0/40}

#sac = SACTrace(data=np.random.random(100), **header)
filename0 = '/home/user/workspace/GMPlots/synth_VMs/Medium_VMs/flexwin/test_data/data/events_lh/1995.122.05.32.16.0000.II.ABKT.00.LHZ.D.SAC'
sac0 = SACTrace.read(filename0)
sac = SACTrace(nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
               t1=23.5, data=sac0.data)
filename = 'AAA.000.sac'
sac.write(filename, byteorder='little')
print(sac) 

