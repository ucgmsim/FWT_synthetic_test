#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:14:05 2019

@author: user
"""
import numpy as np
ishot=2
fi=open('iShot.dat','w')
(np.int64(ishot)).tofile(fi)
fi.close()
print('isource='+str(ishot))
