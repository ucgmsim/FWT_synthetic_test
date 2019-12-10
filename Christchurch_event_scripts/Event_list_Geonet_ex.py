# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:34:43 2019

@author: user

GeoNet FDSN webservice with Obspy demo - Event Service
"""

from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read_inventory

client = FDSN_Client("GEONET")



starttime = "2016-11-13 11:00:00.000"
endtime = "2016-11-14 11:00:00.000"
cat = client.get_events(starttime=starttime, endtime=endtime,latitude=-42.693,longitude=173.022,maxradius=0.5,minmagnitude=5)
print(cat)
_=cat.plot(projection="local")


#cat = client.get_events(eventid="2016p858000")
#print(cat)

dir(cat[1])
cat[1]['resource_id']