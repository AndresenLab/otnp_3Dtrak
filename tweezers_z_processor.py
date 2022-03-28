# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:33:57 2018

@author: jm2080
"""

import zdetect as zd
import os
import numpy as np
import time

t_start = time.time()

root = '/Volumes/Backup Plus/3DTrak/DIR/'
vidfiles = []

for dirpath, dirnames, files in os.walk(root):
    for name in files:
        if name.lower().endswith('.avi'):
             vidfiles.append(os.path.join(dirpath, name))

save = '/Volumes/Backup Plus/3DTrak/SAVEHERE/'

i = 0
n = len(vidfiles)
totalz = np.zeros(210)
backupz = np.zeros(210)

while i < n:
#    vidfiles[i].replace('\\','/')
    s1 = vidfiles[i].split('\\')[-1]
    s1b = s1.split('.')[0]
    s2 = s1b.split('_')[0] #date
    s3 = s1b.split('_')[3] #time
    zl = zd.zanalyzevid3('/Volumes/Backup Plus/3DTrak/DIR2/14450_CCD_tweezers_14450.avi')
    if np.size(zl) > 350:
        totalz = zl[0:210]
        np.save(save+s2+"_zPos_"+s3+".npy", totalz)
        totalz = zl[-211:-1]
        s3b = str(int(s3) + 7)
        if int(s3b[4:6]) > 59:
           p3 = str("%02d" % (int(s3b[4:6]) - 60))
           p2 = str("%02d" % (int(s3b[2:4]) + 1))
           p1 = s3b[0:2]
           if int(p2) > 59:
               p2 = str("%02d" % (int(p2) - 60))
               p1 = str("%02d" % (int(p1) + 1))
           s3b = p1 + p2 + p3
        np.save(save+s2+"_zPos_"+s3b+".npy", totalz)
    elif np.size(zl) < 210:
        j = 210 - np.size(zl)
        zl = np.pad(zl, (0, j), 'constant')
        totalz = zl[0:210]
        np.save(save+s2+"_zPos_"+s3+".npy", totalz)
    else:
        totalz = zl[0:210]
        np.save(save+s2+"_zPos_"+s3+".npy", totalz)    
    i=n
t_end = time.time() - t_start
print(t_end)
