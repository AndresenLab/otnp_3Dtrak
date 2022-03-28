# -*- coding: utf-8 -*-
"""
Example:
radarray=lutcalc2('C:/Temp/LUT.csv', 'C:/Temp/vid_57810.avi')

Created on Wed Aug  1 16:42:28 2018

@author: kandrese
"""

def lutcalc2(zfile, vidfile):
    
    import csv
    import cv2
    import numpy as np
    from centandrad2 import centandrad2
    from progressbar import ProgressBar, Percentage, Bar
     
    skipzlines=44 #First line of csv file to analyze
    endanalysis=818 #Last line of csv file to analyze
    #numberoflut=100
    numberoflut=100 #Number of images the LUT should have
    
    
    #Read in csv and get times and z positions
    with open(zfile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        times = []
        piezoz = []
        i=0
        for row in readCSV:
            i+=1
            times.append(float(row[0]))
            piezoz.append(float(row[3]))
            
    times=np.asarray(times) #Turn lists into arrays
    piezoz=np.asarray(piezoz)
    
    times=times-times[0] #Subtract zero time
    
    zspace=int((endanalysis-skipzlines)/numberoflut) #Calculate spacing between z steps
    
    cap=cv2.VideoCapture(vidfile)

    ret=True
    it=0
    zstep=skipzlines
    radarray=[]
    radavg=[]
    zlist=[]
    frame=0
    progcount=0
    pbar=ProgressBar(widdgets=[Percentage(), Bar()], maxval=numberoflut).start()
    try:
        while(ret and it<(times.shape[0]-1) and len(zlist)<=numberoflut):
            framepos = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            ret=cap.grab()
            frame+=1
            if framepos>times[it]:
                it+=1
                numframes=0
            if it != zstep:
                continue
            numframes+=1
            ret, image=cap.retrieve()
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
            image = image[160:280, 380:500]
            [newcenx, newceny, rad]=centandrad2(image) #This prints centers

            if numframes==1:
                radavg=rad
            else:
                radavg=rad+radavg
                #numframes+=1
            if (framepos+1/cap.get(cv2.CAP_PROP_FPS))>times[it]:
                radavg=radavg/numframes
                
                if numframes<10:
                    print("Numframes=", numframes)
                    print("It=%d, zstep=%d, framepos=%.2f, time=%.2f" % (it,zstep,framepos,times[it]))
                if zstep==skipzlines:
                    radarray=radavg
                else:
                    radarray=np.c_[radarray, radavg]
                zstep+=zspace
                radavg=[]
                zlist.append(piezoz[it])
                if progcount<numberoflut:
                    progcount+=1
                pbar.update(progcount)
    except KeyboardInterrupt:
        pass
    except UnboundLocalError as ulerror:
        print("Something went wrong at frame", frame)
        print(ulerror)    
    #           print((it-skipzlines)/(times.shape[0]-skipzlines)*100, '% Complete. Time elapsed:', tm.time()-start)
#            start=tm.time()
    cap.release()
    pbar.finish()
    zarray=np.asarray(zlist)
    return radarray, zarray    
