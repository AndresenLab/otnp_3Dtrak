# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:16:51 2018

Collection of functions for determining the z position of a particle.
Written by Kurt Andresen and Jeffrey Mc Hugh based on work by Marijn T.J. van Loenhout et al,
published in Biophysical Journal 102, 2362 (2012).

"""

def zanalyzevid3(vidfilename, skipframes=0, zippy=0):
    """
    Gets z position from videos taken on tweezers setup.
    """
    import cv2
    import numpy as np
    from progressbar import ProgressBar, Percentage, Bar

    #control for incorrect video filename, etc
    vidopen = False
    while vidopen != True:
        cap = cv2.VideoCapture(vidfilename)
        #open avi file
        vidopen = cap.isOpened()
        if vidopen:
            continue
        print("The file did not seem to open correctly.")
        vidfilename = input("Please enter a new filename or press q to quit:")
        if vidfilename == ('q' or 'Q'):
            return None
        continue
    ret = True
    #empty list for z positions
    zval = []
    #start=time.time()
    i = 0
    zipframe = zippy
    framemax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #total no of frames
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=framemax).start()
    for i in range(framemax):
        #ret, image = cap.read()
        ret = cap.grab()
        if i < skipframes:
            continue
        elif zipframe != zippy:
            zipframe += 1
            continue
        else:
            zipframe = 0
        if ret is False:
            break #escape loop if no frame to read in, also used to be ret == False
        ret, image = cap.retrieve()
        #converts colour space of frame to greyscale
        im = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        im = im[160:280, 380:500]
        #input the cropped image into centerandrad
        [_, _, imrad] = centandrad2(im)
        #takes third output of centerandrad only
        #takes the imrad output and iteration and inputs to calcheight 'i+1' is frame no
        [zpos, _, _] = calcheight3(imrad)
        #takes first output of calcheight2
        zval = np.append(zval, zpos)
        pbar.update(i+1)
    cap.release()
    pbar.finish() #these two used to come after the return statement
    return zval

def calcheight3(rad):
    """
    Compares radial integral of frame to lookup table (LUT), fits chi square
    and finds minimum of fit.
    """
    import numpy as np

    #windows about which chi squared is fit with 2nd order polynomial
    fitsize = 20
    #load LUT
    radarray = np.load('LUTfine.npy',  allow_pickle=True)
    #
    lut = radarray[0]
    #z positions
    zlut = radarray[1]
    chivec = np.empty([lut.shape[1]], float)
    for i in range(0, lut.shape[1]):
        radn = rad
        chisq = (lut[:, i]-radn)**2/abs(lut[:, i])
        chivec[i] = np.sum(chisq)
    chiminin = np.argmin(chivec)
    if chiminin<fitsize:
        chivecfitter=np.concatenate((np.flip(chivec[chiminin:chiminin+fitsize],0),chivec[chiminin:chiminin+fitsize]))
    elif chiminin>(lut.shape[1]-1)-fitsize:
        chivecfitter=np.concatenate((chivec[chiminin-fitsize:chiminin], np.flip(chivec[chiminin-fitsize:chiminin], 0)))
#    if chiminin < fitsize or chiminin > (lut.shape[1]-1)-fitsize:
#        print("Near edge of LUT. Please reconsider. Returning \
#              approximate z position.\n Z is approximately", zlut[chiminin])
#        zval = zlut[chiminin]
#        return None
#        #return zval, chimin, chivec
    else:
        chivecfitter = chivec[chiminin-fitsize:chiminin+fitsize]
    x = np.arange(0, 2*fitsize, 1)
    try:
        chifit = np.polyfit(x, chivecfitter, 2)
        chider = np.polyder(chifit)
        polychider = np.poly1d(chider)
    except TypeError:
        print("chiminin = ", chiminin)
    chimin = polychider.r[0]+chiminin-fitsize
    if chiminin < fitsize or chiminin > (lut.shape[1]-1)-fitsize:
        #print("Near edge of LUT. Please reconsider. Returning approximate
        #z position.\n Z is approximately", zlut[chiminin])
        zval = zlut[chiminin]
        return zval, chimin, chivec
    zx = np.arange(chiminin-fitsize, chiminin+fitsize, 1)
    zfit = np.polyfit(zx, zlut[chiminin-fitsize:chiminin+fitsize], 1)
    zval = zfit[0]*chimin+zfit[1]
#        print("\n approx z=", zlut[chiminin], "z exact=", zval)
    return zval, chimin, chivec#, lut[chimin]

def centandrad2(image):
    """
    This function is used to get the center of an image and obtain the radial
    integral profile. It first uses "getcenter2" to obtain the center. It
    returns a new (cropped) image and the center (xcen, ycen) in that image.
    It also returns the center of the image in the coordinates of the original
    uncropped image (confusing named newcenx, newceny). The found center and
    cropped image is then fed into the radial integrator program which takes the
    radial integral quandrant-by-quadrant and then returns these integrals. In the
    final line we just average them and return the integral profile as well as the
    center in the original coordinates.
    """
    [im, xcen, ycen, newcenx, newceny] = getcenter2(image)
    [qtr, qtl, qbr, qbl] = radintv2(im, xcen, ycen)
    rad = (qtr+qtl+qbr+qbl)/4
    return newcenx, newceny, rad

def getcenter2(image):
    """
    Run the cross correlation algorithm to find center. This first
    performs COM centering, crops the file, and then cross correlates to
    find the center.
    """
    import numpy as np

    [newimage, xcen, ycen, newcen] = crosscorrv2(image)
    [qtr, qtl, qbr, qbl] = radintv2(newimage, xcen, ycen)

    #-------------------find xcen final----------------#
    qr = qtr+qbr
    ql = qtl+qbl
    iprofx = np.concatenate([np.flip(ql, 0), qr])
    xcorr = crossandfindv2(iprofx)
    #print("Xcorr=", xcorr)
    xcenfinal = xcen-xcorr
    #-------------------find xcen final----------------#
    qt = qtr+qtl
    qb = qbr+qbl
    iprofy = np.concatenate([np.flip(qb, 0), qt])
    ycorr = crossandfindv2(iprofy)
    #print("Ycorr=", ycorr)
    ycenfinal = ycen-ycorr
    newcenx = newcen[0]-xcorr
    newceny = newcen[1]-ycorr

    return newimage, xcenfinal, ycenfinal, newcenx, newceny

def pol2cart(rho, phi):
    """
    Converts polar coordinates to cartesian, all in 2D.
    """
    import math
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    return(x, y)

def radintv2(image, centerx, centery):
    """
    Seperates image into 4 quadrants, interpolates all pixels,
    calculates radial integral per quadrant.

    Example usage:
        [qtr, qtl, qbr, qbl] = radintv2(newimage, xcen, ycen)
    """
    import numpy as np
    import math
    from scipy import interpolate

    dtheta = 2/image.shape[0]
    dr = 1/3 #1/3 of pixel size
    x = np.arange(0, image.shape[0], 1)
    y = np.arange(0, image.shape[1], 1)
    #f=interpolate.interp2d(x, y, image, kind='linear') #Too slow
    f = interpolate.RectBivariateSpline(x, y, image) #Faster! :)
    r = np.arange(0, image.shape[0], dr)

    thetaq1 = np.arange(0, math.pi/2, dtheta)
    thetaq2 = np.arange(math.pi/2, math.pi, dtheta)
    thetaq3 = np.arange(math.pi, 3*math.pi/2, dtheta)
    thetaq4 = np.arange(3*math.pi/2, 2*math.pi, dtheta)

    x1 = np.empty([r.shape[0], thetaq1.shape[0]], float)
    x2 = np.empty([r.shape[0], thetaq2.shape[0]], float)
    x3 = np.empty([r.shape[0], thetaq3.shape[0]], float)
    x4 = np.empty([r.shape[0], thetaq4.shape[0]], float)
    y1 = np.empty([r.shape[0], thetaq1.shape[0]], float)
    y2 = np.empty([r.shape[0], thetaq2.shape[0]], float)
    y3 = np.empty([r.shape[0], thetaq3.shape[0]], float)
    y4 = np.empty([r.shape[0], thetaq4.shape[0]], float)

    for i in range(0, r.shape[0]):
        for j in range(0, thetaq1.shape[0]):
            [x1[i, j], y1[i, j]] = pol2cart(r[i], thetaq1[j])
            [x2[i, j], y2[i, j]] = pol2cart(r[i], thetaq2[j])
            [x3[i, j], y3[i, j]] = pol2cart(r[i], thetaq3[j])
            [x4[i, j], y4[i, j]] = pol2cart(r[i], thetaq4[j])
    q1 = f(x1+centerx, y1+centery, grid=False)
    q1sum = (np.sum(q1, axis=1))/q1.shape[1]
    q2 = f(x2+centerx, y2+centery, grid=False)
    q2sum = (np.sum(q2, axis=1))/q2.shape[1]
    q3 = f(x3+centerx, y3+centery, grid=False)
    q3sum = (np.sum(q3, axis=1))/q3.shape[1]
    q4 = f(x4+centerx, y4+centery, grid=False)
    q4sum = (np.sum(q4, axis=1))/q4.shape[1]
    return q1sum, q2sum, q3sum, q4sum

def crosscorrv2(image):
    """
    Takes an image cropped to image = image[160:280, 380:500] and then finds
    the center via COM and then cross-correlation.
    """
    import numpy as np
#Note: current center is 120, 120 due to cropping
#Find the COM center on our cropped image
    [xcom, ycom] = comv2(image)
    image = image[xcom-40:xcom+40, ycom-40:ycom+40]
    #---First find average over pixels for a small region
    #image=abs(image-np.median(image)) should I subtract the background?
    px = np.empty([image.shape[1]], float)
    py = np.empty([image.shape[0]], float)
#------------------x-------------------------
    #   jmin=int(round(0.4*image.shape[0]))
    #   jmax=int(round(0.6*image.shape[0]))
    jmin = int(round(0.2*image.shape[0]))
    jmax = int(round(0.8*image.shape[0]))
    for i in range(0, image.shape[1]):
        px[i] = 0
        for j in range(jmin, jmax):
            px[i] += image[j, i]
        #px[i]=1/(0.2*image.shape[0])*px[i]
        px[i] = 1/(0.6*image.shape[0])*px[i]
#--------------------------y-----------------
    for i in range(0, image.shape[0]):
        py[i] = 0
        for j in range(jmin, jmax):
            py[i] += image[i, j]
        #py[i]=1/(0.2*image.shape[1])*py[i]
        py[i] = 1/(0.6*image.shape[1])*py[i]
#Perform cross correlation
    crosserx = np.correlate(px, np.flip(px, 0), 'same')
    crossery = np.correlate(py, np.flip(py, 0), 'same')
#Fit correlation to 5th order polynomial
    #------------x-----------------------
    fitrangemin = int(round(crosserx.shape[0]/2))-20
    fitrangemax = int(round(crosserx.shape[0]/2))+20

#-----------Fit correlation and find roots in x-----------------------
    x = np.arange(fitrangemin, fitrangemax, 1)
    pxfit = np.polyfit(x, crosserx[fitrangemin:fitrangemax], 2)
    pxder = np.polyder(pxfit)
    polypxder = np.poly1d(pxder)
    maxflag = 0
    for i in range(0, polypxder.r.shape[0]):
        if polypxder.r[i].imag == 0 and polypxder.r[i] > 0 and polypxder.r[i] < fitrangemax-1 and polypxder.r[i] > fitrangemin:
            if crosserx[int(round(polypxder.r[i].real))] > maxflag:
                centerx = polypxder.r[i]
                maxflag = crosserx[int(round(polypxder.r[i].real))]
    #------------y-----------------------
    y = np.arange(fitrangemin, fitrangemax, 1)
    pyfit = np.polyfit(y, crossery[fitrangemin:fitrangemax], 2)
    pyder = np.polyder(pyfit)
    polypyder = np.poly1d(pyder)
    maxflag = 0
    for i in range(0, polypyder.r.shape[0]):
        if polypyder.r[i].imag == 0 and polypyder.r[i] > 0 and polypyder.r[i] < fitrangemax-1 and polypyder.r[i] > fitrangemin:
            if crossery[int(round(polypyder.r[i].real))] > maxflag:
#                print('i=',i, 'Index', int(round(polypxder.r[i].real)))
                centery = polypyder.r[i]
                maxflag = crossery[int(round(polypyder.r[i].real))]
    centerx = centerx.real
    centery = centery.real
    centerx = (centerx-40)/2+40
    centery = (centery-40)/2+40
    newcen = [380+xcom+(centerx-40), ycom+160+(centery-40)]#in old coordinates
    return image, centerx, centery, newcen

def crossandfindv2(profile):
    """
    Cross-correlates via comparing radial integrals.
    """
    import numpy as np
    import math

    crosserx = np.correlate(profile, np.flip(profile, 0), 'same')
    fitrangemin = int(round(crosserx.shape[0]/2))-30 #slightly expanded fit range to 30
    fitrangemax = int(round(crosserx.shape[0]/2))+30
#-----------Fit correlation and find roots in x-----------------------
    x = np.arange(fitrangemin, fitrangemax, 1)
    pxfit = np.polyfit(x, crosserx[fitrangemin:fitrangemax], 5)
    pxder = np.polyder(pxfit)
    polypxder = np.poly1d(pxder)
    maxflag = 0
    for i in range(0, polypxder.r.shape[0]):
        if polypxder.r[i].imag == 0 and polypxder.r[i] > 0 and polypxder.r[i] < fitrangemax-1 and polypxder.r[i] > fitrangemin:
            if crosserx[int(round(polypxder.r[i].real))] > maxflag:
                centerx = polypxder.r[i]
                maxflag = crosserx[int(round(polypxder.r[i].real))]
    centerx = centerx.real
    centerx = centerx-crosserx.shape[0]/2
    #The difference between the peak position and the center is
    #2*the displacement of the particle from the center
    dr = centerx/2
    #The divided by 3 corrects for differences in scales between radial distribution and xy grid
    dx = dr/(math.pi/2)/3
    return dx

def comv2(imagearray):
    """
    Calculates centre of mass.
    """
    import numpy as np

    imagearray = abs(imagearray-np.median(imagearray))
    xcomtop = 0
    ycomtop = 0
#This is a modified algorithm to try do deal with our very low signal. We subtract
#the background of the sum in one dimension before doing the final COM calculation.
    isum = np.sum(imagearray, 1)
    isum = abs(isum-(np.mean(isum[0:10])+np.mean(isum[(isum.shape[0]-10):isum.shape[0]]))/2)
    jsum = np.sum(imagearray, 0)
    jsum = abs(jsum-(np.mean(jsum[0:10])+np.mean(jsum[(jsum.shape[0]-10):jsum.shape[0]]))/2)
    for i in range(imagearray.shape[0]):
        xcomtop += i*(isum[i])
    for j in range(imagearray.shape[1]):
        ycomtop += j*(jsum[j])
    normi = np.sum(isum)
    normj = np.sum(jsum)
    #We get the center of mass simply by adding the points in each
    #dimension and dividing by number of points.
    xcom = int(round(xcomtop/normi))
    ycom = int(round(ycomtop/normj))
    return xcom, ycom
