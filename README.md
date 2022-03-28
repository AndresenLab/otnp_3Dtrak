# otnp_3Dtrak
Contains code to track particles in 3D as described in our paper: 3D flow field measurements outside nanopores

lutcalc2 takes an array of z positions with times alongside a calibration video of a fixed particle, determining a radial intensity profile for each frame and linking it to the z position from the array, in order to generate a lookup table (LUT).

Details of the various functions in zdetect, as well as how they relate to each other. zdetect outputs an estimate for the particle centre so can also be used for off-line tracking of a particle in plane:

---

Zanalyzevid
Opens video
Take frame
Convert to greyscale
Crop
Input to 'centreandrad2'
	This function is used to get the center of an image and obtain the radial
	    integral profile. It first uses "getcenter2" to obtain the center. It
	    returns a new (cropped) image and the center (xcen, ycen) in that image.
	    It also returns the center of the image in the coordinates of the original
	    uncropped image (confusing named newcenx, newceny). The found center and
	    cropped image is then fed into the radial integrator program which takes the
	    radial integral quandrant-by-quadrant and then returns these integrals. In the
	    final line we just average them and return the integral profile as well as the
	    center in the original coordinates.
		Uses:
		'getcenter2'
		Run the cross correlation algorithm to find center. This first
		    performs COM centering, crops the file, and then cross correlates to
		    find the center.
			Uses:
			'crosscorv2'
			Takes an image cropped to image = image[160:280, 380:500] and then finds
			    the center via COM and then cross-correlation.
				Uses:
				'comv2'
				This increases dynamic range by first subtracting the median of the image
				Then the background is subtracted in each dimension
				Finally the pixels in each dimension are summed and divided by the total number of pixels
				This returns the centre of mass calc on the input frame
			The COM is fed back:
		Uses the COM values to define the centre of a new ROI 40 pixels either side of centre, eg if COM is 40, 40, the ROI will be (0, 0) (80, 80). After that cross-correlation is performed in same manner as xy tracking to better define centre, except 5th order polynomial is fit to cross-correlation output. This defines the centre of the particle and returns it.
	The centre is fed back:
	'centreandrad2' now uses:
	'radintv2'
	Seperates image into 4 quadrants, interpolates all pixels,
	    calculates radial integral per quadrant. 'dr' is 1/3 of a pixel
	    Example usage:
	        [qtr, qtl, qbr, qbl] = radintv2(newimage, xcen, ycen)
	    Separate into 4, interpolate between pixels using rectbivariate spline
		Uses:
		'pol2cart'
		Converts polar coordinates to cartesian, all in 2D.
	   x and y positions are fed back and then used to calculate the integral in 'radintv2'
	The integrals are fed back:
'centreandrad' now returns the radial profile (avg of the 4 radial profiles) and the centre
The radial profile is now inputted to 'calcheight3'
	Compares radial integral of frame to lookup table (LUT), fits chi square
	    and finds minimum of fit. Chi square takes the difference between the frame's profile and each profile in the LUT, squares that and divides the result by that LUT profile, point by point. It finds the minimum chi^2 value, then takes the region approaching that minimum and fits a quadratic equation to it get sub-LUT accuracy. Gives back answer as z-position.
      
---
      
LUTfine.npy is an example LUT.

filedialog is a simple tkinter function to enable selection of a single file or folder using the Finder/Windows Explorer interface.

FolderWatch2 monitors the contents of a chosen directory and generates a dictionary of its contents when updated.

tweezers_z_processor uses zdetect to determine the z position of a particle over the length of a recorded video.

XYZ_TotalForce combines force data generated using the z tracking with x,y force data generated in LabVIEW.
