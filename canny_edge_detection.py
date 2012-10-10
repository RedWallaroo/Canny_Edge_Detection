
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/pil")
sys.path.append("/usr/local/lib/python2.7/site-packages/scipy")
import Image
import pdb
import os
from scipy import *
from scipy.ndimage import *
from scipy.signal import convolve2d as conv
import numpy as np
from numpy import *

def Myimread(imname, flatten):
	
		#img = Image.open("/users/sandralombardi/ppm images/girlwithdog.ppm")
		im = Image.open(imname)
		if flatten:
			im = im.convert('F')
			#After converting the image to grayscale, we put it into a two-dimensional array. 
			#each pair represents a pixel
			result = array(im)
			return result

class Cannydetector:

	'''
		Cannydetector: The Canny edge detector is an edge detection operator that uses a multi-stage 
		algorithm to detect a wide range of edges in images. It was developed by John F. Canny in 1986. 
		Canny also produced a computational theory of edge detection explaining why the technique works.

		The steps to perform Canny edge detection implemented in this program are:

		1 - Read Image into 2D array (achieved using PIL)
		2 - Noise Reduction by means of applying a Gaussian blur.
			- Create Gaussian kernel: def makegausskernel()
			- Apply filter kernel to original image by means of convolution: def FFTConvolve()
		3 - Find intensity gradient using a Sobel operator
			- Declare 5x5 Sobel filters
			- Apply Sobel filters to blurred image using convolution: def FFTConvolve()
			- Calculate gradient magnitude
			- Calculate gradient angle/direction
		4 - Perform non-maxima supression:  
			- Create angle buckets:
			- Determine angle direction:
			- Discard values: def Non_Maxima_Supression()
		5 - Perform clean-up through Hysteresis 

	'''

	def __init__(self,imname, sigma,thresHigh = 80, thresLow = 65):
	
		
		'''	STEP ONE: Read image '''
		self.imin = Myimread(imname,flatten = True)
		
		'''	STEP TWO: Noise Reduction

			Calling function to create kernel/core for gaussian filter.
		   	this will be convolved with the image to create the blur'''
		
		gausskernel = self.makegausskernel(sigma,5)
		
		''' Convolution: convolution is an operation on two functions f and g, which produces a third function 
			that can be interpreted as a modified version of f. In 2d convolution, you have an input 2d array
			and a kernel (2d array as well). The convolution process begins by flipping the kernel 
			(horizontally and vertically), then overlapping each point from the input array with the center of 
			kernel and multiplying each point in the kernel + the overlap. The resulting number is the value that
			belongs to that position in the output array.

			In this call, the input image and the kernel for the gaussian blur is being passed into a 
			convolution 2d function from the scipy.signal library. Normal convolution is slower than separable 
			convolution. so this function from scipy might not be the best:

			convolve2d(input,kernel)[1:-1,1:-1]

			Difference between normal convolution and separable convolution
			http://www.songho.ca/dsp/convolution/convolution2d_separable.html

			Basically, you can perform two 1-d convolutions to get the 2d convolution. It does not matter
			which you choose to do first (horizontal or vertical)

		'''
	
		#Using FFTConvolve as it is the most stable/accurate right now.	
		self.imout = self.FFTConvolve(self.imin,gausskernel)
		#imout = self.Separable_Convolution(self.imin,gausskernel)
		
		''' Preview after blur '''
		im = Image.fromarray(self.imout)
		im.show()

		

		''' STEP THREE: Find intensity gradient using Sobel operator

			These are 5x5 Sobel filters:
			The sobel operator is used in image processing to compute an approximation
			of the gradient of the image intensity function. At each point in the image,
			the result of the Sobel operator is either the corresponding gradient vector
			or the norm of this vector. 

		'''

		fx = self.createFilter([1, 2, 0, -2, -1,
								4, 8, 0, -8, -4,
								6, 12, 0, -12, -6,
								4, 8, 0, -8, -4,
								1, 2, 0, -2, -1])

		fy = self.createFilter([-1, -4, -6, -4, -1,
								-2, -8, -12, -8, -2,
								 0, 0, 0, 0, 0,
								 2, 8, 12, 8, 2,
								 1, 4, 6, 4, 1])

		''' We need to apply convolution to the Sobel operator filters and the image
			before we can calculate the gardient magnitude. This process will result
			in two edge maps. One horizontal and one vertical. The final edge map
			is a combination of these two edge maps. I'll try and use my convolution
			FFT function for this
		'''

		gradx = self.FFTConvolve(self.imout,fx)
		grady = self.FFTConvolve(self.imout,fy)

		'''	Calculating grad magnitude....
			The formula for gradient magnitude is |g| = sqrt(gx^2 + gy^2)
		'''
		#gradmagnitude = (0.5+sqrt(pow(gradx,2.0) + pow(grady,2.0)))
		gradmagnitude =  (sqrt(pow(gradx,2.0) + pow(grady, 2.0)))

		'''	The angle of orientation of the edge (relative to the pixel grid)
			giving rise to the spatial gradient is givn by: 
			theta = arctan(gy/gx) 
		'''
		angle = arctan2(gradx,grady)
		angle = 180 + (180/pi)*angle

		#angle = arctan2(gradx,grady)*(180/pi)

		x,y = where(gradmagnitude < 10)
		angle[x,y] = 0
		gradmagnitude[x,y] = 0

		''' Preview of angle map '''

		self.angle = angle
		anglemap = Image.fromarray(self.angle).convert('L')
		anglemap.show()

		

		'''	STEP FOUR: Perform non-maxima supression

			Now it is time to remove edges below a certain threshold. This helps
			in keeping only the major edges. For this purpose, we need to apply 
			the non-maximum suppression algorithm. Basically, for a given pixel,
			there are 8 neighboring pixels. Hence, there are 4 possible directions:
			0 degrees, 45 degrees, 90 degrees, and 135 degrees. 

			The angles are hence quantized into these 4 angle buckets. Then for every
			pixel in the image, we look at the angle of the gradient at that point. 
			For example, if the angle of the gradient is 0 degrees. This implies that the 
			direction of the edge should be perpendicular to it (run along the 90 degrees line)
			Hence, we check for the left and right neighboring gradient values. If the 
			gradient at the present pixel is greater than its left and right neighbors, then 
			we can safely consider it an edge. If not, we discard it. This process results
			in sigle pixel thick edges. 
		'''

		'''Allocating edges of image map in angle buckets'''
		self.Edge_Direction(self.angle)

		''' Now we perform the non-maxima supressions '''
		self.gradmagnitude = gradmagnitude.copy()
		self.mag_sup= self.Non_Maxima_Supression(self.angle, self.gradmagnitude)

		''' STEP FIVE: Clean up using Hysteresis:

			Hysteresis makes the assumption that important edges should be along continuous 
			curves. By providing two thresholds (high and low), an edge based scanning can be
			performed where detected edges that do not seem to belong to a line will
			be discarded. This will help clean up the edge map previously obtained through 
			non-maxima supression.
		'''
		
		self.gradmagnitude = self.Hysteresis(self.mag_sup)

	def makegausskernel(self,sigma,windowsize = 3):

		''' this creates an array of zeros using the windowsize dimensions.
			in this case it is a 5 x 5 matrix. 3x3 matrices are used most commonly.
		'''

		kernel = zeros((windowsize, windowsize))
		center = windowsize // 2

		'''Passed windowsize 5 so center is 2. 
			We need the center variable because the Gaussian formula will
			create a surface whose contours are concentric circles with a 
			Gaussian distribution from the center point.'''

		for x in range(windowsize):
			for y in range(windowsize):
				''' radious is the x^2+y^2 in the Gaussian function.
					hypot in python works like this: 
					hypot(x,y) = sqrt((x^2)+(y^2))
				''' 
				radious = hypot((x-center),(y-center))
				
				''' fx is the two-dimensional version of the Gaussian function
					see link for more info: http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
				'''
				fx = (1.0/2*pi*sigma*sigma)*exp(-(radious*radious)/(2*sigma*sigma))
				'''we assign the result of the Gaussian function to each pixel in kernel.'''
				kernel[x,y] = fx
				''' calculate the weighted mean and return results
					the values returned in this kernel will be used for as the convolution
					matrix. convolution is the treatment of a matrix by another one which is called
					a 'kernel' 
					Convolution: It is a mathematical operation on two functions f ang g, producing a third
					function that is typically viewed as a modified version of one of the original
					functions
				'''
		
		return kernel / kernel.sum()
	
	
		#convolution function
	def NormalConvolve(self,input,kernel):

		if kernel.shape[0]%2 != 1 or kernel.shape[1] % 2 != 1:
			raise ValueError("Only odd dimensoons on filter supported")

		inputAmax = input.shape[0]
		inputBmax = input.shape[1]
		kernelAmax = kernel.shape[0]
		kernelBmax = kernel.shape[1]
		kernelAmid = kernelAmax // 2
		kernelBmid = kernelBmax // 2
		xMax = inputAmax + 2*kernelAmid
		yMax = inputBmax + 2*kernelBmid

		#Allocate result image

		output = np.zeros([xMax,yMax], dtype=input.dtype)

		#Do convolution

		for x in range(xMax):
			for y in range(yMax):
				s_from = max(kernelAmid - x, - kernelAmid)
				s_to = min((xMax - x) - kernelAmid, kernelAmid + 1)
				t_from = max(kernelBmid - y, - kernelBmid)
				t_to = min((yMax - y) - kernelBmid, kernelBmid + 1)
				value = 0 
				for s in range (s_from, s_to):
					for t in range(t_from, t_to):
						v = x - kernelAmid + s
						w = y - kernelBmid + t
						value += kernel[kernelAmid - s, kernelBmid - t] * input[v,w]
				output[x,y] = value
		return output

	def FFTConvolve(self, input, kernel):

		''' this function for convolution takes care of the edge padding
			by replacing the edges with pixels from the original image edges
			instead of using zeros. It then uses the FFT functions to do the 
			convolution. 
			I'm using this instead of the normal convolution or convolve2d 
			because this one is a lot faster

		'''

		inputX, inputY = input.shape
		kernelX, kernelY = kernel.shape

		Xcord = inputX + kernelX - 1
		Ycord = inputY + kernelY - 1
		#padX = (inputX - kernelX) / 2
		#padY = (inputY - kernelY) / 2

		padX = 2
		padY = 2

		#pad the edges with the same values found on edges

		Lside = np.empty((input.shape[0], padY))
		Rside = np.empty((input.shape[0], padY))

		for i in range(padY):
			Lside[:,i] = np.copy(input[:,0]).T
			Rside[:,i] = np.copy(input[:,input.shape[1]-1]).T

		input = np.hstack((Lside,input,Rside))
		top = np.empty((padX,input.shape[1]))
		bottom = np.empty((padX,input.shape[1]))

		#pad top and bottom

		for i in range(padX):
			top[i] = np.copy(input[0,:]).T
			bottom[i] = np.copy(input[input.shape[0]-1,:]).T

		input = np.vstack((top,input,bottom))

		xOrig = Xcord
		yOrig = Ycord

		pX2 = int(log(Xcord)/log(2.0) + 1.0)
		pY2 = int(log(Ycord)/log(2.0) + 1.0)

		Xcord = 2**pX2
		Ycord = 2**pY2

		FFTimage = np.fft.fft2(input, s=(Xcord, Ycord)) * np.fft.fft2(kernel[::-1,::-1], s=(Xcord,Ycord))

		output = np.fft.ifft2(FFTimage).real

		return output[(xOrig - inputX):xOrig, (yOrig - inputY):yOrig]
		

	def Separable_Convolution(self,input,kernel):

		''' this is a failed attempt at implementing separable convolution.
		for some reason, python does not allow for the multiplication of the 
		5 row kernel times the 512 row set of the input image.
		after trying to process a 5 x 5 multiplication at the time, it turned out that
		after the resulting array grows over 300 items, the values of the items
		were changed and looked weird. perhaps i need to keep trying this though.

		'''
		#inputs original image and kernel then it applies a 1D convolution horizontally
		#and then vertically

		rows,cols = input.shape
		kernelRows, kernelColumns = kernel.shape
		#convolve horizontal direction
		#find kernel center for x
		kernelXcenter = kernelRows // 2
		kernelYcenter = kernelColumns // 2

		xMax = rows + 2 * kernelXcenter
		yMax = cols + 2 * kernelYcenter
		endYindex = cols - kernelYcenter #512 - 2 = 510
		ksizeX = kernel.shape[0]

		#Need to pad output image to 516 for edges

		#Allocate result image
		output = np.zeros([xMax,yMax], dtype=input.dtype)
		

		# START OF HORIZONTAL CONVOLUTION
		#start horizontal convolution up to total number of rows
		for r in range(rows):
			kOffset = 0
			#run through columns 0 and 1
			#column from index = 0 to index = kcenter-1
			for c in range(kernelXcenter):
				m = 0
				value = 0
				for k in range(kernelXcenter + kOffset,-1,-1):
					value += input[r,c + m] * kernel[k,0]
					m += 1
				output[r,c] = value
				kOffset += kOffset


			#run through columns 2 and 509
			#column from index = kcenter to index (datasizex - kcenter - 1)
			for c in range(kernelXcenter,endYindex + 1):
				m = 0
				value = 0
				
				for k in range(ksizeX - 1,-1,-1):
					inputCol = c+m
					if inputCol <= cols-1:
						value += input[r,c + m] * kernel[k,0]
						m += 1
				output[r,c] = value
			
			kOffset = 1

			#run through columns 510 and 511
			#column from index = kcenter to index (datasizex - 1)
			for c in range(endYindex, cols):
				m = 0
				value = 0
				for k in range(ksizeX - 1,kOffset,-1):
					inputCol = c+m
					if inputCol <= cols -1:
						value += input[r,c + m] * kernel[k,0]
						m += 1
				output[r,c] = value

		# END OF HORIZONTAL CONVOLUTION
		
		# START OF VERTICAL CONVOLUTION	

		for r in range(rows):
			kOffset = 0
			#run through columns 0 and 1
			#column from index = 0 to index = kcenter-1
			for c in range(kernelXcenter):
				m = 0
				value = 0
				for k in range(kernelXcenter + kOffset,-1,-1):
					value += input[r,c + m] * kernel[k,0]
					m += 1
				output[r,c] = value
				kOffset += kOffset


			#run through columns 2 and 509
			#column from index = kcenter to index (datasizex - kcenter - 1)
			for c in range(kernelXcenter,endYindex + 1):
				m = 0
				value = 0
				
				for k in range(ksizeX - 1,-1,-1):
					inputCol = c+m
					if inputCol <= cols-1:
						value += input[r,c + m] * kernel[k,0]
						m += 1
				output[r,c] = value
			
			kOffset = 1

			#run through columns 510 and 511
			#column from index = kcenter to index (datasizex - 1)
			for c in range(endYindex, cols):
				m = 0
				value = 0
				for k in range(ksizeX - 1,kOffset,-1):
					inputCol = c+m
					if inputCol <= cols -1:
						value += input[r,c + m] * kernel[k,0]
						m += 1
				output[r,c] = value	

		return output

	def createFilter(self,rawfilter):
	
		order = pow(len(rawfilter),0.5)
		order = int(order)
		filt_array = array(rawfilter)
		outfilter = filt_array.reshape((order,order))
		return outfilter	

	def Edge_Direction(self,angle):

		x,y = self.angle.shape
		#allocating output image
		temp = Image.new('RGB',(y,x),(255,255,255))

		for i in range(x):
			for j in range(y):
				if (self.angle[i,j] < 22.5 and self.angle[i,j] >=0) or \
				(self.angle[i,j] >= 157.5 and self.angle[i,j] < 202.5) or \
				(self.angle[i,j] >= 337.5 and self.angle[i,j] <= 360):
					self.angle[i,j] = 0
					temp.putpixel((j,i),(0,0,255))
				elif (self.angle[i,j] >= 22.5 and self.angle[i,j] < 67.5) or \
				(self.angle[i,j] >= 202.5 and self.angle[i,j] < 247.5):
					self.angle[i,j] = 45
					temp.putpixel((j,i),(255,0,0))
				elif (self.angle[i,j] >= 67.5 and self.angle[i,j] < 112.5) or \
				(self.angle[i,j] >= 247.5 and self.angle[i,j] < 292.5):
					self.angle[i,j] = 90
					temp.putpixel((j,i),(255,255,0))
				else:
					self.angle[i,j] = 135
					temp.putpixel((j,i),(0,255,0))
		
		temp.show()

	def Non_Maxima_Supression(self,angle,gradmagnitude):

		mag_sup = self.gradmagnitude.copy()
		x,y = self.gradmagnitude.shape

		

		for i in range(x-1):
			for j in range(y-1):
				if self.angle[i,j]==0:
					if (self.gradmagnitude[i,j]<=self.gradmagnitude[i,j+1]) or \
						(self.gradmagnitude[i,j]<=self.gradmagnitude[i,j-1]):
							mag_sup[i,j]=0
				elif self.angle[i,j]==45:
					if (self.gradmagnitude[i,j]<=self.gradmagnitude[i-1,j+1]) or \
						(self.gradmagnitude[i,j]<=self.gradmagnitude[i+1,j-1]):
							mag_sup[i,j]=0
				elif self.angle[i,j]==90:
					if (self.gradmagnitude[i,j]<=self.gradmagnitude[i+1,j]) or \
						(self.gradmagnitude[i,j]<=self.gradmagnitude[i-1,j]):
							mag_sup[i,j]=0
				else:
					if (self.gradmagnitude[i,j]<=self.gradmagnitude[i+1,j+1]) or \
						(self.gradmagnitude[i,j]<=self.gradmagnitude[i-1,j-1]):
							mag_sup[i,j]=0
		

		im = Image.fromarray(mag_sup)
		im.show()
		return mag_sup
	

	def Hysteresis(self,mag_sup):

		
		x,y = self.mag_sup.shape
		thresHigh = 1100
		thresLow = 700

		for j in range(y):
			for i in range(x):
				if (self.mag_sup[i,j] < thresLow):
					self.mag_sup[i,j] = 0
				if (self.mag_sup[i,j]> thresHigh):
					self.mag_sup[i,j] = 255

		#if pixel (x,y) has gradient magnitude between threhlow and threshigh and any of its neighbors in a 
		# 3x3 region around it have gradient magnitude gradient that threshigh, keep the edge

		if (self.mag_sup[i,j] >= thresLow) and (gself.mag_sup[i,j] <= thresHigh):
			greaterFound = False
			betweenFound = False
			for n in range(x):
				for m in range(y):
					if (self.mag_sup[i+n,j+m] > thresHigh):
						self.mag_sup[i,j] = 255
						greaterFound = True
					if (self.mag_sup[i,j] > thresLow) and (self.mag_sup[i,j] < thresHigh):
						betweenFound = True

			if ( not greaterFound  and betweenFound):
				for n in range(x):
					for m in range(y):
						if(self.mag_sup[i+n,j+m] > thresHigh):
							greaterFound = True

			if greaterFound:
				self.mag_sup[i,j] = 255
			else:
				self.mag_sup[i,j] = 0

		return mag_sup


#canny = Cannydetector('/users/sandralombardi/ppm images/girlwithdog.ppm',1.4,50,10)
#im = canny.gradmagnitude
#Image.fromarray(im).show()

canny = Cannydetector('/users/sandralombardi/downloads/test3.jpg',1.4,50,10)
im = canny.gradmagnitude
Image.fromarray(im).show()

