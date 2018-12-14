import sys
import os
import argparse
import re

from time import sleep
from math import sin, cos, pi

import SimpleITK as sitk

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from mpl_toolkits.mplot3d import axes3d

import numpy as np
from scipy import stats
from scipy import interpolate

import pandas as pd

from skimage.draw import polygon, circle_perimeter_aa, circle, line_aa
from skimage.measure import find_contours
from skimage.io import imread, imsave

from Analyser import Analyser

def toPolar(x,y):
	r = np.sqrt( np.square(x) + np.square(y) )
	t = np.arctan2(y,x)
	return r,t

def binnedStatistic(x, vals, mask, nBins=100):
	statistic, edges, binnumber = stats.binned_statistic(x[mask>0].flatten(), vals[mask>0].flatten(), bins = nBins)
	x = 0.5* (edges[:-1] + edges[1:])
	return x, statistic

class SectorAnalyser(Analyser):
	def __init__(self, dataFileName, outputDir):
		Analyser.__init__(self, dataFileName, outputDir)

	def addTarget(self, verts):
		print("Added target[{0}]".format(len(verts)))

		self.targets.append( [ np.array( vert ) for vert in verts ] )

	def computePolygon(self, verts, n):
		verts = [ np.array( vert ) for vert in verts ]
		newVerts = []

		for i in range( len(verts) ):
			j = (i + 1) % len(verts)
			newVerts = newVerts + [ verts[i] + (verts[j] - verts[i]) * k / n for k in range(n+1) ]

		self.addTarget( newVerts )

	def computeCircle(self, event, radius, n):
		verts = [ ( radius * cos( 2*pi * a/n ) + event.xdata, radius * sin( 2*pi * a/n ) + event.ydata )  for a in range(n) ]
		self.addTarget(verts)

	def getTargets(self, image, method, **kwargs):
		print("Add targets...\n")
		
		if method in ['poly', 'point']:
			figure = plt.figure()
			ax = plt.gca()
			plt.imshow(image)

		radius = kwargs['radius'] if 'radius' in kwargs else 2
		n = kwargs['n'] if 'n' in kwargs else 100

		if method == 'poly':
			ps = PolygonSelector(ax, lambda verts: self.computePolygon(verts, n = n) )
			plt.show()
		elif method =='point':
			figure.canvas.mpl_connect('button_press_event', lambda event: self.computeCircle(event, radius = radius, n = n) )
			plt.show()
		else:
			self.getTargetsFromImage( kwargs['L'], n )

	def labelTargets(self, t):
		for label, target in enumerate(self.targets):
			rr, cc = polygon( [p[1] for p in target], [p[0] for p in target] )
			self.L[t,rr,cc] = label+1

	def getTargetsFromImage(self, L, n):
		L = imread(L)
		for i in range( L.max() ):
			
			LI = np.copy(L)
			LI[ L!=i+1 ] = 0

			print (L.shape, LI.shape)

			c = find_contours(LI, 0)[0]
			print (len(c))
			step = len(c) // n if len(c) // n > 0 else 1
			c = c[::step]
			c[:, 0], c[:, 1] = c[:, 1], c[:, 0].copy()

			self.addTarget( c.tolist() )



	def trackPointField(self, method, **kwargs):
		print("\n########## Tracking point field forward ##########\n")
		
		self.L = np.zeros( ( self.nFrames, self.frameSize[0], self.frameSize[1] ), dtype = np.uint8 )

		self.getTargets(self.data["I"][0,:,:,0], method, **kwargs)


		targetRange = range(len(self.targets))

		pointXLabel = 'X {0}'
		pointYLabel = 'Y {0}'

		pointXColumns = [ pointXLabel.format(i) for i in targetRange ]
		pointYColumns = [ pointYLabel.format(i) for i in targetRange ]
		timeColumn = 'Time'

		columns = [timeColumn] + pointXColumns + pointYColumns
		targetData = pd.DataFrame(columns = columns)

		targetData[timeColumn] = [0]
		
		for i, points in enumerate(self.targets):
			meanPoint = np.mean(self.targets[i], axis = 0)
			targetData[pointXLabel.format(i)] = [meanPoint[0]]
			targetData[pointYLabel.format(i)] = [meanPoint[1]]

		print("Processing targets...\n")

		if self.targets:
			print (self.targets)
			self.labelTargets(0)

			for t in range( self.nFrames-1 ):
				transform = self.compileTransform(t, inverse = True)

				newData = pd.DataFrame( columns = columns )
				newData[timeColumn] = [ (t+1) * self.data['metadata']['dt']]

				for i, points in enumerate(self.targets):
					self.targets[i] = [  np.array( transform.TransformPoint( point ) ) for point in points ]
					
					meanPoint = np.mean(self.targets[i], axis = 0)
					newData[pointXLabel.format(i)] = [meanPoint[0]]
					newData[pointYLabel.format(i)] = [meanPoint[1]]

				self.labelTargets(t+1)
				targetData = targetData.append(newData, ignore_index = True)
				# print ("targetData:", targetData)

			sL = sitk.GetImageFromArray(self.L)
			sI = sitk.GetImageFromArray(self.data['I'][:,:,:,0])

			preview = sitk.LabelOverlay( sitk.GetImageFromArray(self.data['I'][:,:,:,0]), sL)
			
			self.saveImage( preview, self.nextFileName("track") )
			
			sitk.Show( preview )

			targetDataFileName = os.path.join( self.outputDir, "{0}_{1}.tsv".format(self.name, self.nextFileName('targetData', ext = 'tsv')) )
			targetData.to_csv(targetDataFileName, sep = '\t', header = True)


	def plotTrajectory(self):
		trajectory = np.zeros( ( self.nFrames, self.frameSize[0], self.frameSize[1] ), dtype = np.uint8 )
		self.getTargets(self.data["I"][0,:,:,0], 'point', n = 1, r = 1)

		if self.targets:
			for t in range(self.nFrames-1):
				transform = self.compileTransform(t, inverse = True)

				for i, points in enumerate(self.targets):
					self.targets[i] = [  np.array( transform.TransformPoint( point ) ) for point in points ]

					rr,cc,val = circle_perimeter_aa( int(points[0][1]), int(points[0][0]), 2)
					trajectory[t,rr,cc] = 255*val

					rr,cc = circle( int(points[0][1]), int(points[0][0]), 2 )
					trajectory[t,rr,cc] = 255
					

		self.saveImage( sitk.GetImageFromArray(trajectory), self.nextFileName("trajectory") )


	def trackPointFieldRev(self, method):
		print("\n########## Tracking point field reverse ##########\n")
		self.L = np.zeros( ( self.nFrames, self.frameSize[0], self.frameSize[1] ), dtype = np.uint8 )

		self.getTargets(self.data["I"][-1,:,:,0], method)
		
		print("Processing targets...\n")

		if self.targets:
			self.labelTargets( self.nFrames - 1)

			for t in range(self.nFrames-2, -1,-1):
				transform = self.compileTransform(t, inverse = False)

				for i, points in enumerate(self.targets):
					self.targets[i] = [  np.array( transform.TransformPoint( point ) ) for point in points ]

				self.labelTargets(t)

			sL = sitk.GetImageFromArray(self.L)
			sI = sitk.GetImageFromArray(self.data['I'][:,:,:,0])

			preview = sitk.LabelOverlay( sitk.GetImageFromArray(self.data['I'][:,:,:,0]), sL)
			
			self.saveImage( preview, self.nextFileName("trackR") )
			
			sitk.Show( preview )

	def getJacobianMapFwd(self):
		print("\n########## Calculating Jacobian ##########\n")

		print("Progress:")

		J = sitk.Image( self.frameSize[0], self.frameSize[1], self.nFrames, sitk.sitkFloat32 )
		displacementFilter = sitk.TransformToDisplacementFieldFilter()

		for t in range(self.nFrames):
			print("{0}/{1}".format(t, self.nFrames))

			compositeTransform = sitk.Transform(2, sitk.sitkComposite)
			
			for t in range(self.nFrames-2,t-1,-1):
				transform = self.compileTransform(t, inverse = True)
				compositeTransform.AddTransform(transform)

			displacementFilter = sitk.TransformToDisplacementFieldFilter()
			displacementFilter.SetReferenceImage( sitk.GetImageFromArray( self.data["I"][t,:,:,0] ) )
			
			U = displacementFilter.Execute(compositeTransform)

			UX = sitk.VectorIndexSelectionCast(U,0)
			UY = sitk.VectorIndexSelectionCast(U,1)

			dUX = sitk.Gradient( UX )
			dUY = sitk.Gradient( UY )

			dUXX = sitk.VectorIndexSelectionCast(dUX, 0)
			dUXY = sitk.VectorIndexSelectionCast(dUX, 1)

			dUYX = sitk.VectorIndexSelectionCast(dUY, 0)
			dUYY = sitk.VectorIndexSelectionCast(dUY, 1)

			jacobian = sitk.Mask( (dUXX+1) * (dUYY+1) - dUXY * dUYX, sitk.GetImageFromArray(self.data["M"][t,:,:]) )
			jacobian = sitk.Mask( jacobian, (jacobian <= 100) )
			jacobian = sitk.Mask( jacobian, (jacobian >= 1 ) )

			J = sitk.Paste( J, sitk.JoinSeries(jacobian), jacobian.GetSize(), destinationIndex = [0, 0, t] )

		W = sitk.GetImageFromArray(self.data["I"][:,:,:,0])
		M = sitk.GetImageFromArray(self.data["M"] * 50)
		iJ = sitk.Cast(J, sitk.sitkUInt8) * 5
		
		preview = sitk.Compose(  M, W, iJ )
		
		self.saveImage(J, "J0")
		self.saveImage(preview, "JPreview")

		sitk.Show( preview )
	
	def getJacobianMap(self):
		print("\n########## Calculating Jacobian ##########\n")

		print("Progress:")

		J = sitk.Image( self.frameSize[0], self.frameSize[1], self.nFrames, sitk.sitkFloat32 )
		displacementFilter = sitk.TransformToDisplacementFieldFilter()

		for t in range(self.nFrames):
			print("{0}/{1}".format(t, self.nFrames))

			# nX = np.indices( (self.frameSize[0], self.frameSize[1]) )
			
			# X = sitk.Cast( sitk.GetImageFromArray(nX[1]), sitk.sitkFloat64 )
			# Y = sitk.Cast( sitk.GetImageFromArray(nX[0]), sitk.sitkFloat64 )

			compositeTransform = sitk.Transform(2, sitk.sitkComposite)
			
			for tau in range(t):
				transform = self.compileTransform(tau)
				compositeTransform.AddTransform(transform)
			
			displacementFilter.SetReferenceImage( sitk.GetImageFromArray( self.data["I"][t,:,:,0] ) )
			U = displacementFilter.Execute(compositeTransform)

			UX = sitk.VectorIndexSelectionCast(U,0)
			UY = sitk.VectorIndexSelectionCast(U,1)

			dUX = sitk.Gradient( UX )
			dUY = sitk.Gradient( UY )

			dUXX = sitk.VectorIndexSelectionCast(dUX, 0)
			dUXY = sitk.VectorIndexSelectionCast(dUX, 1)

			dUYX = sitk.VectorIndexSelectionCast(dUY, 0)
			dUYY = sitk.VectorIndexSelectionCast(dUY, 1)

			jacobian = sitk.Mask( 1 / ( (dUXX+1) * (dUYY+1) - dUXY * dUYX), sitk.GetImageFromArray(self.data["M"][t,:,:]) )
			
			jacobian = sitk.Mask( jacobian, (jacobian <= 100) )
			jacobian = sitk.Mask( jacobian, (jacobian >= 1 ) )
			
			J = sitk.Paste( J, sitk.JoinSeries(jacobian), jacobian.GetSize(), destinationIndex = [0, 0, t] )

		W = sitk.GetImageFromArray(self.data["I"][:,:,:,0])
		M = sitk.GetImageFromArray(self.data["M"] * 50)
		iJ = sitk.Cast(J, sitk.sitkUInt8) * 5
		
		preview = sitk.Compose(  M, W, iJ )
		
		self.saveImage(J, "J")
		self.saveImage(preview, "JPreview")

		sitk.Show( preview )

	def getVelocity(self,t):
		displacementFilter = sitk.TransformToDisplacementFieldFilter()
		displacementFilter.SetReferenceImage( sitk.GetImageFromArray( self.data["I"][t,:,:,0] ) )

		forwardTransform = self.compileTransform(t-1)
		inverseTransform = self.compileTransform(t, inverse = True)

		forwardU = -displacementFilter.Execute(forwardTransform)
		backwardU = displacementFilter.Execute(inverseTransform)

		v2 = backwardU + forwardU

		vX = sitk.VectorIndexSelectionCast(v2, 0) * 0.5 / self.data["metadata"]["dt"]
		vY = sitk.VectorIndexSelectionCast(v2, 1) * 0.5 / self.data["metadata"]["dt"]

		return vX, vY

	def getTransformGrid(self, transform):
		grid = sitk.GridSource(outputPixelType=sitk.sitkUInt8, size=(self.frameSize[0], self.frameSize[1]), sigma=(0.05, 0.05), gridSpacing=(10.0, 10.0), gridOffset=(0.0, 0.0), spacing=(1.0,1.0))
		interpolator = sitk.sitkCosineWindowedSinc
		val = 1000.0
		return sitk.Resample(grid, grid, transform, interpolator, val)

	def getTenaformViz(self):
		transformStackF = sitk.Image( [self.frameSize[0], self.frameSize[1], self.nFrames - 1], sitk.sitkUInt8)
		transformStackR = sitk.Image( [self.frameSize[0], self.frameSize[1], self.nFrames - 1], sitk.sitkUInt8)

		uF = np.ones( (self.nFrames, self.frameSize[0], self.frameSize[1]), dtype = 'uint8' ) * 255
		uR = np.ones( (self.nFrames, self.frameSize[0], self.frameSize[1]), dtype = 'uint8' ) * 255

		for t in range(1, self.nFrames-1):

			displacementFilter = sitk.TransformToDisplacementFieldFilter()
			displacementFilter.SetReferenceImage( sitk.GetImageFromArray( self.data["I"][t,:,:,0] ) )

			forwardTransform = self.compileTransform(t-1)
			inverseTransform = self.compileTransform(t, inverse = True)

			gridF = sitk.JoinSeries ( self.getTransformGrid(forwardTransform) )
			gridR = sitk.JoinSeries ( self.getTransformGrid(inverseTransform) )

			transformStackF = sitk.Paste( transformStackF, gridF, gridF.GetSize(), destinationIndex=[0,0,t] )
			transformStackR = sitk.Paste( transformStackR, gridR, gridR.GetSize(), destinationIndex=[0,0,t] )

			forwardU = displacementFilter.Execute(forwardTransform)
			backwardU = displacementFilter.Execute(inverseTransform)

			uF[t] = self.getDisplacementMap( sitk.VectorIndexSelectionCast(forwardU, 0), sitk.VectorIndexSelectionCast(forwardU, 1), 60, 2, mask = self.data['M'][t] )
			uR[t] = self.getDisplacementMap( sitk.VectorIndexSelectionCast(backwardU, 0), sitk.VectorIndexSelectionCast(backwardU, 1), 60, 2, mask = self.data['M'][t] )

		self.saveImage(transformStackF, 'gridTransformF')
		self.saveImage(transformStackR, 'gridTransformR')

		self.saveImage(sitk.GetImageFromArray(uF), 'displacementF')
		self.saveImage(sitk.GetImageFromArray(uR), 'displacementR')

	def getDisplacementMap(self, xx, yy, steps, scale, mask):
		xx = sitk.GetArrayFromImage(xx)
		yy = sitk.GetArrayFromImage(yy)

		dx = int(self.frameSize[0] / steps)
		dy = int(self.frameSize[1] / steps)

		dmap = np.ones( (self.frameSize[0], self.frameSize[1]), dtype = 'uint8' ) * 255

		for x in range(0, self.frameSize[0], dx):
			for y in range(0, self.frameSize[1], dy):

				if mask[y,x] > 0:
					rr,cc,val = circle_perimeter_aa( y, x, 2 )
					print(rr,cc)
					dmap[rr,cc] = 255 - (val * 58)
					
					rr,cc,val = line_aa( y, x, int(y + yy[y,x] * scale), int(x + xx[y,x] * scale) )
					print(rr,cc)
					dmap[rr,cc] = 255- (val * 255)
		return dmap

	def plotVelocityField(self, steps, scale):
		v = np.ones( (self.nFrames, self.frameSize[0], self.frameSize[1]), dtype = 'uint8' ) * 255
		s = np.zeros( (self.nFrames, self.frameSize[0], self.frameSize[1]), dtype = 'float32' )

		x0 = np.zeros( (self.nFrames, self.frameSize[0], self.frameSize[1]), dtype = 'uint8' )
		x, y = np.meshgrid( np.linspace(0, self.frameSize[0]), np.linspace(0, self.frameSize[1]) )

		dx = int(self.frameSize[0] / steps)
		dy = int(self.frameSize[1] / steps)

		for t in range(1, self.nFrames-1):
			vX, vY = self.getVelocity(t)
			mask = self.data['M'][t]

			vX = sitk.GetArrayFromImage(vX)
			vY = sitk.GetArrayFromImage(vY)

			s[t] = np.sqrt( vX**2 + vY**2 )

			v[t] = self.getDisplacementMap(vX, vY, step, scale, mask)

			# for x in range(0, self.frameSize[0], dx):
			# 	for y in range(0, self.frameSize[1], dy):

			# 		if mask[y,x] > 0:

			# 			rr,cc,val = circle_perimeter_aa( y, x, 2 )
			# 			v[t,rr,cc] = 255 - (val * 58)
						
			# 			rr,cc,val = line_aa( y, x, int(y + vY[y,x] * scale), int(x + vX[y,x] * scale) )
			# 			v[t,rr,cc] = 255- (val * 255)

		self.saveImage( sitk.GetImageFromArray(s), "speedfield" )
		self.saveImage( sitk.GetImageFromArray(v), "velocityfield" )


	def getGrowthField(self):
		print("\n########## Calculating Growth Field ##########\n")

		print("Progress:")
		# V2 = sitk.Image( self.frameSize[0], self.frameSize[1], self.nFrames, sitk.sitkFloat64 )
		G = sitk.Image( self.frameSize[0], self.frameSize[1], self.nFrames, sitk.sitkFloat32 )

		# displacementFilter = sitk.TransformToDisplacementFieldFilter()

		for t in range(1, self.nFrames-1):
			print("{0}/{1}".format(t, self.nFrames))

			vX, vY = self.getVelocity(t)

			dvX = sitk.Gradient( vX )
			dvY = sitk.Gradient( vY )

			divV = sitk.JoinSeries( sitk.VectorIndexSelectionCast( dvX, 0 ) + sitk.VectorIndexSelectionCast(dvY, 1) )
			
			G = sitk.Paste( G, divV, divV.GetSize(), destinationIndex = [0, 0, t])

		return G

	def plotGrowthField(self):
		G = self.getGrowthField()

		M = sitk.GetImageFromArray(self.data["M"])

		G = sitk.Mask(G, M)

		self.saveImage( G, "G" )

		G = sitk.Mask( G, G > 0 )
		G = sitk.Mask( G, G < 0.255)
		iG = sitk.Cast(G*1000, sitk.sitkUInt8)


		preview = sitk.Compose(M*50, sitk.GetImageFromArray( self.data["I"][:,:,:,0] ), iG )
		sitk.Show(preview)

		self.saveImage(preview, "GPreview")

	def reconstructSignal(self, outline = True):
		print("\n########## Reconstructing Signal ##########\n")

		reconstructed = np.zeros( (self.nFrames, 2, self.frameSize[0], self.frameSize[1]), dtype = "uint8" )

		for t, nuclearSlice in enumerate(self.data['N']):
			reconstructedSlice = np.zeros( (2, self.frameSize[0], self.frameSize[1]) )
			for centroid, radius, signal in zip(nuclearSlice['centroid'], nuclearSlice['radius'], nuclearSlice['signal']):
				
				nuclearRow = int(centroid[1])
				nuclearColumn = int(centroid[0])

				rr, cc = circle( nuclearRow, nuclearColumn, radius )
				reconstructedSlice[0, rr,cc ] = int(signal)
				
				if outline:
					rr, cc, intensity = circle_perimeter_aa( nuclearRow, nuclearColumn, int(radius)+4 )
					reconstructedSlice[1,rr,cc] = 100*intensity

			reconstructed[t] = reconstructedSlice

		imsave( os.path.join( self.outputDir, "reconstructedSignal.tif"), reconstructed, imagej = True ) 

	def getSignalMap(self):
		print("\n########## Calculating Signal Map ##########\n")

		print("Progress:")

		Y, X = np.mgrid[0:self.frameSize[1], 0:self.frameSize[0]]
		S = sitk.Image( self.frameSize[0], self.frameSize[1], self.nFrames, sitk.sitkFloat32 )
		
		for t in range(self.nFrames):
			print("{0}/{1}".format(t+1, self.nFrames))

			nX = np.array( [ float(nx[0]) for nx in self.data["N"][t]['centroid'] ] )
			nY = np.array( [ float(nx[1]) for nx in self.data["N"][t]['centroid'] ] )
			signal = np.array( [ float(s) for s in self.data["N"][t]['signal'] ] )
			
			if len(signal) > 400:
				idx = np.random.choice(len(signal), 400, replace = False)
				nX = nX[idx]
				nY = nY[idx]
				signal = signal[idx]

			nXY = np.vstack( [nX,nY] ).T
			
			rbf = interpolate.Rbf(nX, nY, signal, epsilon=2)
			nS = rbf(X, Y)
			s = sitk.JoinSeries( sitk.Cast( sitk.GetImageFromArray(nS), sitk.sitkFloat32 ) )
			S = sitk.Paste(S, s, s.GetSize(), destinationIndex = [0, 0, t])

			del rbf

		M = sitk.GetImageFromArray( self.data["M"] )
		S = sitk.Mask(S, M)

		self.saveImage(S, "SRBF")

		iS = sitk.Cast(S, sitk.sitkUInt8)

		preview = sitk.Compose( M*50, sitk.GetImageFromArray( self.data["I"][:,:,:,0] ), iS )

		sitk.Show(preview)
		self.saveImage(preview, "SRBFPreview")

	def getNuclearDensity(self):
		print("\n########## Calculating Nuclear Density ##########\n")
		print("Progress:")

		D = sitk.Image( self.frameSize[0], self.frameSize[1], self.nFrames, sitk.sitkFloat32 )

		for t in range(self.nFrames):
			print("{0}/{1}".format(t+1, self.nFrames))

			nX = np.array( [ int(nx[0]) for nx in self.data["N"][t]['centroid'] ] )
			nY = np.array( [ int(nx[1]) for nx in self.data["N"][t]['centroid'] ] )

			nXY = np.vstack( [nX, nY] )

			kernel = stats.gaussian_kde(nXY)

			Y,X = np.mgrid[0:self.frameSize[0], 0:self.frameSize[1]]
			
			positions = np.vstack([X.ravel(), Y.ravel()])
			
			nD = np.reshape(kernel(positions).T, X.shape)

			d = sitk.JoinSeries( sitk.Cast( sitk.GetImageFromArray(nD), sitk.sitkFloat32 ) )

			D = sitk.Paste(D, d, d.GetSize(), destinationIndex = [0, 0, t])

		# self.saveImage(D, "nuclearDensity")
		sitk.Show(D)


	def getAnnotation(self, image, targets, d, values=[0.1,0.25,0.5]):
		annotations = np.zeros_like(image)
		for target in targets:
			for value in values:
				rr, cc, intensity = circle_perimeter_aa( int(target[0][1]), int(target[0][0]), int(value*d), shape = annotations.shape)
				annotations[rr,cc] = intensity*255

		return annotations

	def plotRelativeDistance(self):
		# Requesting relative origins (notches)
		self.getTargets(self.data["I"][-1,:,:,0], 'point', radius = 0, n = 1)

		annotations = np.zeros_like( self.data["I"][:,:,:,0] )

		d = np.linalg.norm( self.targets[0][0] - self.targets[1][0] )
		annotations[-1] = self.getAnnotation(self.data["I"][-1,:,:,0], self.targets, d)

		if self.targets:
			for t in range(self.nFrames-2, -1, -1):
				transform = self.compileTransform(t)
				dTargets = [ [ np.array(transform.TransformPoint( target[0] )) - target[0] ] for target in self.targets ]
				self.targets = [ [ target[0] + dTarget[0] ] for target, dTarget in zip(self.targets, dTargets) ]

				d = np.linalg.norm( self.targets[0][0] - self.targets[1][0] )
				annotations[t] = self.getAnnotation(self.data["I"][t,:,:,0], self.targets, d)

		annotantedImage = np.stack( (self.data["I"][:,:,:,0], annotations), axis = 1)
		print (annotantedImage.shape)
		imsave(os.path.join(self.outputDir, "annotations.tiff"), annotantedImage, plugin = "tifffile", imagej = True)


	def getRelativeStats(self, doVelocity = True, doSignal = True):
		print("\n########## Calculating Relative Stats ##########\n")

		# Requesting relative origins (notches)
		self.getTargets(self.data["I"][-1,:,:,0], 'point', radius = 0, n = 1)

		if self.targets:

			signalData = pd.DataFrame(columns = ['r', 'theta', 't', 's', 'd', 'label'])
			velocityData = pd.DataFrame(columns = ['r', 'theta', 't', 'vr','vt','divV', 'divVT', 'd', 'label'])
			
			for t in range(self.nFrames-2, 0, -1):
				
				print("{0}/{1}".format(t+1, self.nFrames))

				# Finding origin displacement and moving it
				transform = self.compileTransform(t)
				dTargets = [ [ np.array(transform.TransformPoint( target[0] )) - target[0] ] for target in self.targets ]
				self.targets = [ [ target[0] + dTarget[0] ] for target, dTarget in zip(self.targets, dTargets) ]

				if doSignal:
					print("\tCalculating Relative Signal...\n")
					signalData = signalData.append( self.getRelativeSignalStats(t, dTargets), ignore_index = True )
				
				if doVelocity:
					print("\tCalculating Relative Velocity...\n")
					velocityData = velocityData.append( self.getRelativeVelocityStats(t, dTargets), ignore_index = True )

			if doSignal:
				pd.to_pickle(signalData, os.path.join(self.outputDir, "signal.stat") )

			if doVelocity:
				pd.to_pickle(velocityData, os.path.join(self.outputDir, "velocity.stat") )

	def getRelativeCoordinates(self, dTargets, YX):

		assert YX.shape[1] == 2
		assert len(YX.shape) == 2
		
		pY = YX[:,0]
		pX = YX[:,1]

		pR = np.zeros( (len(self.targets), len(YX) ) )
		pT = np.zeros_like(pR)

		for i,target in enumerate(self.targets):
			dT = np.arctan2( -dTargets[i][0][1], -dTargets[i][0][0] )

			nXrel = pX - target[0][0]
			nYrel = pY - target[0][1]

			pR[i] = np.square(nXrel) + np.square(nYrel)
			pT[i] = np.mod( np.arctan2( nYrel, nXrel ) - dT + np.pi, 2*np.pi) - np.pi

		rMap = np.argmin(pR, axis = 0)

		pR = np.sqrt( np.choose( rMap, pR ) )
		pT = np.choose( rMap, pT )

		return np.stack( (pR, pT) ).T, rMap

	def getRelativeSignalStats(self, t, dTargets):
		nSlice = self.data['N'][t]

		# Getting relative coordinates and signal values
		YX = np.array( [ (centroid[1], centroid[0]) for centroid in nSlice["centroid"] ] ).astype('float32')
		RT,_ = self.getRelativeCoordinates(dTargets, YX)
		nS = np.array(nSlice["signal"])

		# Compiling data

		newData = {}
		newData['t']		= t * self.data["metadata"]["dt"]
		newData['r']		= RT[:,0] * self.data["metadata"]["res"]
		newData['theta']	= RT[:,1]
		newData['s']		= nS
		newData['d']		= np.linalg.norm( self.targets[0][0] - self.targets[1][0] ) * self.data["metadata"]["res"]
		newData['label']	= self.name

		return pd.DataFrame(newData)

	def getRelativeVelocityStats(self, t, dTargets):
		s0 = self.frameSize[0]
		s1 = self.frameSize[1]

		# Creating a list of coordinates of all frame pixels
		YX = np.indices( (s0, s1) ).reshape( (2,s0*s1) ).T
		
		RT, rMap = self.getRelativeCoordinates( dTargets, YX )

		# Converting points RT back to s0 x s1 image
		RT = np.stack( (RT[:,0].reshape(s0,s1), RT[:,1].reshape(s0,s1))  )
		rMap = rMap.reshape( s0,s1 )
		# Finding divergence
		vX, vY = self.getVelocity(t)
		
		dvX = sitk.Gradient( vX )
		dvY = sitk.Gradient( vY )
		divV = sitk.GetArrayFromImage( sitk.VectorIndexSelectionCast( dvX, 0 ) + sitk.VectorIndexSelectionCast(dvY, 1) )

		# Finding direction of principal strain
		L = np.zeros( (self.frameSize[0], self.frameSize[1], 2, 2) )
		
		L[:,:,0,0] = sitk.GetArrayFromImage( sitk.VectorIndexSelectionCast( dvX, 0 ) )
		L[:,:,0,1] = sitk.GetArrayFromImage( sitk.VectorIndexSelectionCast( dvX, 1 ) )
		L[:,:,1,0] = sitk.GetArrayFromImage( sitk.VectorIndexSelectionCast( dvY, 0 ) )
		L[:,:,1,1] = sitk.GetArrayFromImage( sitk.VectorIndexSelectionCast( dvY, 1 ) )

		D = 0.5 * ( L + L.transpose(0,1,3,2) )

		W,V = np.linalg.eig(D)

		maxEigValIdx = np.argmax(W, axis = 2)
		maxEigVectors = np.choose( maxEigValIdx, V.transpose(3,2,0,1) )
		
		gT = np.abs ( np.arctan2( maxEigVectors[1,:,:], maxEigVectors[0,:,:] ) )

		# Getting polar velocities
		vX = sitk.GetArrayFromImage(vX)
		vY = sitk.GetArrayFromImage(vY)

		vR, vT = toPolar(vX, vY)

		# Rotating velocity and direction of principal strain according to closest origin
		for i in range(len(self.targets)):
			dT = np.arctan2( -dTargets[i][0][1], -dTargets[i][0][0] )
			vT[rMap==i] = np.mod( vT[rMap==i] - dT + np.pi, 2*np.pi ) - np.pi
			gT[rMap==i] = np.mod( gT[rMap==i] - dT + np.pi, 2*np.pi ) - np.pi

		# Compiling data
		mask = self.data['M'][t]

		newData = {}
		newData['t']		= t * self.data["metadata"]["dt"]
		newData['r']		= RT[0, mask > 0 ] * self.data["metadata"]["res"]
		newData['theta']	= RT[1, mask > 0 ]

		newData['vr']		= vR[ mask > 0 ]
		newData['vt']		= vT[ mask > 0 ]
		
		newData['divV']		= divV[ mask > 0 ]
		newData['divVT']	= np.abs( np.cos(gT[ mask >0 ]) )
		
		newData['d']		= np.linalg.norm( self.targets[0][0] - self.targets[1][0] ) * self.data["metadata"]["res"]
		newData['label']	= self.name

		return pd.DataFrame(newData)

	def segment(self):
		W = sitk.GetImageFromArray( self.data['I'][0,:,:,0] )

		L = sitk.Image( W.GetWidth(), W.GetHeight(), sitk.sitkUInt16 )

		for i in range( len(self.data["N"][0]['centroid']) ):
			x = int(self.data["N"][0]['centroid'][i][0])
			y = int(self.data["N"][0]['centroid'][i][1])
			l = int(self.data["N"][0]['labels'][i])

			L[x,y] = l

		# Gradient Filter
		gFilter = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
		gFilter.SetSigma(0.5)
		walls = gFilter.Execute(W)

		sitk.Show( sitk.LabelOverlay(sitk.Cast(walls, sitk.sitkUInt8), L) )

		labels = sitk.MorphologicalWatershedFromMarkers(walls, L, markWatershedLine=False, fullyConnected=True)
		
		# # WaterShed Filter
		wFilter = sitk.MorphologicalWatershedImageFilter()
		wFilter.SetLevel(20)
		# wFilter.SetFullyConnected(False)
		# wFilter.SetMarkWatershedLine(False)
		labels = wFilter.Execute(walls)
		
		sitk.Show(sitk.LabelOverlay(W, labels))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='TimelapseAnalyser')

	parser.add_argument('inputFile', type=str, help = 'Output of the TimelapseProcessor.py *.prc file.')
	parser.add_argument('outputDir', type=str, help = 'Analysis output directory')

	parser.add_argument('-j', action = 'store_true', default = False, help = 'Output expansion (jacobian) map')
	parser.add_argument('-g', action = 'store_true', default = False, help = 'Output expansion rate (growth) map')
	parser.add_argument('-v', action = 'store_true', default = False, help = 'Output velocity field')
	parser.add_argument('-t', action = 'store_true', default = False, help = 'Run microsecror tracking')
	parser.add_argument('-s', action = 'store_true', default = False, help = 'Run relative stats')
	parser.add_argument('-p', action = 'store_true', default = False, help = 'Plot relative distance')
	parser.add_argument('-r', action = 'store_true', default = False, help = 'Reconstruct signal')
	parser.add_argument('-tm', action = 'store_true', default = False, help = 'Output transform map')

	parser.add_argument('--reverse', action = 'store_true', default = False, help = 'Run tracking in reverse. Affects expansion calculation, and tracking.')
	parser.add_argument('--trajectory', action = 'store_true', default = False, help = 'Run trajectory tracking')
	parser.add_argument('--trackElement', choices = ['poly', 'point', 'labels'], default='poly', help = 'Type of tracking elements.')
	parser.add_argument('--labelFile', type=str, default = '', help = 'Label file. Required if trackElement set to "labels" ')
	parser.add_argument('-n', type=int, default = 20, help ='Number of points per element. Default n = 20')

	parser.add_argument('--relativeVelocity', action = 'store_true', default = False, help ='Compute relative velocity statistics. Requires definition of origins.')
	parser.add_argument('--relativeSignal', action = 'store_true', default = False, help ='Compute relative signal. Requires definition of origins.')
	
	args = parser.parse_args()

	analyser = SectorAnalyser(args.inputFile, args.outputDir)

	done = False

	if args.r:
		analyser.reconstructSignal()

	if args.j:
		if args.reverse:
			analyser.getJacobianMapFwd()
		else:
			analyser.getJacobianMap()

	if args.g:
		analyser.plotGrowthField()

	if args.v:
		analyser.plotVelocityField(60, 170)

	if args.tm:
		analyser.getTenaformViz()

	if args.t:
		if (args.trackElement == 'labels' and args.labelFile == ''):
			parser.error("labelFile should be specified for label tracking.")

		if args.trajectory:
			analyser.plotTrajectory()
		else:
			if args.reverse:
				analyser.trackPointFieldRev( args.trackElement, n = args.n, L = args.labelFile )
			else:
				analyser.trackPointField( args.trackElement, n = args.n, L = args.labelFile )
		done = True

	if args.s:
		if done:
			parser.error("Can't do tracking and relative stats at the same time")
		
		analyser.getRelativeStats(args.relativeVelocity, args.relativeSignal)

	if args.p:
		if done:
			parser.error("Can't do tracking and relative stats at the same time")
		analyser.plotRelativeDistance()
