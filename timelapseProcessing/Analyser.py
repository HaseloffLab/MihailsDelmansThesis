import pickle
import os
import re

import SimpleITK as sitk

class Analyser:
	def __init__(self, dataFileName, outputDir):
		print("\n########## Reading Data ##########\n")

		try:
			dataFile = open( dataFileName, 'rb' )
			self.data = pickle.load(dataFile, encoding='latin1')
		except:
			print("Failed to read {0}".format(dataFileName))
			raise

		print("Data contains the following keys:\n")

		for key in self.data:
			print("\t{0}[{1}]".format(key, len(self.data[key])))

		if "metadata" in self.data:
			print ("\nMetdata:")
			for key in self.data["metadata"]:
				print("\t{0}: {1}".format(key, self.data["metadata"][key]))

		self.targets = []
		self.frameSize = (self.data['I'].shape[1], self.data['I'].shape[2])
		self.nFrames = len(self.data['T']['fwd']) + 1

		self.name = os.path.splitext(os.path.basename(dataFileName))[0]
		self.outputDir = os.path.join(outputDir, self.name)

		if not os.path.exists(self.outputDir):
			print("Creating output directory {0}".format(self.outputDir))
			os.makedirs( self.outputDir )

	def saveImage(self, image, title = "temp"):
		imagePath = os.path.join( self.outputDir, "{0}_{1}.tif".format(self.name, title) )

		print("Saving image to {0}".format(imagePath))

		sitk.WriteImage( image, os.path.join( self.outputDir, "{0}_{1}.tif".format(self.name, title) ) )

	def nextFileName(self, prefix, ext = 'tif'):
		matches = [ re.search( '{0}_{1}([0-9]*)\.{2}'.format(self.name, prefix, ext), fileName ) for fileName in os.listdir( self.outputDir ) ]
		matches = [ m for m in matches if m ]
		if len(matches) > 0:
			idx =  max([ int(m.group(1)) for m in matches]) + 1
		else:
			idx = 0

		return "{0}{1}".format(prefix, idx)

	def TVector(self, i, key, inverse = False):
		if inverse:
			return [ float(x) for x in self.data['T']['inv'][i][key] ]
		else:
			return [ float(x) for x in self.data['T']['fwd'][i][key] ]

	def compileTransform(self, i, inverse = False):
		transform = sitk.BSplineTransform(2, 3)
		transform.SetTransformDomainDirection( self.TVector(i, 'Direction', inverse) )
		transform.SetTransformDomainOrigin( self.TVector(i, 'Origin', inverse) )
		transform.SetTransformDomainPhysicalDimensions( self.TVector(i, 'Size', inverse) )
		transform.SetTransformDomainMeshSize( [int(x) - 3 for x in self.data['T'][ 'inv' if inverse else 'fwd' ][i]['GridSize']] )
		transform.SetParameters( self.TVector(i, 'TransformParameters', inverse) )

		return transform

	def getJacobian(self, t):
		nX = np.indices( (self.frameSize[0], self.frameSize[1]) )
			
		X = sitk.Cast( sitk.GetImageFromArray(nX[1]), sitk.sitkFloat64 )
		Y = sitk.Cast( sitk.GetImageFromArray(nX[0]), sitk.sitkFloat64 )

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

		return jacobian

	def getGrowth(self, t):
			
			displacementFilter = sitk.TransformToDisplacementFieldFilter()
			displacementFilter.SetReferenceImage( sitk.GetImageFromArray( self.data["I"][t,:,:,0] ) )
			
			zeroU = sitk.Image( self.frameSize[0], self.frameSize[1], sitk.sitkVectorFloat64 )
			k = 0.5
			
			if t > 0:
				forwardTransform = self.compileTransform(t-1)
				forwardU = -displacementFilter.Execute(forwardTransform)
			else:
				forwardU = zeroU
				k = 1

			if t < self.nFrames-1:	
				inverseTransform = self.compileTransform(t, inverse = True)
				backwardU = displacementFilter.Execute(inverseTransform)
			else:
				backwardU = zeroU
				k = 1

			v2 = backwardU + forwardU

			vX = sitk.VectorIndexSelectionCast(v2, 0) * k
			vY = sitk.VectorIndexSelectionCast(v2, 1) * k

			dvX = sitk.Gradient( vX )
			dvY = sitk.Gradient( vY )

			divV = sitk.VectorIndexSelectionCast( dvX, 0 ) + sitk.VectorIndexSelectionCast(dvY, 1)

			return divV







