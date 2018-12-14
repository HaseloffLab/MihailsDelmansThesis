import sys
import os
import argparse

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pickle
from math import floor
from skimage.io import imread

class TimelapseProcessor:
	def __init__(self, inputFile, outputDir, metadata):		
		self.inputFile = inputFile
		self.metadata = metadata

		try:
			i = imread( args.inputFile )
			self.I = sitk.GetImageFromArray(i, isVector = True)
		except:
			raise Exception( "Failed to open {0}".format(self.inputFile) )

		print("\n########## Reading Image ##########\n")

		if self.I.GetDepth() == 0:
			self.I = sitk.JoinSeries(self.I)

		print("Read image {0}x{1}x{2} {3}[{4}]".format( self.I.GetHeight(), self.I.GetWidth(), self.I.GetDepth(), self.I.GetPixelIDTypeAsString(), self.I.GetNumberOfComponentsPerPixel() ))
		
		self.N = None
		self.M = None
		self.T = None

		self.name = os.path.splitext(os.path.basename(inputFile))[0]
		self.outputDir = os.path.join(outputDir, self.name)

		if not os.path.exists(self.outputDir):
			print("Creating output directory {0}".format(self.outputDir))
			os.makedirs( self.outputDir )

		self.outputFilePath = os.path.join( self.outputDir, self.name ) + ".prc"
		
		if os.path.exists( self.outputFilePath ):
			print("Found existing data file.")
			outputFile = open(self.outputFilePath, 'rb')
			data = pickle.load(outputFile, encoding='latin1')

			if 'N' in data:
				self.N = data['N']
			if 'T' in data:
				self.T = data['T']
			if 'M' in data:
				self.M = sitk.GetImageFromArray(data['M'])
		
			outputFile.close()

	def saveImage(self, image, title = "temp"):
		sitk.WriteImage( image, os.path.join( self.outputDir, "{0}_{1}.tif".format(self.name, title) ) )

	def getChannel(self, cid):
		return sitk.VectorIndexSelectionCast(self.I, self.metadata[cid])

	def ExecuteSliceWise(self, imageFilter, inputImage):
		print("Executing ", imageFilter)

		# Getting filter OutputType
		testSlice = imageFilter.Execute( inputImage[:,:,0] )

		outputImage = sitk.Image( inputImage.GetHeight(), inputImage.GetWidth(), inputImage.GetDepth(), testSlice.GetPixelID() )

		for i in range( inputImage.GetDepth() ):
			
			processedSlice = sitk.JoinSeries( imageFilter.Execute( inputImage[:,:,i] ) )

			try:
				outputImage = sitk.Paste(outputImage, processedSlice, processedSlice.GetSize(), destinationIndex=[0,0,i] )
			except:
				print("Failed to Paste {0}th slice".format(i))
				print("Input: ", inputImage.GetSize(), inputImage.GetPixelIDTypeAsString())
				print("Slice: ", processedSlice.GetSize(), processedSlice.GetPixelIDTypeAsString())
				print("Output: ", outputImage.GetSize(), outputImage.GetPixelIDTypeAsString())
				raise
		return outputImage

	def processMask(self):

		print("\n########## Processing Mask ##########\n")

		aC = self.getChannel("aC")

		self.saveImage(aC, "chlorophyl")

		# Gaussian filter
		gFilter = sitk.DiscreteGaussianImageFilter()
		gFilter.SetVariance(0.5)
		aC = self.ExecuteSliceWise(gFilter, aC)
		
		self.saveImage(aC, "chlorophyl-gaussian")
		sitk.Show(aC, "Gaussian" )

		# Background detection
		cFilter = sitk.ConnectedThresholdImageFilter()
		cFilter.AddSeed([0,0])
		labels = self.ExecuteSliceWise(cFilter, aC)

		self.saveImage(labels, "chlorophyl-connected")

		# Closing
		cFilter = sitk.BinaryMorphologicalClosingImageFilter()
		cFilter.SetKernelType(sitk.sitkBall)
		cFilter.SetKernelRadius(10)
		labels = self.ExecuteSliceWise(cFilter, labels)

		self.saveImage(labels, "chlorophyl-closing")

		# Dialate
		dFilter = sitk.BinaryDilateImageFilter()
		dFilter.SetKernelType(sitk.sitkBall)
		dFilter.SetKernelRadius(10)
		labels = self.ExecuteSliceWise(dFilter, labels)

		self.saveImage(labels, "chlorophyl-dialate")

		# Inverting labels
		self.M = (labels - 1) * -1

		preview = sitk.LabelOverlay(self.getChannel("aC"), self.M)

		sitk.Show(preview, "Mask" )
		self.saveImage(preview, "mask")

	def processRegistration(self):

		print("\n########## Processing Registration ##########\n")

		wC = self.getChannel("wC")

		elastixImageFilter = sitk.ElastixImageFilter()
		elastixImageFilter.LogToConsoleOff()

		parameterMap = sitk.GetDefaultParameterMap("bspline")
		
		parameterMap['MaximumNumberOfIterations'] = ['512']
		
		elastixImageFilter.SetParameterMap(parameterMap)

		print("Running registration with the following parameters: ")
		print(elastixImageFilter.PrintParameterMap())

		self.T = {'fwd': [], 'inv': []}

		outputStack = sitk.Image( [self.I.GetHeight(), self.I.GetWidth(), self.I.GetDepth() - 1], sitk.sitkVectorUInt8, 3 )

		print("Progress:")
		for i in range( self.I.GetDepth() - 1):
			
			# Finding forward transfrom
			elastixImageFilter.SetFixedImage( wC[:,:,i+1] )
			elastixImageFilter.SetMovingImage( wC[:,:,i] )
			if self.M:
				elastixImageFilter.SetFixedMask( self.M[:,:,i+1] )
				elastixImageFilter.SetMovingMask( self.M[:,:,i] )
			
			elastixImageFilter.Execute()

			registredImage = sitk.Cast( elastixImageFilter.GetResultImage(), wC.GetPixelID() )
			
			composer = sitk.ComposeImageFilter()
			composedImage = sitk.JoinSeries( composer.Execute( wC[:,:,i], wC[:,:,i+1], registredImage ) )

			outputStack = sitk.Paste( outputStack, composedImage, composedImage.GetSize(), destinationIndex=[0,0,i] )

			self.T['fwd'].append( dict(elastixImageFilter.GetTransformParameterMap()[0]) )

			# Finding backward transform
			elastixImageFilter.SetFixedImage( wC[:,:,i] )
			elastixImageFilter.SetMovingImage( wC[:,:,i+1] )
			if self.M:
				elastixImageFilter.SetFixedMask( self.M[:,:,i] )
				elastixImageFilter.SetMovingMask( self.M[:,:,i+1] )
			
			elastixImageFilter.Execute()
			self.T['inv'].append( dict(elastixImageFilter.GetTransformParameterMap()[0]) )
			
			print("{0}/{1}".format(i+1, self.I.GetDepth() - 1))

		sys.stdout.write('\n')

		sitk.Show( outputStack, "registration")
		self.saveImage(outputStack, "registration")

	def processNuclei(self):

		print("\n########## Processing Nuclei ##########\n")

		nC = self.getChannel("nC")
		sC = self.getChannel("sC")

		self.saveImage(nC, "nChannel")

		# Gaussian filter
		gFilter = sitk.DiscreteGaussianImageFilter()
		gFilter.SetVariance(0.5)
		nC = self.ExecuteSliceWise(gFilter, nC)

		self.saveImage(nC, "nChannel-gaussian")

		# Otsu Filter
		oFilter = sitk.OtsuThresholdImageFilter()
		oFilter.SetInsideValue(0)
		oFilter.SetOutsideValue(255)
		binary = self.ExecuteSliceWise(oFilter, nC)

		self.saveImage(binary, "nChannel-otsu")

		cFilter = sitk.ConnectedComponentImageFilter()

		oLabels = self.ExecuteSliceWise(cFilter, binary)

		print(oLabels.GetPixelIDTypeAsString())

		self.saveImage(sitk.Cast(oLabels, sitk.sitkUInt16), "nChannel-oLabels")

		# Distance Filter
		dFilter = sitk.SignedMaurerDistanceMapImageFilter()
		dFilter.SetInsideIsPositive(False)
		dFilter.SetSquaredDistance(False)
		dFilter.SetUseImageSpacing(False)

		d = self.ExecuteSliceWise(dFilter, oLabels)

		self.saveImage(d, "nChannel-d")

		## Adding information from the nuclei channel
		d = d + 0.05 * sitk.Cast( sitk.InvertIntensity(nC), d.GetPixelID() )

		self.saveImage(d, "nChannel-d-plus")

		# Watershed Filter
		wFilter = sitk.MorphologicalWatershedImageFilter()
		wFilter.SetLevel(1)
		wFilter.SetMarkWatershedLine(False)

		dLabels = self.ExecuteSliceWise(wFilter, d)

		self.saveImage(sitk.Cast(dLabels, sitk.sitkUInt16), "nChannel-dLabels")

		dLabels = sitk.Mask( dLabels, sitk.Cast( oLabels, dLabels.GetPixelID() ) )

		self.saveImage(sitk.Cast(dLabels, sitk.sitkUInt16), "nChannel-dLabels-mask")

		# Label shape and signal
		shapeFilter = sitk.LabelShapeStatisticsImageFilter()
		
		signalStatisticsFilter = sitk.LabelStatisticsImageFilter()
		referenceStaristicFilter = sitk.LabelStatisticsImageFilter()

		self.N = []


		for t in range(dLabels.GetDepth()):
			dLabelsSlice = dLabels[:,:,t]
			signalSlice = sC[:,:,t]
			referenceSlice = nC[:,:,t]

			shapeFilter.Execute(dLabelsSlice)
			
			signalStatisticsFilter.Execute( signalSlice, dLabelsSlice )
			referenceStaristicFilter.Execute( referenceSlice, dLabelsSlice )

			nucleiSlice = {}
			nucleiSlice['centroid'] = []
			nucleiSlice['radius'] = []
			nucleiSlice['label'] = []
			nucleiSlice['signal'] = []
			nucleiSlice['reference'] = []
			labels = shapeFilter.GetLabels()

			for label in labels:
				radius = shapeFilter.GetEquivalentSphericalRadius(label)
				if radius > 1:
					nucleiSlice['label'].append(label)
					nucleiSlice['centroid'].append( shapeFilter.GetCentroid(label) )
					nucleiSlice['radius'].append( radius )
					nucleiSlice['signal'].append( signalStatisticsFilter.GetMean(label) )
					nucleiSlice['reference'].append( referenceStaristicFilter.GetMean(label) )

			self.N.append(nucleiSlice)

		preview = sitk.LabelOverlay(nC, dLabels)

		sitk.Show( preview, "Nuclei", debugOn=True )
		self.saveImage(preview, "nuclei")

	def export(self):
		print("\n########## Exporting ##########\n")

		data = {}
		
		data["I"] = sitk.GetArrayViewFromImage(self.I)
		data["metadata"] = self.metadata

		if self.M:
			data["M"] = sitk.GetArrayViewFromImage(self.M)
		if self.N:
			data["N"] = self.N
		if self.T:
			data["T"] = self.T

		outputFile = open(self.outputFilePath, "wb")
		pickle.dump(data, outputFile)


		print("Data saved to {0}".format(self.outputDir))

if __name__ == "__main__":
	metadata = {"dt"	: 72,
				"res" 	: 10,
				"wC"	: 0,
				"aC"	: 1,
				"sC"	: 2,
				"nC"	: 3
				}

	parser = argparse.ArgumentParser(description='TimelapseProcessor')
	
	parser.add_argument('inputFile', type=str, help = '*.tif input image stack')
	parser.add_argument('outputDir', type=str, help = 'output directory')
	
	parser.add_argument('-wC', type=int, default = 0, help = 'index of wall cahnnel for registration. Default: 0')
	parser.add_argument('-aC', type=int, default = 1, help = 'index of autofluorescence cahannel for masking. Default: 1')
	parser.add_argument('-sC', type=int, default = 2, help = 'index of a nuclear signal channel. Default: 2')
	parser.add_argument('-nC', type=int, default = 3, help = 'index of nuclei channel. Default: 3')
	
	parser.add_argument('-dt', metavar = 'dt', type=int, default = 1, help= 'time interval in minutes. Default: 1')
	parser.add_argument('-dx', metavar = 'dx', type=float, default = 1.0, help = 'spatial resolution of the stack in um/px. Default: 1')

	parser.add_argument('-m', action = 'store_true', default = False, help = 'recalculate masks')
	parser.add_argument('-r', action = 'store_true', default = False, help = 'recalculate registration')
	parser.add_argument('-n', action = 'store_true', default = False, help = 'recalculate nuclei')

	parser.add_argument('-nomask', action = 'store_true', default = False, help = 'do not calculate mask')
	parser.add_argument('-noreg', action = 'store_true', default = False, help = 'do not calculate registration')
	parser.add_argument('-nonucl', action = 'store_true', default = False, help = 'do not calculate nuclei')


	args = parser.parse_args()

	metadata = {"dt" : args.dt,
				"res" : args.dx,
				"wC" : args.wC,
				"aC" : args.aC,
				"sC" : args.sC,
				"nC" : args.nC
				}

	processor = TimelapseProcessor(args.inputFile, args.outputDir, metadata = metadata)

	if (not args.nomask) and ( (not processor.M) or args.m or args.r ):
		processor.processMask()
	if (not args.noreg) and ( (not processor.T) or args.r):
		processor.processRegistration()
	if (not args.nonucl) and ( (not processor.N) or args.n):	
		processor.processNuclei()
	processor.export()

