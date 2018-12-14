import sys
import os
import argparse

import itertools

import pandas as pd
import numpy as np

from scipy.interpolate import griddata

from scipy.interpolate import spline

from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from mpl_toolkits.mplot3d import axes3d

from skimage.external.tifffile import imsave
from skimage.filters import gaussian

class StatProcessor:
	def __init__(self, inputDir, stat, outputDir, sampleN = 1000000):
		self.analysisDir = inputDir
		self.outputDir = outputDir
		self.data = pd.DataFrame()

		print("\n########## Reading Data ##########\n")

		for analysisDir in self.analysisDir:
			for folder in os.listdir(analysisDir):
				outDir = os.path.join(analysisDir, folder)
				if os.path.isdir( outDir ):
					statPath = os.path.join(outDir, '{0}.stat'.format(stat) )
					try:
						newData = pd.read_pickle(statPath)
						print ("Reading {0}".format( statPath ))
						self.data = self.data.append( newData.sample( min(sampleN, len(newData) ) ) )
						
					except Exception as e:
						print( "Failed to find {0}".format(statPath) )
				

		self.data['x'] = self.data['r'] * np.cos( self.data['theta'] )
		self.data['y'] = self.data['r'] * np.sin( self.data['theta'] )

	def xyPlot( self, column, res = 1000, norm = False, interpolation = None):
		print("\n########## Preparing {0} xyPlot ##########\n".format(column))

		if norm:
			self.data['x'] = self.data['x'] / self.data['d']
			self.data['y'] = self.data['y'] / self.data['d']
			lim = 2.0
		else:
			lim = max( np.abs(self.data['x']).max(), np.abs(self.data['y']).max() ) * 2

		step = lim / (res-1)

		steps = np.array( range(res + 1) ) * step - 0.5*( lim + step )

		self.data['xbin'] = pd.cut( self.data['x'], steps, labels = False )
		self.data['ybin'] = pd.cut( self.data['y'], steps, labels = False )

		timeGroups = self.data.groupby('t')
		output = np.zeros( (0, res,res), dtype = np.float32 )

		for t, tDf in self.data.groupby(['t']):
			tDf = tDf.assign(t=t)
			binned = tDf.groupby( ['ybin', 'xbin'] ).mean().add_suffix('_mean').reset_index()

			if interpolation:
				binCenters = 0.5*(steps[:-1] + steps[1:])
				xx, yy = np.meshgrid(binCenters, binCenters)
				points = np.stack( (binned['y_mean'].astype(np.float32), binned['x_mean'].astype(np.float32)) ).T
				values = binned[column+'_mean'].values
				image = griddata(points, values, (yy,xx), method = interpolation)
			else:
				image = np.zeros((res, res))
				image[ binned['ybin'].astype(np.int_), binned['xbin'].astype(np.int_) ] = binned[column+'_mean']
			
			output = np.append( output, [image], axis = 0)

		fileName = '{0}_{1}.tif'.format(column, 'norm' if norm else 'real')
		path =  os.path.join(self.outputDir, fileName)

		imsave( path, output.astype(np.float32) )
		print('Saved image to {0}'.format(path))
		print('Limits: -{0}:{0}, Resolution: {1} um/px'.format( 0.5*lim, step ))

	def timePlot(self, binned, along, steps, column, normOut):
		tMax = np.max( binned['t'] )

		for t, tDf in binned.groupby(['t']):

			binAlong = '{0}_bin'.format(along)

			tDf[binAlong] = pd.cut( tDf[along] , steps, labels = False)


			tDf = tDf.groupby([binAlong]).agg({ column: ['mean', 'std', 'count'] }).reset_index()


			centers = 0.5 * ( steps[:-1] + steps[1:] )

			tDf[along] = centers[ tDf[binAlong].astype(np.int_) ]

			if normOut:
				maxVal = tDf[column]['mean'].max()
			else:
				maxVal = 1.0

			tDf['mean'] = tDf[column]['mean'] / maxVal
			tDf['pSTD'] = (tDf[column]['mean'] + 1.96*( tDf[column]['std'] / np.sqrt(tDf[column]['count']) ) ) / maxVal
			tDf['nSTD'] = (tDf[column]['mean'] - 1.96*( tDf[column]['std'] / np.sqrt(tDf[column]['count']) ) ) / maxVal

			plt.plot( tDf[along], tDf['mean'], to_hex([ t/tMax, 1-t/tMax, 0 ]) )

		plt.show()

	def plotMeanAgainst(self, column, along, fixing, at, window, res = 1000, norm = False, normOut = False, cutAt = None):
		print("\n########## Preparing {0} along {1} plot ##########\n".format(column, along))

		if norm:
			self.data[along] = self.data[along] / self.data['d']
			low = 0.0
			high = 1.0
		else:
			low = np.min( self.data[along]  )
			high = np.max( self.data[along] )

		if cutAt:
			self.data = self.data[self.data[along] < cutAt]

		lim = high - low
		step = lim / (res-1)

		steps = np.array( range(res + 1) ) * step - 0.5*step + low

		binned = self.data.loc[:, ('t', column, along, fixing) ]

		if fixing:
			binned = binned[ binned[fixing] > at - window ]
			binned = binned[ binned[fixing] < at + window ]

		self.timePlot(binned, along, steps, column, normOut)

		binAlong = '{0}_bin'.format(along)

		binned[binAlong] = pd.cut( binned[along] , steps, labels = False)


		binned = binned.groupby([binAlong]).agg({ column: ['mean', 'std', 'count'] }).reset_index()


		centers = 0.5 * ( steps[:-1] + steps[1:] )

		binned[along] = centers[ binned[binAlong].astype(np.int_) ]

		if normOut:
			maxVal = binned[column]['mean'].max()
		else:
			maxVal = 1

		binned['mean'] = binned[column]['mean'] / maxVal
		binned['pSTD'] = (binned[column]['mean'] + 1.96*(binned[column]['std'] / np.sqrt(binned[column]['count'])) ) / maxVal
		binned['nSTD'] = (binned[column]['mean'] - 1.96*(binned[column]['std'] / np.sqrt(binned[column]['count'])) ) / maxVal

		print ("Saving {0} vs {1} to {2}".format(column, along, self.outputDir))

		outFile = open( os.path.join(self.outputDir, '{0}{1}vs_{2}_fixing_{3}_at_{4}_{5}_mean.txt'.format(column, '_norm' if norm else '', along, fixing, at, window)), 'w' )
		outFile.write( binned[ [along, 'mean'] ].to_csv(index = False) )

		outFile.close()
		outFile = open( os.path.join(self.outputDir, '{0}{1}vs_{2}_fixing_{3}_at_{4}_{5}_pSTD.txt'.format(column, '_norm' if norm else '', along, fixing, at, window)), 'w' )
		outFile.write( binned[ [along, 'pSTD'] ].to_csv(index = False) )

		outFile.close()
		outFile = open( os.path.join(self.outputDir, '{0}{1}vs_{2}_fixing_{3}_at_{4}_{5}_nSTD.txt'.format(column, '_norm' if norm else '', along, fixing, at, window)), 'w' )
		outFile.write( binned[ [along, 'nSTD'] ].to_csv(index = False) )
		plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='TimelapseAggregator')
	
	parser.add_argument('-i', type=str, nargs = '+', help = 'location of "velocity.stat" files')
	parser.add_argument('-o', type=str, help = 'output directory')
	parser.add_argument('stat', type=str, help = 'stat file name')
	parser.add_argument('column', type=str, help = 'column to plot')
	parser.add_argument('plotType', choices = ['xy', 'along'])

	parser.add_argument('--interpolation', type=str, choices = ['nearest', 'linear', 'cubic'], default = None)

	parser.add_argument('--res', type=int, default=1000, help = 'resolution of the plot')
	parser.add_argument('--n', type=int, default=1000000, help = 'sample size for each input file')
	parser.add_argument('--norm', action = 'store_true', default = False, help = 'normalise with d')
	parser.add_argument('--along', type=str, help = 'axis along which to plot. Required if plotType = along')
	parser.add_argument('--fixing', type=str, help = 'fixing column')
	parser.add_argument('--at', type=float, help = 'mean values of fixing column to accept. Required if --fixing is set')
	parser.add_argument('--window', type=float, help = 'window around mean of fixing column to accept. Required if --fixing is set')
	parser.add_argument('--normout', action = 'store_true', default = False, help = 'normalise output from 0 to 1')
	parser.add_argument('--cutat',type=float, default=None, help = 'cut along values at this value')

	args = parser.parse_args()

	if args.plotType == 'along':
		if not args.along:
			parser.error("--along flag is required for 'along' plot type")
		else:
			if args.fixing and (args.at == None or args.window == None) :
				parser.error("--at and --window values are required when using --fixing")


	vc = StatProcessor( args.i, args.stat, args.o, sampleN = args.n)
	
	if args.plotType == 'along':
		vc.plotMeanAgainst( args.column, args.along, args.fixing, args.at, args.window, res = args.res, norm = args.norm, normOut = args.normout, cutAt = args.cutat)
	else:
		vc.xyPlot(args.column, args.res, args.norm, args.interpolation)

