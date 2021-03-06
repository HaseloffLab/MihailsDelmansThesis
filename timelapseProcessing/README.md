# Image processing scripts
Mihails Delmans

The image-processign scripts used in the PhD thesis on 'Engineering morphogenesis of Marchantia polymorpha gemmae'.

See details of the pipeline in Chapter 2.

## TimelapseProcessor

### Description

Process the time-lapse image by calculating transformations using the membrane channel; masking the gemma using chlorophyl channel; extracting the signal from the signal channel at the locations specified by the reference channel. Outputs a *.prc file, which can be used as an input to TimelapseAnalyser, as well as intermediate images for verification of the pipeline.

### Dependencies

`SimpleITK`

`matplotlib`

`numpy`

`scikit-image`

### Usage

``` bash
python3 timelapseProcessor.py inputFile outputDir [options]
```

__inputFile__  multichannel *.tif file

__outputDir__  the directory for the output files

__Options:__

__-wC__      index of the wall cahnnel for registration. Default: 0

__-aC__      index of the autofluorescence cahannel for masking. Default: 1

__-sC__      index of the nuclear signal channel. Default: 2

__-nC__      index of the nuclear reference channel. Default: 3

__-dt__      time interval in minutes. Default: 1

__-dx__      spatial resolution of the stack in um/px. Default: 1

__-m__       recalculate masks

__-r__       recalculate registration

__-n__       recalculate nuclei

__-nomask__  do not calculate mask

__-noreg__   do not calculate registration

__-nonucl__  do not calculate nuclei
	
---
## TimelapseAnalyser

### Description

Performs analysis of the processed time-lapse (*.prc file).

Can output expansion, growth rate or velocity fields; track microsectors; calculate signal / velocity relative to the notches;

See usage for details.

### Dependencies

`SimpleITK`

`matplotlib`

`numpy`

`scipy`

`pandas`

`scikit-image`

### Usage

``` bash
python3 timelapseAnalyser.py inputFile outputDit [options]
```

__inputFile__ processed pack (*.prc file), produced by the TimelapseProcessor

__outputDir__ the directory for the output files

__Options:__

__-j__                    output expansion (jacobian) map

__-g__                    output expansion rate (growth) map

__-v__                    output velocity field

__-t__                    run microsecror tracking

__-s__                    calculate statistics relative to the notches (outputs to *.stats file)

__-r__                    run signal reconstruction

__-tm__                   output transform map

__Tracking options:__

__--reverse__             run tracking in reverse. Affects expansion calculation, and tracking.

__--trajectory__          run trajectory tracking

__--trackElement {poly,point,labels}__ type of tracking elements

__--labelFile__ label file. Required if trackElement set to "labels"

__-n__                    Number of points per element. Default n = 20

__Statistics options:__

__--relativeVelocity__    compute relative velocity statistics

__--relativeSignal__      compute relative signal

---
## TimelapseAggregator

### Description
Average statistics from several time-lapses, stored in the *.stats files produced by the TimelapseAnalyser.

### Dependencies

`pandas`

`numpy`

`scipy`

`matplotlib`

`scikit-image`

### Usage

``` bash
python3 timelapseAggregator.py stat column plotType [options]
```

__stat__ 					- type of statistics to process ( velocity / signal )

__column__ 					- column in the stat file to process

__plotType {xy, along}__ 	- type of plot. xy - plot the stat specified by the 'column' relative to the x and y coordinates.

__along__ - plot the stat specified by the 'column' against the stat specified by the '--along' option

__Options:__

__-i__          		location of "velocity.stat" files (usually output directory of the TimelapseAnalyser)

__-o__                output directory

__--interpolation {nearest,linear,cubic}__

__--res__             	resolution of the plot in pixels. Default: 1000

__--n__                 sample size for each input file. Default: 1000000

__--norm__              normalise coordinates by the value specified in the 'd' column (usually the distance between the notches calculated by the TimelapseAnalyser)

__Plot along options:__

__--along__         	column along which to plot, e.g., 'x'

__--fixing__            optionally fix another column, e.g, 'y'

__--at__            	at the mean value

__--window__      		with a window around mean of the fixing column

__--normout__           normalise the output to range [0,1]

__--cutat__       		cut 'along' column at this value
