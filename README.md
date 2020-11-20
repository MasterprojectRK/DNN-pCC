# DNN-pCC

This repository attempts predicting chromatin conformation in form of so-called Hi-C matrices from various chromatin factors using a neural network.
The network setup is based on the paper [Dense neural networks for predicting chromatin conformation](https://doi.org/10.1186/s12859-018-2286-z) 
by Farre, Heurteau, Cuvier and Emberly (2018).

## Installation
DNN-pCC is designed to run on Linux operating systems.
For now, only a manual installation is supported.
To install DNN-pCC, just clone the github repository to some local directory 
e.g. using git clone.
It is recommended to use [conda](https://docs.conda.io/en/latest/miniconda.html) and run DNN-pCC in a virtual environment.
```
conda create -n YOUR_ENVIRONMENT_NAME
conda activate YOUR_ENVIRONMENT_NAME
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```
DNN-pCC depends on the following packages, which need to be installed:
```
conda install cooler=0.8.5 pybigwig=0.3.17 scipy=1.5.2 matplotlib=3.3.2 tqdm=4.50.2 pydot=1.4.1 scikit-learn=0.23.2 graphviz=2.42.3
conda install -c anaconda click=7.1.2 tensorflow-gpu=2.2.0
```

## Usage
DNN-pCC consists of two modules, training and prediction.
The training module builds and trains a neural network using an existing Hi-C matrix and a selected number of chromatin factors as inputs.
The prediction module takes the trained network, inputs the chromatin factors
e.g. from a different cell line or chromosome and thus predicts the
interaction matrix for the given cell line or chromosome.

### Training
Usage:
```
python training.py [options]
```
Options:
- --help: print help and exit
- --trainMatrices | -tm: Training Hi-C matrix
  - required, must be in cooler format
  - the bin size (resolution) is determined from the matrix
  - multiple matrices can be specified (must match number of chromatin paths, see below)
- --trainChromatinPaths | -tcp: Path to directory with chromatin factor data
  - required
  - multiple paths can be specified (must match number of trainMatrices, see above)
  - chromatin factors must be in bigwig format
  - filenames in the folder are important, make sure to read the notes below
- --trainChromosomes | -tchroms: The chromosome(s) used for training
  - required
  - format depends on your data, e.g. "chr17" or "17". In most cases, both should work.
  - specify multiple chromosome like so: -tchroms "10 11 17"
  - all chromosomes specified must be available both in all cooler matrices
  and all bigwig files
- --validationMatrices | -vm: Validation Hi-C matrix
  - required, must be in cooler format
  - same as above for trainMatrices, just for validation
- --validationChromatinPaths | -vcp: Path to directory with chromatin factor data
  - required
  - same as above for trainChromationPaths, just for validation
- --validationChromosomes | -vchroms: The chromosome(s) used for validation
  - required
  - same as above for trainChromosomes, just for validation
- --sequenceFile | -sf: Text file containing DNA sequence
  - optional
  - experimental feature, usage not recommended
  - must be in fasta format and contain entries at least for the training- and validation chromosomes
  - can be downloaded e.g. from [University of California](http://hgdownload.cse.ucsc.edu/goldenpath/hg19/chromosomes/)
- --outputPath | -o: Output path for logfiles etc.
  - required 
  - data for tensorboard will be stored here
  - intermediate trained models will be stored here
  - debug output and TFRecords will also be stored here
  - sufficient disk space is required for the TFRecord files (several 100s of MiB to GiB, depending on number of matrices and chromosomes used)
- --modelfilepath | -mfp: Filename for trained model
  - required
  - file extension should be ".h5"
  - default: "trainedModel.h5"
- --learningRate | -lr: Learning rate for the optimizer
  - required
  - numerical value > 1e-10, default 1e-5
- --numberEpochs | -ep: Number of epochs for training the network
  - required, numerical value > 20
  - default: 1000
- --batchsize | -bs: Batch size for network
  - required, numerical value > 5
  - default: 256
- --recordsize | -rs: Recordsize for TFRecord files
  - optional
  - integer > 100
  - default: 2000, use smaller values when experimenting with -sf option
- --windowsize | -ws: Chromatin window to consider for training
  - required, numerical value > 10
  - default: 80
- --scaleMatrix | -scm: Scale matrices
  - required
  - default: False
  - scale matrices (separately) to value range [0..1] (min-max scaling)
  - currently not recommended
- --clampFactors | -cfac: Clamp chromatin factors 
  - optional
  - clamp chromatin factors (separately) to value range lowerQuartile-1.5xInterquartile...upperQuartile+1.5xInterquartile
  - experimental, currently not recommended
- --scaleFactors | -scf: Scale chromatin factors
  - optional
  - scale chromatin factors (separately) to value range [0..1] (min-max scaling)
  - default: True
  - recommended
- --modelType | -mod: Model type
  - optional
  - choose from "initial", "wider", "longer", "wider-longer", "sequence"
  - default: "initial"
  - experimental feature, using "initial" is recommended
- --optimizer | -opt: Optimizer to use
  - optional
  - choose from "SGD", "Adam", "RMSprop"
  - default: "SGD" 
  - SGD is recommended in conjunction with small learning rates, e.g. 1e-5
- --loss | -l: Loss function to use for optimizing
  - optional
  - choose from "MSE", "MAE", "MAPE", "MSLE", "Cosine"
  - default: "MSE"
  - refer to [Keras documentation](https://keras.io/api/losses/regression_losses/) for more information on losses
  - MSE (Mean Squared Error) is recommended
- --earlyStopping | -early: Patience for early stopping
  - optional
  - stop training prematurely if the validation loss has not been improving for the given number of epochs
  - default: no early stopping is used (i.e. the network will be trained for the number of epochs given in -ep option)
- --debugState | -dbs: Debug state
  - optional
  - choose from "0", "Figures"
  - "Figures" will add some figures to the outfolder specified above, e.g. boxplots for the chromatin factors
  - experimental feature, usage not recommended
- --figureType | -ft: Figure type
  - optional
  - the figure type and file extension for all plots
  - choose from "png", "pdf", "svg"
  - default: "png"
- --saveFreq | -sfreq: save frequency
  - optional
  - number of epochs after which an intermediate state of the trained model is saved
  - integer in [1..1000]
  - default: 50
  - values smaller than 20 not recommended
  - the checkpoints can be used for prediction e.g. if the training process crashes or
  overfits

Outputs:
- trained Model
  - stored to output file specified above
  - in .h5 container format
  - required as in input to the prediction step below
- tensorboard data
  - stored to output folder specified above
  - use `tensorboard --logdir=$outputfolderFromAbove` to visualize the progress
  - a coarse plot with loss over time is stored in the output folder specified above
- intermediate states of the network ("checkpoint" files) in .h5 format
  - stored to output folder specified above with frequency specified by means of the "-sfreq" option
  - the epoch number is part of the filename
  - can be used as input for prediction in place of final trained model
    in case the process crashes, the model overfits, etc.
- TFRecords used for feeding the training samples to the network
  - training samples will be taken on from these files on the fly 
  to keep memory load in an acceptable range when multiple training matrices and chromosomes are used
  - these files can be large, several 100s of MiB
  - the number and size of TFRecords files can be tuned using the -rs option above
  - samples will be loaded from these files on the fly during training,
  do not attempt to delete them manually while the training process is running
  - when no "-dbs" option is set, these files will be removed at the end of the training run

## Prediction
Usage:
```
python prediction.py [options]
```
Options:
- --validationmatrix | -vm: Hi-C matrix for evaluation
  - optional
  - must be in cooler format, if provided
  - allows evaluating the performance of the network for known outputs
- --chromatinPath | -cp: Path to directory with chromatin factor data
  - required
  - chromatin factors must be in bigwig format
  - filenames in the folder are important, make sure to read the notes below
- --sequenceFile | -sf: Text file containing DNA sequence
  - this option is normally ignored
  - required, if the model has been trained with the -sf option
  - experimental feature, usage not recommended
  - see description above for training.py for more details
- --outputPath | -o: Path where the results will be stored
  - required
- --trainedmodel | -trm: trained model from "training" above
  - required
  - is created by running training.py
  - must be in .h5 format
- --chromosome | -chrom: Chromosome to predict
  - required
  - format depends on your data, e.g. "chr17" or "17". In most cases, both will work.
  - specify multiple chromosomes like so: -chrom "10 13 17"
- --multiplier | -mul: Multiplier for scaling output matrix
  - required
  - default 1.0
  - output matrices are scaled to range [0...1] by default and will be multiplied by given factor
- --trainParamFile | -tpf: csv file with training parameters as created by training.py  ($outputfolder/trainParams.csv)

Outputs:
- Hi-C matrix with predictions for all specified chromosomes in cooler format
- text file with the prediction parameters in csv format for reference
- if a target matrix is provided, the evaluation loss is computed and printed to stdout, but currently not stored anywhere. Redirect output to a file, if required.

## Notes:
- Bigwig files which represent the same chromatin factor, e.g. CTCF, H3K9me3 and so on, must have the same filename in the training folder and the prediction folder. For three factors, for example, one might have the following structure:
```
./TrainingFactors/
./TrainingFactors/factor1.bigwig
./TrainingFactors/factor2.bigwig
./TrainingFactors/factor3.bigwig

./PredictionFactors/
./PredictionFactors/factor1.bigwig
./PredictionFactors/factor2.bigwig
./PredictionFactors/factor3.bigwig
```
- The actual name of the factors ("factor1.bigwig", "factor2.bigwig", "factor3.bigwig" in the example) does not matter, but it must be the same in the training folder and the prediction folder. Also see structure and filenames of example data provided in the repository under "train_test_data/".
- It is possible - but probably useless - to use the same chromosome multiple times for training and validation
- Too large values for the learning rate will make the network diverge or run into NAN losses
- Small values for the learning rate are recommended because the absolute values of the loss can become very large, causing numerical problems. Try e.g. learning rates of 10e-4.
- The implementation has been tested under LUbuntu 20.04.1 LTS and CentOS Linux 7 (Core), but should also work with other recent Linux distributions.
- Training without a supported GPU is possible (tested), but will be slow.

## Examples
The following example will work with the data provided in the github repository ("train_test_data" folder), just the ./logs folder needs to be created manually.

Training:
```
cd PATH_TO_YOUR_REPOSITORY_CLONE
mkdir logs #if not already existing
python training.py -tm train_test_data/GM12878/GSE63525_GM12878_combined_30_25kb_chr17.cool -tcp train_test_data/GM12878/ -tchroms 17 -vm train_test_data/GM12878/GSE63525_GM12878_combined_30_25kb_chr17.cool -vcp train_test_data/GM12878/ -vchroms 17 -o logs/ -ep 3000
```

Prediction:
```
python prediction.py -cp train_test_data/K562/ -o ./ -trm ./trainedModel.h5 -vm train_test_data/K562/GSE63525_K562_combined_30_25kb_chr17.cool -mul 1000 -tpf logs/trainParams.csv
```

Note that for the sake of simplicity, the same chromosome is used for training and validation in this example, which is generally a bad idea.

# Creating input files
The input files required for DNN-pCC can either be downloaded from the relevant sources or be created from raw experiment data using open source tools.
See below for detailed instructions how the example files provided in this repository have been created.

## Chromatin factors / Bigwig files
The chromatin factor data have been downloaded from [UCSC](https://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/) in bam format and processed with `samtools`, `bamCoverage` and `bigwigCompare` as follows:

```
#bash-style code
#indexing a bam file
samtools index ${BAMFILE} ${BAMFILE%bam}.bai
#creating a bigwig file from the bam file above
OUTFILE="${BAMFILE%bam}bigwig"
hg19SIZE="2685511504"
COMMAND="--numberOfProcessors 10 --bam ${BAMFILE}
COMMAND="${COMMAND} --outFileName ${OUTFILE}"
COMMAND="${COMMAND} --outFileFormat bigwig"
COMMAND="${COMMAND} --binSize 5000 --normalizeUsing RPGC"
COMMAND="${COMMAND} --effectiveGenomeSize ${hg19SIZE}"
COMMAND="${COMMAND} --scaleFactor 1.0 --extendReads 200"
COMMAND="${COMMAND} --minMappingQuality 30"
bamCoverage ${COMMAND}
#computing mean from replicate 1 and 2 bigwig files
REPLICATE1="${FOLDER1}${PROTEIN}.bigwig"
REPLICATE2="${FOLDER2}${PROTEIN}.bigwig"
OUTFILE="${OUTFOLDER}${PROTEIN}.bigwig"
COMMAND="-b1 ${REPLICATE1} -b2 ${REPLICATE2}"
COMMAND="${COMMAND} -o ${OUTFILE} -of bigwig"
COMMAND="${COMMAND} --operation mean -bs 5000"
COMMAND="${COMMAND} -p 10 -v"
bigwigCompare ${COMMAND}
```

## Hi-C matrices
Hi-C matrices have been downloaded in hic-format from the gene expression omnibus (GEO), accession key GSE63525 (data from Rao et al. 2014). Here, the "combined_30" versions have been used, which contain only high-quality reads.
All matrices have then been coverted to cooler format using `hic2cool convert`
```
hic2cool convert -r 25000 $INFILE $OUTFILE
``` 
Creating cooler matrices from raw data is beyond the scope of this readme, refer e.g. to the [hicExplorer documentation](https://hicexplorer.readthedocs.io/en/latest/content/example_usage.html) for details.