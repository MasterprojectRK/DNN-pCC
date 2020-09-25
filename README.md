# DNN-pCC

This repository attempt predicting chromatin conformation from various chromatin factors using a neural network.
The network setup is strongly based on the paper [Dense neural networks for predicting chromatin conformation](https://doi.org/10.1186/s12859-018-2286-z) 
by Farre, Heurteau, Cuvier and Emberly (2018).

## Installation
For now, only a manual installation is supported.
To install DNN-pCC, just clone the github repository to some local directory 
e.g. using git clone for now.
To satisfy various dependencies, it is recommended to create an empty conda environment and install the following packages:
```
conda install -c anaconda tensorflow #don't use conda-forge channel here
conda install pybigwig
conda install cooler
conda install scipy
conda install matplotlib
conda install tqdm
conda install click
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
- --trainmatrix | -tm: Training Hi-C matrix
  - required, must be in cooler format
  - the bin size (resolution) is determined from the matrix
- --chromatinPath | -cp: Path to directory with chromatin factor data
  - required
  - chromatin factors must be in bigwig format
  - filenames in the folder are important, make sure to read the notes below
- --outputPath | -o: Output path for logfiles etc.
  - required 
  - data for tensorboard will be stored here
  - trained models will be stored here every 20 epochs
- --chromosome | -chrom: The chromosome used for training
  - required
  - format depends on your data, e.g. "chr17" or "17"
- --modelfilepath | -mfp: Filename for trained model
  - required
  - default "trainedModel.h5"
- --learningRate | -lr: Learning rate for stoch. grad. descent
  - required, numerical value > 1e-10, default 0.1
- --numberEpochs | -ep: Number of epochs for network
  - required, numerical value > 20
  - default 1000
- --batchsize | -bs: Batch size for network
  - required, numerical value > 5
  - default 30
- --windowsize | -ws: Chromatin window to consider for training
  - required, numerical value > 10
  - default 80
- --scaleMatrix | -scm: Scale matrix to 0...1
  - required
  - default True

Outputs:
- trained Model
  - stored to output file specified above
  - in .h5 container format
- tensorboard data
  - stored to output folder specified above
  - use `tensorboard --logdir=$outputfolderFromAbove` to visualize the progress
- intermediate states of the network
  - stored to output folder specified above every 20 epochs
  - the epoch number is part of the filename
  - can be used as input for prediction in place of final trained model
    in case the process crashes, the model overfits, etc.

## Prediction
Usage:
```
python prediction.py [options]
```
Options:
- --validationmatrix | -vm: Validation Hi-C matrix
  - optional
  - must be in cooler format, if provided
  - will allow to compute a few stats on prediction quality
- --chromatinPath | -cp: Path to directory with chromatin factor data
  - required
  - chromatin factors must be in bigwig format
  - filenames in the folder are important, make sure to read the notes below
- --outputPath|-o: Path where the results will be stored
  - required
- --trainedmodel | -trm: trained model from "training" above
  - required
  - is created by running training.py
  - must be in .h5 format
- --chromosome | -chrom: Chromosome to predict
  - required
  - format depends on your data, e.g. "chr17" or "17"
  - training and prediction chromosome need not be the same
- --multiplier | -mul: Multiplier for scaling output matrix
  - required
  - default 1.0
  - output matrices are scaled to range 0...1 by default and will be multiplied by given factor
- --trainParamFile | -tpf: csv file with training parameters as created by training.py  ($outputfolder/trainParams.csv)

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
- Too large values for the learning rate will make the SGD algorithm diverge
- Smaller values for the learning rate are recommended when not scaling the matrices because the absolute values of the loss can then become very 
large, causing numerical problems. Try e.g. learning rates of 10e-4.

## Examples
The following example will work with the data provided in the github repository ("train_test_data" folder), just the ./logs folder needs to be created manually.

Training:
```
mkdir logs #if not already existing
python training.py -tm train_test_data/GM12878/GSE63525_GM12878_combined_30_25kb_chr17.cool -cp train_test_data/GM12878/-o ./logs/ -ep 3000
```

Prediction:
```
python prediction.py -cp train_test_data/K562/ -o ./ -trm ./trainedModel.h5 -vm train_test_data/K562/GSE63525_K562_combined_30_25kb_chr17.cool -mul 1000 -tpf logs/trainParams.csv
```