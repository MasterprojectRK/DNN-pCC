from utils import showMatrix,getMatrixFromCooler,composeChromatinFactors,buildMatrixArray,getBigwigFileList,rebuildMatrix,writeCooler,scaleArray
import click
import tensorflow as tf
import numpy as np


@click.option("--validationmatrix","-vm", required=False,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Target matrix in cooler format for statistical result evaluation, if available")
@click.option("--chromatinPath","-cp", required=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where results will be stored")
@click.option("--trainedmodel", "-trm", required=True,
                    type=click.Path(exists=True, dir_okay=False),
                    help="Path+Filename of trained model")
@click.option("--chromosome", "-chrom", required=True,
                    type=str, default="17", help="chromosome to predict")
@click.option("--multiplier", "-mul", required=False,
                    type=click.FloatRange(min=1.0, max=50000), default=1.0,
                    help="Predicted matrices are scaled to value range 0...1.\n Use --multiplier mmm to get range 0...mmm e.g. for better visualization")
@click.command()
def prediction(validationmatrix, 
                chromatinpath, 
                outputpath, 
                trainedmodel,
                chromosome,
                multiplier):
    
    #load the trained model first, since it contains parameters
    #which must be matched by the remaining inputs
    try:
        trainedModel = tf.keras.models.load_model(trainedmodel)
    except Exception as e:
        print(e)
        msg = "Could not load trained model {:s}. Wrong file?"
        msg = msg.format(trainedmodel)
        raise SystemExit(msg)

    #derive window size    
    inputlength = trainedModel.layers[0].input_shape[2]
    if inputlength % 3 != 0:
        msg = "Error. Expected input length is {:d} and does not divide by 3"
        msg = msg.format(inputlength)
        raise SystemExit(msg)
    windowsize = int(inputlength / 3)
    
    #for testing, provide these numbers as constants
    binSizeInt = 25000
    chromLength_bins = 3248

    #load chromatin files first, since it is faster than loading/composing
    #the validation matrices and if there are too few or too much, 
    #we can already stop here without computing the matrices.
    bigwigFileList = getBigwigFileList(chromatinpath)
    nr_chromatinFactors = trainedModel.layers[0].input_shape[1]
    if len(bigwigFileList) != nr_chromatinFactors:
        msg = "Aborting.\n"
        msg += "Did not find the required number of bigwig files in {:s}\n"
        msg += "Required: {:d}, Found: {:d}"
        msg = msg.format(chromatinpath, nr_chromatinFactors, len(bigwigFileList))
        raise SystemExit(msg)
    msg = "Found {:d} chromatin factors in {:s}."
    msg = msg.format(len(bigwigFileList),chromatinpath)
    print(msg)
    for factor in bigwigFileList:
        print(factor)

    #now load relevant part of Hi-C matrix, if provided,
    #since the bin size will be taken from there
    if validationmatrix is not None:
        sparseHiCMatrix, binSizeInt  = getMatrixFromCooler(validationmatrix,chromosome)
        if sparseHiCMatrix is None:
            msg = "Could not read HiC matrix {:s} for training, check inputs"
            msg = msg.format(validationmatrix)
            raise SystemExit(msg)
        msg = "Cooler matrix {:s} loaded.\nBin size (resolution) is {:d}bp."
        msg = msg.format(validationmatrix, binSizeInt)
        print(msg)
        print("matrix shape", sparseHiCMatrix.shape)

    #compose chromatin factors
    chromatinFactorArray = composeChromatinFactors(bigwigFileList,
                                                           pChromLength_bins=chromLength_bins, 
                                                           pBinSizeInt=binSizeInt,
                                                           pChromosomeStr=chromosome,
                                                           pWindowSize_bins=windowsize)

      
    #feed the chromatin factors through the trained model
    predMatrixArray = trainedModel.predict(x=chromatinFactorArray)
    
    #Scale to 0...1 and multiply with given multiplier for better visualization.
    predMatrixArray = scaleArray(predMatrixArray) * multiplier

    #rebuild the cooler matrix from the predictions and write out
    meanMatrix = rebuildMatrix(predMatrixArray,windowsize)
    coolerMatrixName = outputpath + "predMatrix.cool"
    writeCooler(meanMatrix,binSizeInt,coolerMatrixName,chromosome)

    if validationmatrix is not None:
        matrixArray = buildMatrixArray(sparseHiCMatrix, windowsize)
        loss = trainedModel.evaluate(x=chromatinFactorArray, y=matrixArray)
        print("loss: {:.3f}".format(loss))

    #store results

if __name__ == "__main__":
    prediction() #pylint: disable=no-value-for-parameter