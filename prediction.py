from utils import showMatrix
import click
import keras
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
@click.command()
def prediction(validationmatrix, chromatinpath, outputpath, trainedmodel):
    #check inputs
    try:
        trainedModel = keras.models.load_model(trainedmodel)
    except Exception as e:
        print(e)
        msg = "Could not load trained model {:s}. Wrong file?"
        msg = msg.format(trainedmodel)
        print(msg)

    #constants for testing
    windowSize_bins = 80

    #feed trained model with random data to check function
    np.random.seed(111)
    input_test = np.random.rand(1, 3, 240, 1)
    target_test = np.random.rand(1, 3240)
    loss = trainedModel.evaluate(x=input_test, y=target_test)
    print("loss: {:.3f}".format(loss))
    pred = trainedModel.predict(x=input_test)
    predMatrix = np.zeros((windowSize_bins,windowSize_bins))
    print(pred[0])
    predMatrix[np.triu_indices(windowSize_bins)] = pred[0]
    showMatrix(predMatrix)

    #feed inputs through neural network

    #compute statistics, if validation matrix provided

    #store results

if __name__ == "__main__":
    prediction() #pylint: disable=no-value-for-parameter