import click
import tensorflow as tf
import keras
from keras.layers import Conv2D,Dense,Dropout,Flatten
from keras.models import Sequential
import numpy as np

@click.option("--trainmatrix","-tm",required=True,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Training matrix in cooler format")
@click.option("--chromatinPath","-cp", required=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where trained network will be stored")
@click.command()
def training(trainmatrix, chromatinpath, outputpath):
    
    #constants
    nr_Factors = 10 #testing only
    windowSize_bins = 80
    chromLength_bins = 3 * windowSize_bins
    matrixSize_bins = int(1/2 * windowSize_bins * (windowSize_bins + 1))
    kernelWidth = 1
    nr_neurons1 = 460
    nr_neurons2 = 881
    nr_neurons3 = 1690
    nr_neurons4 = matrixSize_bins
    nr_epochs = 10
    ##random input for now
    np.random.seed(42)
    input_train = np.random.rand(1, nr_Factors, chromLength_bins, 1)
    target_train = np.random.rand(1, matrixSize_bins)
    
    #check inputs

    #compose inputs into useful dataset
    
    
    #build neural network as described by Farre et al.
    model = Sequential()
    model.add(Conv2D(filters=1, 
                     kernel_size=(nr_Factors,kernelWidth), 
                     activation="sigmoid",
                     input_shape=(nr_Factors,chromLength_bins,1)))
    model.add(Flatten())
    model.add(Dense(nr_neurons1,activation="relu",kernel_regularizer="l2"))        
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons2,activation="relu",kernel_regularizer="l2"))
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons3,activation="relu",kernel_regularizer="l2"))
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons4,activation="relu",kernel_regularizer="l2"))
    model.compile(optimizer=keras.optimizers.SGD(), 
                  loss=keras.losses.MeanSquaredError())
    model.summary()
    
    #train the neural network
    model.fit(input_train, 
              target_train, 
              epochs= nr_epochs)

    #store the trained network

    #random input and output just for testing
    np.random.seed(111)
    input_test = np.random.rand(1, nr_Factors, chromLength_bins, 1)
    target_test = np.random.rand(1, matrixSize_bins)
    loss = model.evaluate(x=input_test, y=target_test)
    print("loss: {:.3f}".format(loss))
    pred = model.predict(x=input_test)
    print(pred[0:10])





if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter