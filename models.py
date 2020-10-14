import tensorflow
from tensorflow.keras.layers import Conv1D, Conv2D,Dense,Dropout,Flatten,Concatenate,MaxPool1D
from tensorflow.keras.models import Model, Sequential
import numpy as np 


def buildModel(pModelTypeStr, pWindowSize, pNrFactors, pBinSizeInt, pNrSymbols):
    if pModelTypeStr == "initial":
        return buildInitialModel(pWindowSize, pNrFactors)
    elif pModelTypeStr == "current":
        return buildCurrentModel(pWindowSize, pNrFactors)
    elif pModelTypeStr == "sequence":
        return buildSequenceModel(pWindowSize, pNrFactors, pBinSizeInt, pNrSymbols)
    else:
        msg = "Aborting. This type of model is not supported (yet)."
        raise NotImplementedError(msg)

def buildInitialModel(pWindowSize, pNrFactors):
    #neural network as per Farre et al.
    #See publication "Dense neural networks for predicting chromatin conformation" (https://doi.org/10.1186/s12859-018-2286-z).
    kernelWidth = 1
    nr_neurons1 = 460
    nr_neurons2 = 881
    nr_neurons3 = 1690
    nr_neurons4 = int(1/2 * pWindowSize * (pWindowSize + 1)) #always an int, even*odd=even
    model = Sequential()
    model.add(Conv1D(filters=1, 
                     kernel_size=kernelWidth, 
                     activation="sigmoid",
                     data_format="channels_last",
                     input_shape=(3*pWindowSize,pNrFactors)))
    model.add(Flatten())
    model.add(Dense(nr_neurons1,activation="relu",kernel_regularizer="l2"))        
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons2,activation="relu",kernel_regularizer="l2"))
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons3,activation="relu",kernel_regularizer="l2"))
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons4,activation="relu",kernel_regularizer="l2"))
    return model

def buildCurrentModel(pWindowSize, pNrFactors):
    return buildInitialModel(pWindowSize, pNrFactors)

def buildSequenceModel(pWindowSize, pNrFactors, pBinSizeInt, pNrSymbols):
    #consists of two subnets for chromatin factors and sequence, respectively
    #output neurons
    out_neurons = int(1/2 * pWindowSize * (pWindowSize + 1)) #always an int, even*odd=even
    #model for chromatin factors first
    kernelWidth = 1
    nr_neurons1 = 460
    nr_neurons2 = 881
    nr_neurons3 = 1690
    model1 = Sequential()
    model1.add(Conv1D(filters=1, 
                     kernel_size=kernelWidth, 
                     activation="sigmoid",
                     data_format="channels_last",
                     input_shape=(3*pWindowSize,pNrFactors)))
    model1.add(Flatten())
    model1.add(Dense(nr_neurons1,activation="relu",kernel_regularizer="l2"))        
    model1.add(Dropout(0.1))
    model1.add(Dense(nr_neurons2,activation="relu",kernel_regularizer="l2"))
    model1.add(Dropout(0.1))
    model1.add(Dense(nr_neurons3,activation="relu",kernel_regularizer="l2"))
    model1.add(Dropout(0.1))
    
    #CNN model for sequence
    filters1 = 5
    maxpool1 = 5
    kernelSize1 = 6
    kernelSize2 = 10
    model2 = Sequential()
    model2.add(Conv1D(filters=filters1, 
                      kernel_size=kernelSize1,
                      activation="relu",
                      data_format="channels_last",
                      input_shape=(pWindowSize*pBinSizeInt,pNrSymbols)))
    model2.add(MaxPool1D(maxpool1))
    model2.add(Conv1D(filters=filters1,
                      kernel_size=kernelSize1,
                      activation="relu",
                      data_format="channels_last"))
    model2.add(MaxPool1D(maxpool1))
    model2.add(Conv1D(filters=filters1,
                      kernel_size=kernelSize2,
                      activation="relu",
                      data_format="channels_last"))
    model2.add(MaxPool1D(maxpool1))
    model2.add(Conv1D(filters=filters1,
                      kernel_size=kernelSize2,
                      activation="relu",
                      data_format="channels_last"))
    model2.add(MaxPool1D(maxpool1))
    model2.add(Conv1D(filters=filters1,
                      kernel_size=kernelSize2,
                      activation="relu",
                      data_format="channels_last"))                              
    model2.add(Flatten())
    model2.add(Dense(nr_neurons2, activation="relu",kernel_regularizer="l2"))
    model2.add(Dropout(0.1))
    combined = Concatenate()([model1.output,model2.output])
    x = Dense(out_neurons,activation="relu",kernel_regularizer="l2")(combined)
 
    finalModel = Model(inputs=[model1.input, model2.input], outputs=x)
    return finalModel

##the following class is an adapted version from a tutorial at Stanford University
##https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class multiInputGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, matrixDict, factorDict, batchsize, windowsize, shuffle=True):
        self.matrixDict = matrixDict
        self.factorDict = factorDict
        self.batchsize = batchsize
        self.windowsize = windowsize
        self.shuffle = shuffle
        self.chromNames = []
        #get the chrom names
        for folder in self.factorDict:
            self.chromNames.extend([name for name in self.factorDict[folder]["data"]])
        self.chromNames = sorted(list(set(self.chromNames)))     
        #indexing
        #the idea is that data is provided chromosome-wise, then folder/matrix-wise
        #each folder/matrix - chromosome pair has its own local index
        #additionally, there is a mapping from the global index which
        #tells us from which folder/matrix - chromosome pair a sample has to be taken from
        #e.g. global sample number n is to be taken from folder/matrix k, chromosome c
        self.localIndices = dict() #local indices
        self.globalIndexMapping = dict() #mapping from global indices to local indices (chromosome, chromatin files)
        globalIndex = 0
        for chromName in self.chromNames:
            folderIndDict = dict()
            for folder in self.factorDict:
                actDataLength = self.factorDict[folder]["data"][chromName].shape[0]
                nr_samples = actDataLength - 3*windowsize + 1
                folderIndDict[folder] = np.arange(nr_samples)
                globalIndex += nr_samples - 1
                self.globalIndexMapping[globalIndex] = [chromName, folder]
            self.localIndices[chromName] = folderIndDict
        self.globalIndex = np.arange(max([x for x in self.globalIndexMapping]))
        
        if self.matrixDict is None: #for predictions, no target data is available
            self.shuffle = False
        self.on_epoch_end()

    def __len__(self):
        #some batches will come from two matrices/folders or two chromosomes
        #the last batch will have a different size
        return np.ceil(len(self.globalIndex) / self.batchsize)

    def __getitem__(self, index):
        indices = self.globalIndex[index*self.batchsize : (index+1)*self.batchsize]
        return self.__generateData(indices)

    def __generateData(self, indices):
        pass
        # factorArray = np.empty((len(indices), 3*self.windowsize, self.chromatinFactorArray.shape[1]))
        # matrixArray = None
        # if self.sparseMatrix is not None:
        #     matrixArray = np.empty((len(indices), int(self.windowsize*(self.windowsize + 1)/2)))
        # seqArray = None
        # if self.encodedDNAarray is not None:
        #     seqArray = np.empty((len(indices),self.windowsize*self.binsize, self.encodedDNAarray.shape[1]), dtype="uint8")
        # for b,i in enumerate(indices):
        #     if self.sparseMatrix is not None:
        #         #first matrix has a windowsize offset from start of chromosome (boundary handling)
        #         j = i + self.windowsize
        #         k = j + self.windowsize
        #         trainmatrix = self.sparseMatrix[j:k,j:k].todense()[np.triu_indices(self.windowsize)]
        #         matrixArray[b] = np.nan_to_num(trainmatrix)
        #     #the chromatin factors have no offset
        #     factorArray[b] = self.chromatinFactorArray[i:i+3*self.windowsize,:]
        #     if self.encodedDNAarray is not None:
        #         #take just the sequence under the current matrix to save memory
        #         j = i + self.windowsize*self.binsize
        #         k = j + self.windowsize*self.binsize
        #         seqArray[b] = self.encodedDNAarray[j:k,:]
        # if self.sparseMatrix is not None and self.encodedDNAarray is not None:
        #     return [factorArray, seqArray], matrixArray
        # elif self.sparseMatrix is not None and self.encodedDNAarray is None:
        #     return [factorArray], matrixArray
        # elif self.sparseMatrix is None and self.encodedDNAarray is not None: #prediction from factors and sequence
        #     return [factorArray, seqArray]
        # else: #matrix and sequence array are none (prediction from factors only)
        #     return factorArray

    def on_epoch_end(self):
        if self.shuffle == True:
            pass #in-place permutation of local indices
