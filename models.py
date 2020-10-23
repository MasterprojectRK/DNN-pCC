import tensorflow
from tensorflow.keras.layers import Conv1D, Conv2D,Dense,Dropout,Flatten,Concatenate,MaxPool1D
from tensorflow.keras.models import Model, Sequential
import numpy as np 
import utils


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

def buildModelMoreConvolutions(pWindowSize, pNrFactors):
    kernelWidth = 1
    nr_filters1 = 6
    nr_filters2 = 6
    nr_neurons1 = 1500
    nr_neurons2 = 2400
    nr_neurons3 = int(1/2 * pWindowSize * (pWindowSize + 1)) #always an int, even*odd=even
    model = Sequential()
    model.add(Conv1D(name="input_layer",
                     filters=nr_filters1, 
                     kernel_size=kernelWidth, 
                     activation="sigmoid",
                     data_format="channels_last",
                     input_shape=(3*pWindowSize,pNrFactors)))
    model.add(Conv1D(name="conv1D_1",
                     filters=nr_filters2, 
                     kernel_size=kernelWidth, 
                     activation="sigmoid",
                     data_format="channels_last"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(nr_neurons1,activation="relu",kernel_regularizer="l2", name="dense_1"))        
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons2,activation="relu",kernel_regularizer="l2", name="dense_2"))
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons3,activation="relu",kernel_regularizer="l2", name="output_layer"))
    return model


def buildCurrentModel(pWindowSize, pNrFactors):
    return buildModelMoreConvolutions(pWindowSize, pNrFactors)

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


class multiInputGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, matrixDict, factorDict, batchsize, windowsize, shuffle=True):
        self.matrixDict = matrixDict
        self.factorDict = factorDict
        self.batchsize = batchsize
        self.windowsize = windowsize
        self.shuffle = shuffle
        self.nr_factors = max([self.factorDict[folder]["nr_factors"] for folder in self.factorDict])
        #get the chrom names
        self.chromNames = []
        for folder in self.factorDict:
            self.chromNames.extend([name for name in self.factorDict[folder]["data"]])
        self.chromNames = sorted(list(set(self.chromNames)))     
        #index the samples provided in the dicts
        self.localIndices, self.globalIndex, self.globalIndexMapping = self.__buildIndex()
        #disable shuffling when no target data is available (i.e. for predictions)
        if self.matrixDict is None: 
            self.shuffle = False
        #initial shuffling of local indices
        self.on_epoch_end()

    def __len__(self):
        #some batches will come from two matrices/folders or two chromosomes
        #the last batch will have a different size
        return int(np.ceil(len(self.globalIndex) / self.batchsize))

    def __getitem__(self, index):
        indices = self.globalIndex[index*self.batchsize : (index+1)*self.batchsize]
        #print("\nind", indices)
        return self.__generateData(indices)

    def __generateData(self, globalIndices):
        #initialize the return arrays
        #len(globalIndices) is generally equal to batchsize
        #but the last batch may be smaller
        chromatinFactorArray = np.empty((len(globalIndices), 3*self.windowsize, self.nr_factors))
        matrixArray = None
        if self.matrixDict is not None:
            matrixArray = np.empty((len(globalIndices), int(self.windowsize*(self.windowsize + 1)/2)))
        #find the correct global -> local mapping for of the first and last global index in the batch
        indBreakpoints = [bp for bp in self.globalIndexMapping]
        lowerMapInd = next(ind for ind,val in enumerate(indBreakpoints) if val > globalIndices[0])
        upperMapInd = next(ind for ind,val in enumerate(indBreakpoints) if val > globalIndices[-1])
        if lowerMapInd == upperMapInd:
            #all global Indices belong to samples from the same chrom / folder
            localAccessInds = globalIndices.copy() #avoid side effects on globalIndices
            if lowerMapInd > 0:
                localAccessInds -= indBreakpoints[lowerMapInd - 1]
            currentChrom = self.globalIndexMapping[indBreakpoints[lowerMapInd]][0]
            currentFolder = self.globalIndexMapping[indBreakpoints[lowerMapInd]][1]
            localIndices = self.localIndices[currentChrom][currentFolder][localAccessInds]
            #now get the data
            for b, ind in enumerate(localIndices):
                chromatinFactorArray[b] = self.__getFactorData(currentFolder,currentChrom,ind)
                if matrixArray is not None:
                    mName = self.factorDict[currentFolder]["matrixName"]
                    matrixArray[b] = self.__getMatrixData(mName,currentChrom,ind)
                if ind == 0 and matrixArray is not None:
                    m_arr = matrixArray[b].copy()
                    m_mat = np.zeros((self.windowsize, self.windowsize))
                    m_mat[np.triu_indices(self.windowsize)] = m_arr
                    m_mat = np.transpose(m_mat)
                    m_mat[np.triu_indices(self.windowsize)] = m_arr
                    plotTitle = "matrix_" + str(ind) + "_" + str(currentChrom) + "_" + currentFolder.replace("/", "--")
                    #utils.plotMatrix(m_mat, plotTitle + ".png", plotTitle)
                    np.save(plotTitle, m_mat)
                    plotName = "factors_" + str(ind) + "_" + str(currentChrom) + "_" + currentFolder.replace("/", "--")
                    #utils.plotChromatinFactors(chromatinFactorArray[b].copy(),25000,currentChrom,currentFolder,plotName + ".png")
                    np.save(plotName, chromatinFactorArray[b].copy())
            #print("\ngbi", globalIndices)
            #print("locAccInd", localAccessInds)
            #print("localInd" ,localIndices)
            #print("chrom", currentChrom, "folder", currentFolder)
        else:
            #some samples are from one chrom/folder pair and the others from another one 
            #split the access indices into lower and upper part
            indSplit = next(ind for ind,val in enumerate(globalIndices) if val > indBreakpoints[lowerMapInd])
            localAccessIndsLower = globalIndices.copy()[:indSplit-1] #avoid side effects on globalIndices
            if lowerMapInd > 0:
                localAccessIndsLower -= indBreakpoints[lowerMapInd - 1]
            localAccessIndsUpper = globalIndices[indSplit-1:] - indBreakpoints[upperMapInd - 1]
            currentChromLower = self.globalIndexMapping[indBreakpoints[lowerMapInd]][0]
            currentFolderLower = self.globalIndexMapping[indBreakpoints[lowerMapInd]][1]
            currentChromUpper = self.globalIndexMapping[indBreakpoints[upperMapInd]][0]
            currentFolderUpper = self.globalIndexMapping[indBreakpoints[upperMapInd]][1]
            #print("\nchrom lower", currentChromLower, "chrom upper", currentChromUpper)
            #print("\nfolder lower", currentFolderLower, "folder upper", currentFolderUpper)
            #print("\nsplit lower", localAccessIndsLower, "split upper", localAccessIndsUpper)
            #print("gbi", globalIndices)
            localIndicesLower = self.localIndices[currentChromLower][currentFolderLower][localAccessIndsLower]
            localIndicesUpper = self.localIndices[currentChromUpper][currentFolderUpper][localAccessIndsUpper]
            #now load the data
            for b, ind in enumerate(localIndicesLower):
                chromatinFactorArray[b] = self.__getFactorData(currentFolderLower,currentChromLower,ind)
                if matrixArray is not None:
                    mName = self.factorDict[currentFolderLower]["matrixName"]
                    matrixArray[b] = self.__getMatrixData(mName,currentChromLower, ind)
            indOffset = len(localIndicesLower)
            for b, ind in enumerate(localIndicesUpper):
                chromatinFactorArray[b+indOffset] = self.__getFactorData(currentFolderUpper, currentChromUpper, ind)
                if matrixArray is not None:
                    mName = self.factorDict[currentFolderUpper]["matrixName"]
                    matrixArray[b+indOffset] = self.__getMatrixData(mName,currentChromUpper, ind)
        if matrixArray is not None:
            return [chromatinFactorArray], matrixArray
        else:
            return chromatinFactorArray

    def on_epoch_end(self):
        if self.shuffle == True:
            #permutation of local indices
            for chromname in self.chromNames:
                for folder in self.localIndices[chromname]:
                    np.random.shuffle(self.localIndices[chromname][folder])
            

    
    def __buildIndex(self):
        #indexing of samples
        #the idea is that data is provided chromosome-wise, then folder/matrix-wise
        #i.e. chr1 - folder1, folder2,... chr2, folder 1, folder 2
        #because the matrices and sequences are structured in chromosomes
        #each folder/matrix - chromosome pair has its own local index
        #this allows keeping the global index linearly increasing while shuffling local indices.
        #There is a mapping from the global index which
        #tells us from which folder/matrix - chromosome pair a sample has to be taken from
        #e.g. global sample number n is to be taken from folder/matrix k, chromosome c
        localIndices = dict() #for each chromosome/folder pair
        globalIndexMapping = dict() #mapping from global indices to local indices (chromosome/folder pairs)
        globalIndex = 0
        for chromName in self.chromNames:
            folderIndDict = dict()
            for folder in self.factorDict:
                actDataLength = self.factorDict[folder]["data"][chromName].shape[0]
                nr_samples = actDataLength - 3*self.windowsize + 1
                folderIndDict[folder] = np.arange(nr_samples)
                globalIndex += nr_samples
                globalIndexMapping[globalIndex] = [chromName, folder]
                #this means that all samples with global index [0...$globalIndex) 
                #are taken from chromosome $chromName, bigwig folder $folder
                #and so on for higher indices
            localIndices[chromName] = folderIndDict
        globalIndex = np.arange(max([x for x in globalIndexMapping]))
        return localIndices, globalIndex, globalIndexMapping

    def __getMatrixData(self, mName, chromName, idx):
        if self.matrixDict is None:
            return None
        #the 0-th matrix starts a windowsize away from the boundary
        startInd = idx + self.windowsize
        stopInd = startInd + self.windowsize
        trainmatrix = self.matrixDict[mName]["data"][chromName][startInd:stopInd,startInd:stopInd].todense()[np.triu_indices(self.windowsize)]
        trainmatrix = np.nan_to_num(trainmatrix)
        return trainmatrix

    def __getFactorData(self, folder, chromName, idx):
        startInd = idx
        stopInd = startInd + 3*self.windowsize
        return self.factorDict[folder]["data"][chromName][startInd:stopInd,:]
