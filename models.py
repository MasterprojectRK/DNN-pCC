import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D,Dense,Dropout,Flatten,Concatenate,MaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import Input
import numpy as np
import threading 
import utils


def buildModel(pModelTypeStr, pWindowSize, pNrFactors, pBinSizeInt, pNrSymbols, pFlankingSize=None, pMaxDist=None):
    flankingsize = None
    maxdist = None
    if pFlankingSize is None:
        flankingsize = pWindowSize
    else:
        flankingsize = pFlankingSize
    if pMaxDist is None:
        maxdist = pWindowSize
    else:
        maxdist = min(pWindowSize, pMaxDist)
    sequentialModel = False
    nrFiltersList = []
    kernelSizeList = []
    nrNeuronsList = []
    dropoutRate = 0.5
    if pModelTypeStr == "initial":
        #original model by Farre et al
        #See publication "Dense neural networks for predicting chromatin conformation" (https://doi.org/10.1186/s12859-018-2286-z).
        nrFiltersList = [1]
        kernelSizeList = [1]
        nrNeuronsList = [460,881,1690]
        sequentialModel = True
        dropoutRate = 0.1
    elif pModelTypeStr == "wider":
        #test model with wider filters
        nrFiltersList = [1]
        kernelSizeList = [6]
        nrNeuronsList = [460,881,1690]
        sequentialModel = True
    elif pModelTypeStr == "longer":
        #test model with more convolution filters
        nrFiltersList = [6,6]
        kernelSizeList= [1,1]
        nrNeuronsList = [1500,2400]
        sequentialModel = True
    elif pModelTypeStr == "wider-longer":
        #test model with more AND wider convolution filters
        nrFiltersList = [6,6]
        kernelSizeList= [6,6]
        nrNeuronsList = [1500,2400]
        sequentialModel = True
    
    if sequentialModel == True:
        return buildSequentialModel(pWindowSize=pWindowSize,
                                    pFlankingSize=flankingsize,
                                    pMaxDist=maxdist,
                                    pNrFactors=pNrFactors,
                                    pNrFiltersList=nrFiltersList,
                                    pKernelWidthList=kernelSizeList,
                                    pNrNeuronsList=nrNeuronsList,
                                    pDropoutRate=dropoutRate)
    elif sequentialModel == False and pModelTypeStr == "sequence":
        return buildSequenceModel(pWindowSize=pWindowSize,
                                  pFlankingSize=flankingsize,
                                  pMaxDist=maxdist, 
                                  pNrFactors=pNrFactors, 
                                  pBinSizeInt=pBinSizeInt, 
                                  pNrSymbols=pNrSymbols,
                                  pDropoutRate=dropoutRate)
    else:
        msg = "Aborting. This type of model is not supported (yet)."
        raise NotImplementedError(msg)

def buildSequentialModel(pWindowSize, pFlankingSize, pMaxDist, pNrFactors, pNrFiltersList, pKernelWidthList, pNrNeuronsList, pDropoutRate):
    msg = ""
    if pNrFiltersList is None or not isinstance(pNrFiltersList, list):
        msg += "No. of filters must be a list\n"
    if pKernelWidthList is None or not isinstance(pKernelWidthList, list):
        msg += "Kernel widths must be a list\n"
    if pNrNeuronsList is None or not isinstance(pNrNeuronsList, list):
        msg += "No. of neurons must be a list\n"
    if msg != "":
        print(msg)
        return None
    if len(pNrFiltersList) != len(pKernelWidthList) or len(pNrFiltersList) < 1:
        msg = "kernel widths and no. of filters must be specified for all 1Dconv. layers (min. 1 layer)"
        print(msg)
        return None
    if pDropoutRate <= 0 or pDropoutRate >= 1: 
        msg = "dropout must be in (0..1)"
        print(msg)
        return None
    model = Sequential()
    model.add(Input(shape=(2*pFlankingSize+pWindowSize,pNrFactors), name="factorData"))
    #add the requested number of 1D convolutions
    for i, (nr_filters, kernelWidth) in enumerate(zip(pNrFiltersList, pKernelWidthList)):
        convParamDict = dict()
        convParamDict["name"] = "conv1D_" + str(i + 1)
        convParamDict["filters"] = nr_filters
        convParamDict["kernel_size"] = kernelWidth
        convParamDict["activation"] = "sigmoid"
        convParamDict["data_format"]="channels_last"
        if kernelWidth > 1:
            convParamDict["padding"] = "same"
        model.add(Conv1D(**convParamDict))
    #flatten the output from the convolutions
    model.add(Flatten(name="flatten_1"))
    #add the requested number of dense layers and dropout
    for i, nr_neurons in enumerate(pNrNeuronsList):
        layerName = "dense_" + str(i+1)
        model.add(Dense(nr_neurons,activation="relu",kernel_regularizer="l2",name=layerName))
        layerName = "dropout_" + str(i+1)
        model.add(Dropout(pDropoutRate, name=layerName))
    #add the output layer (corresponding to a predicted submatrix, 
    #here only the upper triangular part, along the diagonal of a Hi-C matrix)
    #this matrix may additionally be capped to maxDist, so that a trapezoid remains
    diff = pWindowSize - pMaxDist
    nr_elements_fullMatrix = int( 1/2 * pWindowSize * (pWindowSize + 1) ) #always an int, even*odd=even 
    nr_elements_capped = int( 1/2 * diff * (diff+1) )   
    nr_outputNeurons = nr_elements_fullMatrix - nr_elements_capped
    model.add(Dense(nr_outputNeurons,activation="relu",kernel_regularizer="l2",name="out_matrixData"))
    return model

def buildSequenceModel(pWindowSize, pFlankingSize, pMaxDist, pNrFactors, pBinSizeInt, pNrSymbols, pDropoutRate):
    #consists of two subnets for chromatin factors and sequence, respectively
    #output neurons, see above for explanation
    diff = pWindowSize - pMaxDist
    nr_elements_fullMatrix = int( 1/2 * pWindowSize * (pWindowSize + 1) ) #always an int, even*odd=even 
    nr_elements_capped = int( 1/2 * diff * (diff+1) )   
    out_neurons = nr_elements_fullMatrix - nr_elements_capped
    #model for chromatin factors first
    kernelWidth = 1
    nr_neurons1 = 460
    nr_neurons2 = 881
    nr_neurons3 = 1690
    model1 = Sequential()
    model1.add(Input(shape=(2*pFlankingSize + pWindowSize,pNrFactors), name="factorData"))
    model1.add(Conv1D(filters=1, 
                     kernel_size=kernelWidth, 
                     activation="sigmoid",
                     data_format="channels_last"))
    model1.add(Flatten())
    model1.add(Dense(nr_neurons1,activation="relu",kernel_regularizer="l2"))        
    model1.add(Dropout(pDropoutRate))
    model1.add(Dense(nr_neurons2,activation="relu",kernel_regularizer="l2"))
    model1.add(Dropout(pDropoutRate))
    model1.add(Dense(nr_neurons3,activation="relu",kernel_regularizer="l2"))
    model1.add(Dropout(pDropoutRate))
    
    #CNN model for sequence
    filters1 = 5
    maxpool1 = 5
    kernelSize1 = 6
    kernelSize2 = 10
    model2 = Sequential()
    model2.add(Input(shape=(pWindowSize*pBinSizeInt,pNrSymbols), name="sequenceData"))
    model2.add(Conv1D(filters=filters1, 
                      kernel_size=kernelSize1,
                      activation="relu",
                      data_format="channels_last"))
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
    model2.add(Dropout(pDropoutRate))
    combined = Concatenate()([model1.output,model2.output])
    x = Dense(out_neurons,activation="relu",kernel_regularizer="l2")(combined)
 
    finalModel = Model(inputs=[model1.input, model2.input], outputs=x)
    return finalModel


class multiInputGenerator(tf.keras.utils.Sequence):
    def __init__(self, matrixDict, factorDict, 
                 batchsize, windowsize, flankingsize=None,
                 binsize=None, shuffle=True,
                 maxdist=None):
        self.matrixDict = matrixDict
        self.factorDict = factorDict
        self.batchsize = batchsize
        #the size of the submatrix
        self.windowsize = windowsize 
        #the size of the flanking regions left/right of the submatrix
        if flankingsize is not None:
            self.flankingsize = flankingsize
        else:
            self.flankingsize = self.windowsize
        if maxdist is not None:
            self.maxdist = min(maxdist, windowsize)
        else:
            self.maxdist = self.windowsize
        self.matrixSize = int(self.windowsize * (self.windowsize + 1) / 2)
        if self.maxdist is not None:
            self.matrixSize -= int( (self.windowsize - self.maxdist)*(self.windowsize - self.maxdist + 1) / 2 )
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
        #list with DNA sequences per chromosome, two entries
        #the idea is that all threads are usually working on the current entry
        #the first thread to require more data will load it to the second entry
        self._dnaSequenceLock = threading.Lock()
        self.currentSequenceIndex = 0 #to be updated only with lock, 0 or 1
        self.dnaSequenceList = [dict(), dict()] #to be updated only with lock
        self.sequencePresent = False
        for folder in self.factorDict:
            if "seqFile" in self.factorDict[folder]:
                self.sequencePresent = True
                break
        self.sequenceSymbolSet = None
        self.binsize = binsize
        if self.sequencePresent:
            tmpSymbolsList = [factorDict[folder]["seqSymbols"] for folder in factorDict]
            self.sequenceSymbolSet = set([item for sublist in tmpSymbolsList for item in sublist])
        #initial shuffling of local indices
        self.on_epoch_end()

    def __len__(self):
        #some batches will come from two matrices/folders or two chromosomes
        #the last batch will have a different size
        return int(np.ceil(len(self.globalIndex) / self.batchsize))

    def __getitem__(self, index):
        indices = self.globalIndex[index*self.batchsize : (index+1)*self.batchsize]
        return self.__generateData(indices)

    def __generateData(self, globalIndices):
        #initialize the return arrays
        #len(globalIndices) is generally equal to batchsize
        #but the last batch may be smaller
        chromatinFactorArray = np.empty((len(globalIndices), self.windowsize + 2*self.flankingsize, self.nr_factors))
        matrixArray = None
        if self.matrixDict is not None:
            matrixArray = np.empty((len(globalIndices), self.matrixSize))
        sequenceArray = None
        if self.sequencePresent:
            sequenceArray = np.empty((len(globalIndices), int(self.windowsize * self.binsize), len(self.sequenceSymbolSet)), dtype=np.uint8)
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
                if self.sequencePresent == True:
                    sequenceArray[b] = self.__checkGetDNAsequence(currentFolder, currentChrom, ind)
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
                if self.sequencePresent == True:
                    sequenceArray[b] = self.__checkGetDNAsequence(currentFolderLower, currentChromLower, ind)
            indOffset = len(localIndicesLower)
            for b, ind in enumerate(localIndicesUpper):
                chromatinFactorArray[b+indOffset] = self.__getFactorData(currentFolderUpper, currentChromUpper, ind)
                if matrixArray is not None:
                    mName = self.factorDict[currentFolderUpper]["matrixName"]
                    matrixArray[b+indOffset] = self.__getMatrixData(mName,currentChromUpper, ind)
                if self.sequencePresent == True:
                    sequenceArray[b+indOffset] = self.__checkGetDNAsequence(currentFolderUpper, currentChromUpper, ind)
        if matrixArray is not None:
            if self.sequencePresent == True:
                return [chromatinFactorArray, sequenceArray], matrixArray
            else:
                return [chromatinFactorArray], matrixArray
        else:
            if self.sequencePresent == True:
                return [chromatinFactorArray, sequenceArray]
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
                nr_samples = actDataLength - (self.windowsize + 2*self.flankingsize) + 1
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
        #the 0-th matrix starts flankingsize away from the boundary
        startInd = idx + self.flankingsize
        stopInd = startInd + self.windowsize
        trainmatrix = None
        if self.maxdist == self.windowsize: #triangles, i. e. full submatrices
            trainmatrix = self.matrixDict[mName]["data"][chromName][startInd:stopInd,startInd:stopInd].todense()[np.triu_indices(self.windowsize)]
        else: #trapezoids, i.e. distance limited submatrices
            trainmatrix = self.matrixDict[mName]["data"][chromName][startInd:stopInd,startInd:stopInd].todense()[np.mask_indices(self.windowsize, utils.maskFunc, self.maxdist)]
        trainmatrix = np.nan_to_num(trainmatrix)
        return trainmatrix

    def __getFactorData(self, folder, chromName, idx):
        startInd = idx
        stopInd = startInd + self.windowsize + 2*self.flankingsize
        return self.factorDict[folder]["data"][chromName][startInd:stopInd,:]
    
    def __checkGetDNAsequence(self, folder, chromname, idx):
        #check if the DNA sequence is already loaded
        #and reload, if not
        #this method should work as long as there are 
        #more batches in each chromosome than worker threads
        startInd = idx + self.flankingsize * self.binsize
        stopInd = startInd + self.windowsize * self.binsize            
        requiredSeqFile = self.factorDict[folder]["seqFile"]
        requiredChromName = self.factorDict[folder]["seqID"][chromname]
        requiredSeqIdentifier = requiredSeqFile + "_" + str(requiredChromName)
        retArr = None
        with self._dnaSequenceLock:
            #check if the seqFile is loaded already
            oldSeqIndex = (self.currentSequenceIndex + 1) % 2
            currentSeqDict = self.dnaSequenceList[self.currentSequenceIndex]
            oldSeqDict = self.dnaSequenceList[oldSeqIndex]
            if requiredSeqIdentifier not in currentSeqDict \
                and requiredSeqIdentifier not in oldSeqDict:
                presentIdListOld = [identifier for identifier in oldSeqDict]
                #presentIdListCurr = [identifier for identifier in currentSeqDict]
                #load the sequence data from disk
                tmp_seqStr = utils.readSequencesPerId(requiredSeqFile, requiredChromName)
                encodedSeqArr = utils.fillEncodedSequence(utils.encodeSequence(tmp_seqStr),self.binsize)
                retArr = encodedSeqArr[startInd:stopInd]
                #replace "old" dict with new data and update pointer to "current" dict
                self.dnaSequenceList[oldSeqIndex] = {requiredSeqIdentifier: encodedSeqArr}
                self.currentSequenceIndex = oldSeqIndex
                threadName = threading.current_thread().getName()
                msg = "Thread {:s} has loaded {:s}, replacing {:s}"
                msg = msg.format(threadName, requiredSeqIdentifier, ", ".join(presentIdListOld))
                print(msg)
            elif requiredSeqIdentifier in currentSeqDict:
                retArr = currentSeqDict[requiredSeqIdentifier][startInd:stopInd]
            else:
                retArr = oldSeqDict[requiredSeqIdentifier][startInd:stopInd]
        return retArr

class CustomReshapeLayer(tf.keras.layers.Layer):
    '''
    reshape a 1D tensor such that it represents 
    the upper triangular part of a square 2D matrix with shape (matsize, matsize)
    #example: 
     [1,2,3,4,5,6] => [[1,2,3],
                       [0,4,5],
                       [0,0,6]]
    '''
    def __init__(self, matsize, **kwargs):
        super(CustomReshapeLayer, self).__init__(**kwargs)
        self.matsize = matsize
        self.triu_indices = [ [x,y] for x,y in zip(np.triu_indices(self.matsize)[0], np.triu_indices(self.matsize)[1]) ]

    def call(self, inputs):      
        return tf.map_fn(self.pickItems, inputs)
    
    def pickItems(self, inputVec):
        sparseTriuTens = tf.SparseTensor(self.triu_indices, 
                                        values=inputVec, 
                                        dense_shape=[self.matsize, self.matsize] )
        return tf.sparse.to_dense(sparseTriuTens)

class TadInsulationScoreLayer(tf.keras.layers.Layer):
    '''
    Computes TAD insulation scores for square 2D tensors with shape (matsize,matsize)
    and fixed-size insulation blocks ("diamonds") with shape (diamondsize,diamondsize)
    '''
    def __init__(self, matsize, diamondsize, **kwargs):
        super(TadInsulationScoreLayer, self).__init__(**kwargs)
        self.matsize = int(matsize)
        self.diamondsize = int(diamondsize)
        if self.diamondsize >= self.matsize:
            msg = "Diamondsize {:d} must be smaller than matrix size {:d}"
            msg = msg.format(self.diamondsize, self.matsize)
            raise ValueError(msg)
    
    def call(self, inputs):
        return tf.map_fn(self.pickItems, inputs)

    def pickItems(self, inputMat):
        nr_diamonds = self.matsize - 2*self.diamondsize
        start_offset = self.diamondsize
        rowEndList = [i + start_offset for i in range(nr_diamonds)]
        rowStartList = [i-self.diamondsize for i in rowEndList] 
        columnStartList = [i+1 for i in rowEndList]
        columnEndList = [i+self.diamondsize for i in columnStartList]
        l = [ inputMat[i:j,k:l] for i,j,k,l in zip(rowStartList,rowEndList,columnStartList,columnEndList) ]
        l = [ tf.reduce_mean(i) for i in l ]
        return tf.stack(l)

def customLossWrapper(pMatrixsize, pDiamondsize):
    def customLoss(y_true, y_pred):
        #compute the score from the predicted (flattened) upper triangular matrix
        predScore = CustomReshapeLayer(matsize=pMatrixsize)(y_pred)
        predScore = TadInsulationScoreLayer(matsize=pMatrixsize,diamondsize=pDiamondsize)(predScore)
        #compute mean squared error for TAD insulation score
        predLoss = tf.square(y_true - predScore)
        predLoss = tf.reduce_mean(predLoss)
        return predLoss
    return customLoss
