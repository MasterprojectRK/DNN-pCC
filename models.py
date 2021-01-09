import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D,Dense,Dropout,Flatten,Concatenate,MaxPool1D,Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import Input
from tensorflow.keras.applications import vgg16
import numpy as np
import threading 
import utils


def buildModel(pModelTypeStr, pWindowSize, pNrFactors: int, pBinSizeInt: int, pNrSymbols: int, pBinsizeFactor: int, pFlankingSize=None, pMaxDist=None):
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
        kernelSizeList = [pBinsizeFactor]
        nrNeuronsList = [460,881,1690]
        stridesList = [pBinsizeFactor]
        paddingList = ["valid"]
        sequentialModel = True
        dropoutRate = 0.1
    elif pModelTypeStr == "wider":
        #test model with wider filters
        nrFiltersList = [1]
        kernelSizeList = [6*pBinsizeFactor]
        nrNeuronsList = [460,881,1690]
        stridesList = [pBinsizeFactor]
        paddingList = ["same"]
        sequentialModel = True
    elif pModelTypeStr == "longer":
        #test model with more convolution filters
        nrFiltersList = [6,6]
        kernelSizeList= [pBinsizeFactor,1]
        nrNeuronsList = [1500,2400]
        stridesList = [pBinsizeFactor,1]
        paddingList=["valid", "valid"]
        sequentialModel = True
    elif pModelTypeStr == "wider-longer":
        #test model with more AND wider convolution filters
        nrFiltersList = [6,6]
        kernelSizeList= [6*pBinsizeFactor,6]
        nrNeuronsList = [1500,2400]
        stridesList= [pBinsizeFactor,1]
        paddingList["same", "same"]
        sequentialModel = True
    
    if sequentialModel == True:
        return buildSequentialModel(pWindowSize=pWindowSize,
                                    pFlankingSize=flankingsize,
                                    pMaxDist=maxdist,
                                    pNrFactors=pNrFactors,
                                    pNrFiltersList=nrFiltersList,
                                    pKernelWidthList=kernelSizeList,
                                    pNrNeuronsList=nrNeuronsList,
                                    pStridesList=stridesList,
                                    pPaddingList=paddingList,
                                    pDropoutRate=dropoutRate,
                                    pBinsizeFactor=pBinsizeFactor)
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

def buildSequentialModel(pWindowSize, pFlankingSize, pMaxDist, pNrFactors: int, pNrFiltersList: list, pKernelWidthList: list, pNrNeuronsList: list, pStridesList: list, pPaddingList: list, pDropoutRate: float, pBinsizeFactor: int):
    msg = ""
    if len(pNrFiltersList) != len(pKernelWidthList) or len(pNrFiltersList) < 1:
        msg = "Error: Kernel widths and no. of filters must be specified for all 1Dconv. layers (min. 1 layer)"
        print(msg)
        return None
    if len(pStridesList) != len(pNrFiltersList) or len(pPaddingList) != len(pNrFiltersList):
        msg = "Error: Padding and strides must be given for all {:d} filters".format(len(pNrFiltersList))
        print(msg)
        return None
    if pDropoutRate <= 0 or pDropoutRate >= 1: 
        msg = "dropout must be in (0..1)"
        print(msg)
        return None
    inputs = Input(shape=((2*pFlankingSize+pWindowSize)*pBinsizeFactor,pNrFactors), name="factorData")
    x = inputs
    #add the requested number of 1D convolutions
    for i, (nr_filters, kernelWidth, strides, padding) in enumerate(zip(pNrFiltersList, pKernelWidthList, pStridesList, pPaddingList)):
        convParamDict = dict()
        convParamDict["name"] = "conv1D_" + str(i + 1)
        convParamDict["filters"] = nr_filters
        convParamDict["kernel_size"] = kernelWidth
        convParamDict["strides"] = strides
        convParamDict["padding"] = padding
        convParamDict["activation"] = "sigmoid"
        convParamDict["data_format"]="channels_last"
        convParamDict["kernel_regularizer"]=tf.keras.regularizers.l2(0.001)
        x = Conv1D(**convParamDict)(x)
    #flatten the output from the convolutions
    x = Flatten(name="flatten_1")(x)
    #add the requested number of dense layers and dropout
    for i, nr_neurons in enumerate(pNrNeuronsList):
        layerName = "dense_" + str(i+1)
        #x = Dense(nr_neurons,activation="relu",kernel_regularizer="l2",name=layerName)(x)
        x = Dense(nr_neurons,activation="relu",name=layerName)(x)
        layerName = "dropout_" + str(i+1)
        x = Dropout(pDropoutRate, name=layerName)(x)
    #add the output layer (corresponding to a predicted submatrix, 
    #here only the upper triangular part, along the diagonal of a Hi-C matrix)
    #this matrix may additionally be capped to maxDist, so that a trapezoid remains
    diff = pWindowSize - pMaxDist
    nr_elements_fullMatrix = int( 1/2 * pWindowSize * (pWindowSize + 1) ) #always an int, even*odd=even 
    nr_elements_capped = int( 1/2 * diff * (diff+1) )   
    nr_outputNeurons = nr_elements_fullMatrix - nr_elements_capped
    #x = Dense(nr_outputNeurons,activation="relu",kernel_regularizer="l2",name="out_matrixData")(x)
    x = Dense(nr_outputNeurons,activation="linear",name="out_matrixData")(x)
    model = Model(inputs=inputs, outputs=x)
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
        return tf.map_fn(self.pickItems, inputs, parallel_iterations=20, swap_memory=True)
    
    def pickItems(self, inputVec):
        sparseTriuTens = tf.SparseTensor(self.triu_indices, 
                                        values=inputVec, 
                                        dense_shape=[self.matsize, self.matsize] )
        return tf.sparse.to_dense(sparseTriuTens)

    def get_config(self):
        return {"matsize": self.matsize}

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
        return tf.map_fn(self.pickItems, inputs, parallel_iterations=20, swap_memory=True)

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
    
    def get_config(self):
        return {"matsize": self.matsize, "diamondsize": self.diamondsize}

class SymmetricFromTriuLayer(tf.keras.layers.Layer):
    '''
    make upper triangular tensors symmetric
    example:
    [[1,2,3],
     [0,4,5],
     [0,0,6]] 
    becomes:
    [[1,2,3],
     [2,4,5],
     [3,5,6]] 
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.map_fn(self.makeSymmetric, inputs, parallel_iterations=20, swap_memory=True)

    def makeSymmetric(self, inputMat):
        outMat = inputMat + tf.transpose(inputMat) - tf.linalg.band_part(inputMat, 0, 0)
        #the diagonal is the same for input and transpose, so subtract it once
        return outMat

class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self, maxval=1.0, **kwargs):
        super().__init__(**kwargs)
        self.maxval = maxval

    def call(self, inputs):
        return tf.map_fn(self.scale, inputs, parallel_iterations=20, swap_memory=True)

    def scale(self, inputs):
        minTens = tf.reduce_min(inputs)
        maxTens = tf.reduce_max(inputs)
        enumTens = tf.subtract(inputs, minTens)
        denomTens = tf.subtract(maxTens, minTens)
        def d1(): return inputs
        def d2(): return tf.math.divide(enumTens, denomTens) * self.maxval
        retTens = tf.cond(tf.math.equal(minTens, maxTens), d1, d2)
        return retTens

    def get_config(self):
        return {"maxval": self.maxval}

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

def getOptimizer(pOptimizerString, pLearningrate):
    kerasOptimizer = None
    if pOptimizerString == "SGD":
        kerasOptimizer = tf.keras.optimizers.SGD(learning_rate=pLearningrate)
    elif pOptimizerString == "Adam":
        kerasOptimizer = tf.keras.optimizers.Adam(learning_rate=pLearningrate)
    elif pOptimizerString == "RMSprop":
        kerasOptimizer = tf.keras.optimizers.RMSprop(learning_rate=pLearningrate)
    else:
        raise NotImplementedError("unknown optimizer")
    return kerasOptimizer

def lossFunction(pixelLoss="MSE", pixelWeight=1.0,
                 windowsize=None,
                 scoreWeight=0.0, diamondsize=None,\
                 tvWeight=0.0,\
                 msSSIMweight=0.0,\
                 perceptionWeight=0.0):
    #sanity check for inputs
    errorMsg = []
    if scoreWeight > 0 and (not isinstance(windowsize, int) or not isinstance(diamondsize, int)):
        errorMsg.append("If scoreWeight > 0.0, Windowsize and Diamondsize must be set (int32 > 0)")
    if isinstance(windowsize, int) and isinstance(diamondsize, int) and windowsize - 2*diamondsize <= 1:
        errorMsg.append("Diamondsize too large or Windowsize too small, Windowsize must be >> 2*Diamondsize")
    if not isinstance(windowsize, int) and (tvWeight > 0.0 or msSSIMweight > 0.0 or perceptionWeight > 0.0):
        errorMsg.append("TV loss, MS-SSIM loss and Perception loss require Windowsize")
    if len(errorMsg) > 0:
        errorMsg = "\n".join(errorMsg)
        raise ValueError(errorMsg)

    #choose appropriate loss function for "simple" regression loss
    if pixelLoss == "MSE":
        loss_fn = tf.keras.losses.MeanSquaredError()
    elif pixelLoss.startswith("Huber"):
        try:
            delta = float(pixelLoss.lstrip("Huber"))
            loss_fn = tf.keras.losses.Huber(delta=delta)       
        except:
            loss_fn = tf.keras.losses.Huber()
    elif pixelLoss == "MAE":
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    elif pixelLoss == "MAPE":
        loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
    elif pixelLoss == "MSLE":
        loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
    elif pixelLoss == "Cosine":
        loss_fn = tf.keras.losses.CosineSimilarity()
    else:
        raise NotImplementedError("unknown loss function")
    reshapeLayer = None
    tadScoreLayer = None
    makeSymmetricLayer = None
    scalingLayer = None
    perceptionModel = None 

    if isinstance(windowsize, int):
        reshapeLayer = CustomReshapeLayer(windowsize)
        makeSymmetricLayer = SymmetricFromTriuLayer()
        scalingLayer = ScalingLayer(maxval=0.999)
        #max filter size for ms-ssim
        maxfiltersize = min(int(np.floor(windowsize / 2**4)), 11)
    if scoreWeight > 0.0:
        tadScoreLayer = TadInsulationScoreLayer(windowsize, diamondsize)
    if perceptionWeight > 0.0:
        # pre-trained VGG16 model for perception loss
        model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(windowsize, windowsize, 3))  
        model.trainable = False
        structureOutput = model.get_layer("block4_conv3").output
        perceptionModel = Model(inputs=model.inputs, outputs=structureOutput)

    def loss_function(y_true, y_pred):
        loss = tf.zeros(shape=())
        if pixelWeight > 0.0:
            #compute the regression loss
            loss += loss_fn(y_true, y_pred) * pixelWeight
        if tvWeight > 0. or msSSIMweight > 0. or perceptionWeight > 0. or scoreWeight > 0.0:
            #create images from the flat vectors
            y_true_scaled = scalingLayer(y_true) #value range 0..0.999
            y_true_matrix = reshapeLayer(y_true_scaled) #2D embedding as upper triangle
            y_true_symmetric = makeSymmetricLayer(y_true_matrix) #symmetric matrix
            y_true_grayscale = tf.expand_dims(y_true_symmetric, axis=-1) #make it an image with channels last, i.e. shape = (batchsize, matsize, matsize, 1)      
            y_pred_scaled = scalingLayer(y_pred) 
            y_pred_matrix = reshapeLayer(y_pred_scaled)
            y_pred_symmetric = makeSymmetricLayer(y_pred_matrix)
            y_pred_grayscale = tf.expand_dims(y_pred_symmetric, axis=-1)
            #compute total variation loss
            if tvWeight > 0.0:
                tvLoss = tf.reduce_sum(tf.image.total_variation(y_pred_grayscale)) 
                loss += tvLoss * tvWeight
            #compute multi-scale structural similarity index
            if msSSIMweight > 0.0:
                #msSSIM = tf.image.ssim_multiscale(tf.image.convert_image_dtype(y_true_grayscale, tf.uint8), tf.image.convert_image_dtype(y_pred_grayscale, tf.uint8), 255, filter_size=maxfiltersize)
                #msSSIM = tf.where(tf.math.is_nan(msSSIM), tf.ones_like(msSSIM), msSSIM)
                #msSSIM = tf.where(tf.math.is_inf(msSSIM), tf.ones_like(msSSIM), msSSIM)
                #msSSIMloss = 1 - tf.reduce_mean( msSSIM )
                msSSIM = tf.image.ssim(y_true_grayscale, y_pred_grayscale, 1., filter_size=maxfiltersize)
                mSSIMloss = 1.0 - tf.reduce_mean(msSSIM)
                loss += mSSIMloss * msSSIMweight
            #compute TAD insulation scores
            if scoreWeight > 0.0:
                predScore = tadScoreLayer(y_pred_symmetric)
                trueScore = tadScoreLayer(y_true_symmetric)
                scoreLoss = tf.reduce_mean(tf.square(trueScore - predScore))
                loss += scoreLoss * scoreWeight
            #compute perception loss
            if perceptionWeight > 0.0:
                predRGB = tf.image.grayscale_to_rgb(y_pred_grayscale)
                trueRGB = tf.image.grayscale_to_rgb(y_true_grayscale)
                predActivations = perceptionModel(predRGB)
                trueActivations = perceptionModel(trueRGB)
                perceptionLoss = tf.reduce_mean(tf.square(trueActivations - predActivations))
                loss += perceptionLoss * perceptionWeight
        return loss
    return loss_function


def getPerceptionModel(windowsize):
    model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(windowsize, windowsize, 3))  
    model.trainable = False
    structureOutput = model.get_layer("block4_conv3").output
    perceptionModel = Model(inputs=model.inputs, outputs=structureOutput)
    perceptionModel.trainable = False
    for layer in perceptionModel.layers:
        layer.trainable = False
    return perceptionModel

def getPerPixelLoss(pixelLoss: str):
    if pixelLoss == "MSE":
        loss_fn = tf.keras.losses.MeanSquaredError()
    elif pixelLoss.startswith("Huber"):
        try:
            delta = float(pixelLoss.lstrip("Huber"))
            loss_fn = tf.keras.losses.Huber(delta=delta)       
        except:
            loss_fn = tf.keras.losses.Huber()
    elif pixelLoss == "MAE":
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    elif pixelLoss == "MAPE":
        loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
    elif pixelLoss == "MSLE":
        loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
    elif pixelLoss == "Cosine":
        loss_fn = tf.keras.losses.CosineSimilarity()
    else:
        raise NotImplementedError("unknown loss function")
    return loss_fn

def getGrayscaleConversionModel(scalingFactor, windowsize):
    inputs = Input(shape=(int(windowsize * (windowsize + 1) / 2 )) )
    x = ScalingLayer(maxval=scalingFactor)(inputs)
    x = CustomReshapeLayer(matsize=windowsize)(x)
    x = SymmetricFromTriuLayer()(x)
    x = tf.keras.layers.Lambda(lambda z: tf.expand_dims(z, axis=-1))(x)
    model = Model(inputs=inputs, outputs=x)
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False
    return model