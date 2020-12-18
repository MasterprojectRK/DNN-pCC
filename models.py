import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D,Dense,Dropout,Flatten,Concatenate,MaxPool1D,Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import Input
from tensorflow.keras.applications import vgg16
import numpy as np
import os 
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt


class ConversionModel():
    def __init__(self, optimizer: str, learning_rate: float, 
                windowsize: int, flankingsize: int, nr_factors: int, 
                model_type: str, binsize: int = 0,
                nr_symbols: int = 0,
                tv_weight: float = 0.0, ssim_loss_weight: float = 0.0, 
                pixel_loss_weight: float = 1.0, score_loss_weight: float = 0.0,
                perception_loss_weight: float = 0.0,
                pixel_loss_function: str = "MSE",
                outfolder: str = "./",
                figure_type: str = "png"):
        self.optimizer = ConversionModel.getOptimizer(optimizer, learning_rate)
        self.windowsize = windowsize
        self.flankingsize = flankingsize
        self.nr_factors = nr_factors
        self.model_type = model_type
        self.binsize = binsize
        self.nr_symbols = nr_symbols
        self.tv_weight = tv_weight
        self.ssim_loss_weight = ssim_loss_weight
        self.maxfiltersize = 5
        self.score_loss_weight = score_loss_weight
        self.perception_loss_weight = perception_loss_weight
        self.pixel_loss_weight = pixel_loss_weight
        self.perception_model = None
        if self.perception_loss_weight > 0.0:
            self.perception_model = self.getPerceptionModel()
        self.pixel_loss_fn = ConversionModel.getPerPixelLoss(pixel_loss_function)
        self.model = self.buildModel()
        self.outfolder = outfolder
        self.figure_type = figure_type

    def buildModel(self, pMaxDist=None):
        if pMaxDist is None:
            maxdist = self.windowsize
        else:
            maxdist = min(self.windowsize, pMaxDist)
        sequentialModel = False
        nrFiltersList = []
        kernelSizeList = []
        nrNeuronsList = []
        dropoutRate = 0.5
        if self.model_type == "initial":
            #original model by Farre et al
            #See publication "Dense neural networks for predicting chromatin conformation" (https://doi.org/10.1186/s12859-018-2286-z).
            nrFiltersList = [1]
            kernelSizeList = [1]
            nrNeuronsList = [460,881,1690]
            sequentialModel = True
            dropoutRate = 0.1
        elif self.model_type == "wider":
            #test model with wider filters
            nrFiltersList = [1]
            kernelSizeList = [6]
            nrNeuronsList = [460,881,1690]
            sequentialModel = True
        elif self.model_type == "longer":
            #test model with more convolution filters
            nrFiltersList = [6,6]
            kernelSizeList= [1,1]
            nrNeuronsList = [1500,2400]
            sequentialModel = True
        elif self.model_type == "wider-longer":
            #test model with more AND wider convolution filters
            nrFiltersList = [6,6]
            kernelSizeList= [6,6]
            nrNeuronsList = [1500,2400]
            sequentialModel = True
        elif self.model_type == "crazy":
            return self.__buildCrazyModel()
        
        if sequentialModel == True:
            return self.__buildSequentialModel(
                                        pMaxDist=maxdist,
                                        pNrFiltersList=nrFiltersList,
                                        pKernelWidthList=kernelSizeList,
                                        pNrNeuronsList=nrNeuronsList,
                                        pDropoutRate=dropoutRate)
        elif sequentialModel == False and self.model_type == "sequence":
            return self.__buildSequenceModel(pMaxDist=maxdist, pDropoutRate=dropoutRate)
        else:
            msg = "Aborting. This type of model is not supported (yet)."
            raise NotImplementedError(msg)

    def __buildSequentialModel(self, pMaxDist, pNrFiltersList, pKernelWidthList, pNrNeuronsList, pDropoutRate):
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
        inputs = Input(shape=(2*self.flankingsize+self.windowsize,self.nr_factors), name="factorData")
        x = inputs
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
            x = Conv1D(**convParamDict)(x)
        #flatten the output from the convolutions
        x = Flatten(name="flatten_1")(x)
        #add the requested number of dense layers and dropout
        for i, nr_neurons in enumerate(pNrNeuronsList):
            layerName = "dense_" + str(i+1)
            x = Dense(nr_neurons,activation="relu",kernel_regularizer="l2",name=layerName)(x)
            layerName = "dropout_" + str(i+1)
            x = Dropout(pDropoutRate, name=layerName)(x)
        #add the output layer (corresponding to a predicted submatrix, 
        #here only the upper triangular part, along the diagonal of a Hi-C matrix)
        #this matrix may additionally be capped to maxDist, so that a trapezoid remains
        diff = self.windowsize - pMaxDist
        nr_elements_fullMatrix = int( 1/2 * self.windowsize * (self.windowsize + 1) ) #always an int, even*odd=even 
        nr_elements_capped = int( 1/2 * diff * (diff+1) )   
        nr_outputNeurons = nr_elements_fullMatrix - nr_elements_capped
        x = Dense(nr_outputNeurons,activation="relu",kernel_regularizer="l2",name="out_matrixData")(x)
        #make the output a symmetric matrix
        #i.e. add the transpose and subtract the diagonal (since it appears both)
        x = CustomReshapeLayer(self.windowsize, name="upper_triangular_layer")(x)
        y = tf.keras.layers.Lambda(lambda x1: -1 * tf.linalg.band_part(x1, 0, 0), name="get_diagonal_layer")(x)
        z = tf.keras.layers.Permute((2,1), name="transpose_layer")(x)
        x = tf.keras.layers.Add(name="make_symmetric_layer")([x,y,z])
        #scale the outputs to 0...1
        #find the maximum of each sample in the batch and multiply by its inverse
        #factor = tf.keras.layers.Lambda( lambda x1: tf.reduce_max(x1, axis=1), name="row_max_layer" )(x) #row. max for all samples
        #factor = tf.keras.layers.Lambda( lambda x1: tf.reduce_max(x1, axis=1), name="global_max_layer")(factor) #global max for all samples
        #factor = tf.keras.layers.Lambda( lambda x1: tf.cast(1./x1, tf.float32), name="max_division_layer")(factor) #the inverse for multiplication
        #x = tf.keras.layers.Multiply(name="zero_one_scaling_layer")([x, factor])
        #make the output a grayscale image
        x = tf.keras.layers.Reshape((self.windowsize, self.windowsize, 1))(x)
        
        model = Model(inputs=inputs, outputs=x)
        return model

    def __buildSequenceModel(self, pMaxDist, pDropoutRate):
        #consists of two subnets for chromatin factors and sequence, respectively
        #output neurons, see above for explanation
        diff = self.windowsize - pMaxDist
        nr_elements_fullMatrix = int( 1/2 * self.windowsize * (self.windowsize + 1) ) #always an int, even*odd=even 
        nr_elements_capped = int( 1/2 * diff * (diff+1) )   
        out_neurons = nr_elements_fullMatrix - nr_elements_capped
        #model for chromatin factors first
        kernelWidth = 1
        nr_neurons1 = 460
        nr_neurons2 = 881
        nr_neurons3 = 1690
        model1 = Sequential()
        model1.add(Input(shape=(2*self.flankingsize + self.windowsize, self.nr_factors), name="factorData"))
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
        model2.add(Input(shape=(self.windowsize*self.binsize,self.nr_symbols), name="sequenceData"))
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

    def __buildCrazyModel(self, nr_filters_list=[16,16,32,32,64], kernel_width_list=[4,4,4,4,4], nr_neurons_List=[5000,4000,3000]):  
        inputs = tf.keras.layers.Input(shape=(3*self.windowsize, self.nr_factors), name="factorData")
        #add 1D convolutions
        x = inputs
        for i, (nr_filters, kernelWidth) in enumerate(zip(nr_filters_list, kernel_width_list)):
            convParamDict = dict()
            convParamDict["name"] = "conv1D_" + str(i + 1)
            convParamDict["filters"] = nr_filters
            convParamDict["kernel_size"] = kernelWidth
            convParamDict["data_format"]="channels_last"
            if kernelWidth > 1:
                convParamDict["padding"] = "same"
            x = Conv1D(**convParamDict)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
        #make the shape of a 2D-image
        x = Conv1D(filters=64, strides=3, kernel_size=4, data_format="channels_last", padding="same", name="conv1D_final")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        y = tf.keras.layers.Permute((2,1))(x)
        z = tf.keras.layers.Lambda(lambda x1: -1*tf.linalg.band_part(x1, 0, 0))(x)
        x = tf.keras.layers.Add()([x, y, z])
        x = tf.keras.layers.Reshape((self.windowsize, self.windowsize, 1))(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model


    def lossFunction(self, predicted, target):
        loss = 0.0
        if self.pixel_loss_weight > 0.0:
            pixel_loss = self.pixel_loss_fn(target, predicted)
            loss += pixel_loss * self.pixel_loss_weight
        if self.tv_weight > 0.0:
            tv_loss = tf.reduce_mean(tf.image.total_variation(predicted))
            loss += tv_loss * self.tv_weight
        if self.ssim_loss_weight > 0.0:
            ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(target, predicted, 1., filter_size=self.maxfiltersize))
            loss += ssim_loss * self.ssim_loss_weight
        if self.perception_loss_weight > 0.0:
            target_rgb = tf.image.grayscale_to_rgb(target)
            predicted_rgb = tf.image.grayscale_to_rgb(predicted)
            target_activations = self.perception_model(target_rgb)
            predicted_activations = self.perception_model(predicted_rgb)
            perception_loss = tf.reduce_mean(tf.square( target_activations - predicted_activations ))
            loss += perception_loss * self.perception_loss_weight
        return loss

    def getPerceptionModel(self):
        model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(self.windowsize, self.windowsize, 3))  
        model.trainable = False
        structureOutput = model.get_layer("block4_conv3").output
        perceptionModel = Model(inputs=model.inputs, outputs=structureOutput)
        perceptionModel.trainable = False
        for layer in perceptionModel.layers:
            layer.trainable = False
        return perceptionModel

    @tf.function
    def train_step(self, input_factors, target):
        with tf.GradientTape() as tape:
            predicted = self.model(input_factors, training=True)
            loss = self.lossFunction(predicted, target)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function
    def validation_step(self, input_factors, target):
        predicted = self.model(input_factors, training=False)
        loss = self.lossFunction(predicted, target)
        return loss

    @tf.function
    def prediction_step(self, model, input_factors):
        pred_batch = model(input_factors, training=False)
        return pred_batch

    def fit(self, train_ds, validation_ds, nr_epochs: int = 1, steps_per_epoch: int = 1, save_freq: int = 20):
        #filename for plots
        lossPlotFilename = "lossOverEpochs.{:s}".format(self.figure_type)
        lossPlotFilename = os.path.join(self.outfolder, lossPlotFilename)
        #models for converting predictions as needed
        #lists to store loss for each epoch
        trainLossList_epochs = [] 
        valLossList_epochs = []
        a = list(train_ds.take(1).as_numpy_iterator())
        b = list(validation_ds.take(1).as_numpy_iterator())
        train_before = self.model(a[0][0], training=False)
        val_before = self.model(b[0][0], training=False)
        mse_before = tf.reduce_mean(tf.square(train_before-val_before))
        #iterate over all epochs and batches in the train/validation datasets
        #compute gradients and update weights accordingly
        for epoch in range(nr_epochs):
            if epoch % 5 == 0:
                self.generate_images(b[0][0], b[0][1], epoch, name_prefix="validation_")
                self.generate_images(a[0][0], a[0][1], epoch, name_prefix="training_")
            train_pbar = tqdm(train_ds, total=steps_per_epoch)
            train_pbar.set_description("Epoch {:05d}".format(epoch+1))
            trainLossList_batches = [] #lists to store loss for each batch
            for x, y in train_pbar:
                lossVal = self.train_step(input_factors=x, target=y["out_matrixData"])
                trainLossList_batches.append(lossVal)
                if epoch == 0:
                    train_pbar.set_postfix( {"loss": "{:.4f}".format(lossVal)} )
                else:
                    train_pbar.set_postfix( {"train loss": "{:.4f}".format(lossVal),
                                             "val loss": "{:.4f}".format(valLossList_epochs[-1])} )
            trainLossList_epochs.append(np.mean(trainLossList_batches))
            del trainLossList_batches
            valLossList_batches = []
            for x, y in validation_ds:
                val_loss = self.validation_step(input_factors=x, target=y["out_matrixData"])
                valLossList_batches.append(val_loss)
            valLossList_epochs.append(np.mean(valLossList_batches))
            del valLossList_batches
            #plot loss and save model every savefreq epochs
            if (epoch + 1) % save_freq == 0:
                checkpointFilename = "checkpoint_{:05d}.h5".format(epoch + 1)
                checkpointFilename = os.path.join(self.outfolder, checkpointFilename)
                self.model.save(filepath=checkpointFilename,save_format="h5")
                del checkpointFilename
                utils.plotLoss(pLossValueLists=[trainLossList_epochs, valLossList_epochs],
                        pNameList=["train", "validation"],
                        pFilename=lossPlotFilename)
                #save the loss values so that they can be plotted again in different formats later on
                valLossFilename = "val_loss_{:05d}.npy".format(epoch + 1)
                trainLossFilename = "train_loss_{:05d}.npy".format(epoch + 1)
                valLossFilename = os.path.join(self.outfolder, valLossFilename)
                trainLossFilename = os.path.join(self.outfolder, trainLossFilename)
                np.save(valLossFilename, valLossList_epochs)
                np.save(trainLossFilename, trainLossList_epochs)
                del valLossFilename, trainLossFilename
        checkpointFilename = "trainedModel.h5"
        checkpointFilename = os.path.join(self.outfolder, checkpointFilename)
        self.model.save(filepath=checkpointFilename,save_format="h5")
        del checkpointFilename
        utils.plotLoss(pLossValueLists=[trainLossList_epochs, valLossList_epochs],
                        pNameList=["train", "validation"],
                        pFilename=lossPlotFilename)
        train_after = self.model(a[0][0], training=False)
        val_after = self.model(b[0][0], training=False)
        mse_after = tf.reduce_mean(tf.square(train_after, val_after))
        msg = "inter mse before: {:.5f} -- inter mse after: {:.5f}".format(mse_before, mse_after)
        print(msg)
        same_mse_train = tf.reduce_mean(tf.square(train_before, train_after))
        same_mse_val = tf.reduce_mean(tf.square(val_before, val_after))
        msg = "same mse train: {:.5f} -- same mse val: {:.5f}".format(same_mse_train, same_mse_val)
        print(msg)
    
    def predict(self, test_ds, trained_model_filepath: str = ""):
        trainedModel = self.model
        if trained_model_filepath != "":
            try:
                trainedModel = tf.keras.models.load_model(trained_model_filepath, custom_objects={"CustomReshapeLayer": CustomReshapeLayer(self.windowsize)})  
            except Exception as e:
                msg = "Aborting. Could not load model, wrong file or format?"
                raise SystemExit(str(e) + "\n" + msg )
        pred_list = []
        for batch in test_ds:
            pred_batch = self.prediction_step(trainedModel, batch).numpy()
            for i in range(pred_batch.shape[0]):
                pred_list.append(pred_batch[i][:,:,0])
        return np.array(pred_list)

    def plotModel(self):
        #plot the model using workaround from tensorflow issue #38988
        modelPlotName = "model.{:s}".format(self.figure_type)
        modelPlotName = os.path.join(self.outfolder, modelPlotName)
        #self.model._layers = [layer for layer in self.model._layers if isinstance(layer, tf.keras.layers.Layer)] #workaround for plotting with custom loss functions
        tf.keras.utils.plot_model(self.model, show_shapes=True, to_file=modelPlotName)

    def generate_images(self, test_input, target, epoch: int, name_prefix: str = ""):
        prediction = self.model(test_input, training=False)
        pred0 = prediction[0].numpy()
        np.save(os.path.join(self.outfolder, "pred{:05d}".format(epoch) + name_prefix + ".npy"), pred0)
        mse = tf.reduce_mean(tf.square(prediction[0] - target["out_matrixData"][0]))
        figname = name_prefix + "pred_epoch_{:05d}.{:s}".format(epoch, self.figure_type)
        figname = os.path.join(self.outfolder, figname)
        display_list = [test_input["factorData"][0], target["out_matrixData"][0], prediction[0]]
        titleList = ['Input Image', 'Ground Truth', 'Predicted Image (MSE: {:.4f})'.format(mse)]
        fig1, axs1 = plt.subplots(1,len(display_list), figsize=(15,15))
        for i in range(len(display_list)):
            axs1[i].imshow(display_list[i])
            axs1[i].set_title(titleList[i])
        fig1.suptitle(name_prefix + "{:05d}".format(epoch))
        fig1.savefig(figname)
        plt.close(fig1)
        del fig1, axs1

    @staticmethod
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

    @staticmethod
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