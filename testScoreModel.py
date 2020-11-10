import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input
from tensorflow.keras.models import Model

np.random.seed(35)

def main_fn():
    #params
    matsize = 10
    diamondsize = 2
    nr_factors = 6
    nr_batches = 10
    learningrate = 1e-5
    nr_epochs = 10
    modelPlotName = "scoreModel.png"

    train_features, targ_matrices, targ_scores = makeTrainTestSamples(nr_batches=nr_batches, matsize=matsize, nr_factors=nr_factors, diamondsize=diamondsize)


    #build the model
    mod = buildModel(matsize, diamondsize, nr_factors, nr_batches)
    mod.summary()
    tf.keras.utils.plot_model(mod,show_shapes=True, to_file=modelPlotName)
    #compile the model
    losses = {
        "matrix_out": tf.keras.losses.MeanSquaredError(),
        "score_out": tf.keras.losses.MeanSquaredError()
    }
    lossWeights = {"matrix_out": 1.0, "score_out": 1.0}
    opt = tf.keras.optimizers.Adam(learning_rate=learningrate, decay=learningrate/nr_epochs)
    mod.compile(optimizer=opt, loss=losses, loss_weights=lossWeights)


    mod.fit(x=train_features, 
            y={"matrix_out": targ_matrices, "score_out": targ_scores},
            epochs=nr_epochs,
            verbose=1,
    )





def buildModel(pMatsize, pDiamondsize, pNrFactors, pBatchsize):
    inputs = Input(shape=(pMatsize,pNrFactors))
    matBranch = buildMatrixBranch(inputs, pMatsize, pNrFactors) 
    scoreBranch = buildScoreBranch(matBranch, pMatsize, pDiamondsize, pBatchsize)
    #fullModel = Model(inputs=matModel.inputs, outputs=[matModel.outputs,scoreModel.outputs])
    model = Model(inputs=inputs, outputs=[matBranch,scoreBranch], name="testScoreModel")
    return model

def buildMatrixBranch(pInputs,pMatsize, pNrFactors):
    out_neurons = int( pMatsize * (pMatsize + 1) / 2 )
    x = Conv1D(filters=1, activation="relu", kernel_size=1,data_format="channels_last")(pInputs)
    x = Flatten()(x)
    x = Dense(pMatsize, activation="relu")(x)
    x = Dense(25, activation="relu")(x)
    x = Dense(out_neurons, activation="relu", name="matrix_out")(x)
    return x

def buildScoreBranch(pInputs, pMatsize, pDiamondsize, pBatchsize):
    x = CustomReshapeLayer(pMatsize, pBatchsize)(pInputs)
    x = DiamondLayer(pMatsize,pDiamondsize,pBatchsize, name="score_out")(x)
    return x

def makeTrainTestSamples(nr_batches, matsize, nr_factors, diamondsize):
    #train features
    train_features = np.random.rand(nr_batches,matsize,nr_factors).astype("float32")
    #target matrices and scores
    targ_matrices = np.random.rand(nr_batches,int( matsize*(matsize+1)/2 )).astype("float32")
    #target scores computed from target matrices
    nr_diamonds = matsize - 2*diamondsize
    targ_scores = np.zeros(shape=(nr_batches, nr_diamonds), dtype="float32")
    for i in range(nr_batches):
        tmpMat = np.zeros(shape=(matsize, matsize), dtype="float32")
        tmpMat[np.triu_indices(matsize)] = targ_matrices[i]
        rowEndList = [i + diamondsize for i in range(nr_diamonds)]
        rowStartList = [i-diamondsize for i in rowEndList] 
        columnStartList = [i+1 for i in rowEndList]
        columnEndList = [i+diamondsize for i in columnStartList]
        l = [ tmpMat[r:s,t:u] for r,s,t,u in zip(rowStartList,rowEndList,columnStartList,columnEndList) ]
        l = [np.mean(i1) for i1 in l]
        targ_scores[i] = np.array(l)
    return train_features, targ_matrices, targ_scores


class CustomReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, matsize, batchsize):
        super(CustomReshapeLayer, self).__init__()
        self.matsize = matsize
        self.tensList = []
        #self.batchsize = int(batchsize)
        self.triu_indices = [ [x,y] for x,y in zip(np.triu_indices(self.matsize)[0], np.triu_indices(self.matsize)[1]) ]
    
    def build(self, input_shape):
        print("ins", input_shape)
        self.batchsize = input_shape[0]

    @tf.function
    def call(self, inputs):
        #for i in range(self.batchsize):
        if self.batchsize is not None:
            for i in range(self.batchsize):
                try: 
                    self.tensList.append(self.pickItems(i, inputs))
                except:
                    #for batches that do not have the full number of samples
                    #e.g. last batch in a dataset
                    #no idea how to do this properly
                    pass
        return tf.stack(self.tensList)

    def pickItems(self, batch_index, inputs):
        #pick the right items from flattened upper triangular matrix part
        #and stack them such that a (matrix_size, matrix_size) shaped tensor is created
        sparseTriuTens = tf.SparseTensor(self.triu_indices, 
                                        values=inputs[batch_index], 
                                        dense_shape=[self.matsize, self.matsize] )
        return tf.sparse.to_dense(sparseTriuTens)

class DiamondLayer(tf.keras.layers.Layer):
    def __init__(self, matsize, diamondsize, batchsize, **kwargs):
        super(DiamondLayer, self).__init__(**kwargs)
        self.matsize = int(matsize)
        self.diamondsize = int(diamondsize)
        self.batchsize = int(batchsize)
        self.tensList = []
    
    @tf.function
    def call(self, inputs):
        for i in range(self.batchsize):
            try:
                self.tensList.append(self.pickElements(i, inputs))
            except:
                #for batches that do not have the full number of samples
                #e.g. last batch in a dataset
                #no idea how to do this properly
                pass
        return tf.stack(self.tensList)

    def pickElements(self, batch_index, inputs):
        nr_diamonds = self.matsize - 2*self.diamondsize
        start_offset = self.diamondsize
        rowEndList = [i + start_offset for i in range(nr_diamonds)]
        rowStartList = [i-self.diamondsize for i in rowEndList] 
        columnStartList = [i+1 for i in rowEndList]
        columnEndList = [i+self.diamondsize for i in columnStartList]
        l = [ inputs[batch_index][i:j,k:l] for i,j,k,l in zip(rowStartList,rowEndList,columnStartList,columnEndList) ]
        l = [ tf.reduce_mean(i) for i in l ]
        return tf.stack(l)

if __name__ == "__main__":
    main_fn()