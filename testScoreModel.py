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
    nr_batches = 100
    batchsize = 15
    learningrate = 1e-5
    nr_epochs = 10
    modelPlotName = "scoreModel.png"

    #artificial random training data
    train_features, targ_matrices, targ_scores = makeTrainTestSamples(nr_batches=nr_batches, matsize=matsize, nr_factors=nr_factors, diamondsize=diamondsize)

    #build the model
    mod = buildModel(pMatsize=matsize, pDiamondsize=diamondsize, pNrFactors=nr_factors)
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
            batch_size=batchsize
    )


def buildModel(pMatsize, pDiamondsize, pNrFactors):
    inputs = Input(shape=(pMatsize,pNrFactors))
    matBranch = buildMatrixBranch(pInputs=inputs, pMatsize=pMatsize, pNrFactors=pNrFactors) 
    scoreBranch = buildScoreBranch(pInputs=matBranch, pMatsize=pMatsize, pDiamondsize=pDiamondsize)
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


def buildScoreBranch(pInputs, pMatsize, pDiamondsize):
    x = CustomReshapeLayer(matsize=pMatsize)(pInputs)
    x = TadInsulationScoreLayer(matsize=pMatsize, diamondsize=pDiamondsize, name="score_out")(x)
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



if __name__ == "__main__":
    main_fn()