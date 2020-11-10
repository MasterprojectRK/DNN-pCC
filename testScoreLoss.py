import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from testScoreModel import CustomReshapeLayer, TadInsulationScoreLayer, buildMatrixBranch, makeTrainTestSamples

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
    modelPlotName = "scoreLoss.png"

    #generate random training data
    train_features, targ_matrices, targ_scores = makeTrainTestSamples(nr_batches=nr_batches, matsize=matsize, nr_factors=nr_factors, diamondsize=diamondsize)

    #generate the model
    #there is a dummy output so that loss on matrices and scores can be computed
    model = buildModel(pMatsize=matsize, pNrFactors=nr_factors)
    model.summary()
    tf.keras.utils.plot_model(model,show_shapes=True, to_file=modelPlotName)  
    #define the two different loss functions
    #custom loss function computes TAD insulation score from predicted matrices
    losses = {
        "matrix_out": tf.keras.losses.MeanSquaredError(),
        "score_out": customLossWrapper(pMatrixsize=matsize, pDiamondsize=diamondsize)
    }
    lossWeights = {"matrix_out": 1.0, "score_out": 1.0}

    opt = tf.keras.optimizers.Adam(learning_rate=learningrate, decay=learningrate/nr_epochs)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights)

    model.fit(x=train_features, 
            y={"matrix_out": targ_matrices, "score_out": targ_scores},
            epochs=nr_epochs,
            verbose=1,
            batch_size=batchsize
    )

def buildModel(pMatsize, pNrFactors):
    inputs = Input(shape=(pMatsize,pNrFactors))
    matBranch = buildMatrixBranch(inputs, pMatsize, pNrFactors)
    dummyBranch = Activation("linear", name="score_out")(matBranch) #pass-through
    model = Model(inputs=inputs, outputs=[matBranch, dummyBranch], name="testScoreLoss")
    return model

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

if __name__ == "__main__":
    main_fn()