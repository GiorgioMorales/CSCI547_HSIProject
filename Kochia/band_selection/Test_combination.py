import pickle
from tensorflow_model_optimization.sparsity import keras as sparsity



data = 'WEED'
windowSize = train_x.shape[1]
classes = 3

tf.keras.backend.clear_session()
# Load trained pruned network
#loaded_model = tf.keras.models.load_model("Kochia_hyper3DNet_pruned.h5")

# Load model without pruning
loaded_model = tf.keras.models.load_model("weights-hyper3dnetWEED1-best_3layers_4filters.h5")

loaded_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adadelta',
        metrics=['accuracy'])

windowSize = train_x.shape[1]
classes = 3
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

epochs = 8
batch_size = 32;
num_train_samples = train_x.shape[0]
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs

data = 'WEED'
SA = np.zeros((10, train_x.shape[3],))
ntrain = 1
for train, test in kfold.split(train_x, train_y):

    new_pruning_params = {
          'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.10,
                                                      final_sparsity=0.30,
                                                      begin_step=0,
                                                      end_step=end_step,
                                                      frequency=100)
    }
    new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
    new_pruned_model.load_weights("pruned-weights-hyper3dnet" + data + str(ntrain) + "-best_3layers_4filters.h5")

    ytest = tf.keras.utils.to_categorical(train_y[test]).astype(np.int32)
    ypred1 = new_pruned_model.predict(train_x[test])
    loss1 = tf.keras.losses.categorical_crossentropy(ytest, ypred1)
    print("Fold:" + str(ntrain) +" , original accuracy: " + str(np.sum(ypred1*ytest)/len(ytest)))
    for nchannel in range(0, train_x.shape[3]):
      xtest = train_x[test].copy()
      xtest[:, :, :, nchannel, :] = np.zeros((train_x[test].shape[0], train_x.shape[1], train_x.shape[2], 1))
      ypred2 = new_pruned_model.predict(xtest)
      loss2 = tf.keras.losses.categorical_crossentropy(ytest, ypred2)
      print("Analyzing channel " +str(nchannel) + ": " + str(np.sum(ypred2*ytest)/len(ytest)))
      SA[ntrain-1][nchannel] = np.sum(abs(loss2 - loss1))

    with open('SA_fold_pruning'+str(ntrain), 'wb') as f:
      pickle.dump(SA[ntrain-1], f)

    ntrain += 1;