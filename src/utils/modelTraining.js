import * as tf from '@tensorflow/tfjs';
import seedrandom from 'seedrandom';

// Custom Callback class for logging
class LoggingCallback extends tf.Callback {
  constructor(onProgressUpdate, totalEpochs) {
    super();
    this.onProgressUpdate = onProgressUpdate;
    this.totalEpochs = totalEpochs;
  }

  async onEpochEnd(epoch, logs) {
    // Wait for next frame to keep UI responsive
    await tf.nextFrame();
    
    const lossValue = logs.loss !== undefined ? 
      (logs.loss instanceof tf.Tensor ? logs.loss.dataSync()[0].toFixed(4) : logs.loss.toFixed(4)) : 
      'N/A';
    const valLossValue = logs.val_loss !== undefined ? 
      (logs.val_loss instanceof tf.Tensor ? logs.val_loss.dataSync()[0].toFixed(4) : logs.val_loss.toFixed(4)) : 
      'N/A';

    console.log(`Epoch ${epoch + 1}: loss = ${lossValue}, val_loss = ${valLossValue}`);
    this.onProgressUpdate(epoch + 1, this.totalEpochs);
  }

  async onBatchEnd(batch, logs) {
    // Add frame yielding every few batches to prevent UI freezing
    if (batch % 5 === 0) { // Adjust this number based on performance
      await tf.nextFrame();
    }
  }
}

export async function trainModel(tensorData, targetType, onProgressUpdate, config) {
  const { x: trainX, y: trainY } = tensorData.train;
  const { x: validationX, y: validationY } = tensorData.validation;

  // Extract all configuration parameters
  const {
    autoHiddenDim,
    hiddenDimInput,
    l1Penalty,
    dropoutRate,
    seed,
    learningRate,
    epochs,
    batchSize,
    earlyStoppingEnabled,
    numClasses
  } = config;

  // Set the random seed for the epoch shuffling
  seedrandom(seed, { global: true });

  // Create the model. Determine first hidden layer size
  const hidden_dim = autoHiddenDim ? 
    Math.min(512, 2 ** Math.ceil(Math.log2(Math.ceil((trainX.shape[0] / trainX.shape[1]) ** 0.4)))) : 
    Math.min(512, hiddenDimInput);
  console.log('First hidden layer size:', hidden_dim);

  // Create input layer
  const input = tf.input({shape: [trainX.shape[1]]});

  // Add L1 regularization if specified
  let first_layer_args = {
    units: hidden_dim, 
    activation: 'relu', 
    kernelInitializer: tf.initializers.glorotNormal({seed: seed+1})
  };
  if (l1Penalty) {
    first_layer_args = {
      ...first_layer_args,
      kernelRegularizer: tf.regularizers.l1({l1: l1Penalty})
    };
  }

  const dense1 = tf.layers.dense(first_layer_args).apply(input);
  let residual_concat = [input, dense1];
  let next_hidden_dim = Math.ceil(hidden_dim / 4);
  let prev_layer = dense1;

  // Add more layers with decreasing size up to a minimum of 2 units
  while (next_hidden_dim >= 2) {
    // Add dropout after each dense layer if specified
    if (dropoutRate) {
      const dropout = tf.layers.dropout({rate: dropoutRate, seed: seed+next_hidden_dim}).apply(prev_layer);
      prev_layer = dropout;
    }
    const dense = tf.layers.dense({
      units: next_hidden_dim, 
      activation: 'relu', 
      kernelInitializer: tf.initializers.glorotNormal({seed: seed+residual_concat.length})
    }).apply(prev_layer);
    prev_layer = dense;
    residual_concat.push(dense);
    console.log('Added hidden layer of size:', next_hidden_dim);
    next_hidden_dim = Math.ceil(next_hidden_dim / 4);
  }
  
  // Add skip connections by concatenating the outputs of all dense layers
  const concatenated = tf.layers.concatenate().apply(residual_concat);
  let last_layer = concatenated;
  if (dropoutRate) {
    const dropout = tf.layers.dropout({rate: dropoutRate, seed: seed+next_hidden_dim}).apply(last_layer);
    last_layer = dropout;
  }

  // Add the output layer based on the target type
  let output;
  const outputLayerConfig = {
    kernelInitializer: tf.initializers.glorotNormal({seed: seed + residual_concat.length}),
  };
  if (targetType === 'regression') {
    output = tf.layers.dense({...outputLayerConfig, units: 1}).apply(last_layer);
  } else if (targetType === 'binary') {
    output = tf.layers.dense({...outputLayerConfig, units: 1, activation: 'sigmoid'}).apply(last_layer);
  } else if (targetType === 'multiclass') {
    output = tf.layers.dense({
      ...outputLayerConfig,
      units: numClasses, 
      activation: 'softmax'
    }).apply(last_layer);
  }

  const model = tf.model({inputs: input, outputs: output});

  // Print number of parameters
  const numParams = model.countParams();
  console.log('Number of parameters:', numParams);

  // Compile the model
  const optimizer = tf.train.adam(learningRate);
  let loss;

  if (targetType === 'regression') {
    loss = 'meanSquaredError';
  } else if (targetType === 'binary') {
    loss = 'binaryCrossentropy';
  } else if (targetType === 'multiclass') {
    loss = 'sparseCategoricalCrossentropy';
  }

  model.compile({ optimizer, loss });
  console.log('Model compiled with loss:', loss);

  // Print model backend
  console.log("Current backend:", tf.getBackend());

  // Train the model
  const callbacks = [new LoggingCallback(onProgressUpdate, epochs)];
  
  if (earlyStoppingEnabled) {
    callbacks.push(tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      patience: 3,
      verbose: 1
    }));
  }

  // Model training
  const history = await model.fit(trainX, trainY, {
    epochs: epochs,
    batchSize: batchSize,
    validationData: [validationX, validationY],
    callbacks: callbacks
  });

  console.log('Model training completed.');

  return { model, history };
}
