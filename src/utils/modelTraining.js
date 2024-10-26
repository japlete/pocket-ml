import * as tf from '@tensorflow/tfjs';
import { calculateMetrics } from './modelMetrics';

// Custom Callback class for logging
class LoggingCallback extends tf.Callback {
  constructor(onProgressUpdate, totalEpochs) {
    super();
    this.onProgressUpdate = onProgressUpdate;
    this.totalEpochs = totalEpochs;
  }

  async onEpochEnd(epoch, logs) {
    const lossValue = logs.loss !== undefined ? 
      (logs.loss instanceof tf.Tensor ? logs.loss.dataSync()[0].toFixed(4) : logs.loss.toFixed(4)) : 
      'N/A';
    const valLossValue = logs.val_loss !== undefined ? 
      (logs.val_loss instanceof tf.Tensor ? logs.val_loss.dataSync()[0].toFixed(4) : logs.val_loss.toFixed(4)) : 
      'N/A';

    console.log(`Epoch ${epoch + 1}: loss = ${lossValue}, val_loss = ${valLossValue}`);
    this.onProgressUpdate(epoch + 1, this.totalEpochs);
  }
}

export async function trainModel(tensorData, targetType, onProgressUpdate, config) {
  console.log('Starting model training...');

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

  // Create the model. Determine first hidden layer size
  const hidden_dim = autoHiddenDim ? 
    2 ** Math.ceil(Math.log2(Math.ceil((trainX.shape[0] / trainX.shape[1]) ** 0.4))) : 
    hiddenDimInput;
  console.log('First hidden layer size:', hidden_dim);

  // Create input layer
  const input = tf.input({shape: [trainX.shape[1]]});

  // Add L1 regularization if specified
  let first_layer_args = {units: hidden_dim, activation: 'gelu_new'};
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
    const dense = tf.layers.dense({units: next_hidden_dim, activation: 'gelu_new'}).apply(prev_layer);
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
  if (targetType === 'regression') {
    output = tf.layers.dense({units: 1}).apply(last_layer);
  } else if (targetType === 'binary') {
    output = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(last_layer);
  } else if (targetType === 'multiclass') {
    output = tf.layers.dense({
      units: numClasses, 
      activation: 'softmax'
    }).apply(last_layer);
  }

  const model = tf.model({inputs: input, outputs: output});

  // Print model summary and backend
  console.log("Current backend:", tf.getBackend());
  console.log('Model summary:');
  model.summary();

  // Compile the model - simplified, only with loss
  const optimizer = tf.train.adam(learningRate);
  let loss;

  if (targetType === 'regression') {
    loss = tf.losses.meanSquaredError;
  } else if (targetType === 'binary') {
    loss = 'binaryCrossentropy';
  } else if (targetType === 'multiclass') {
    loss = 'sparseCategoricalCrossentropy';
  }

  model.compile({ optimizer, loss });
  console.log('Model compiled with loss:', loss);

  // Train the model
  const loggingCallback = new LoggingCallback(onProgressUpdate, epochs);
  const callbacks = [loggingCallback];
  if (earlyStoppingEnabled) {
    const earlyStoppingCallback = tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      patience: 3
    });
    callbacks.push(earlyStoppingCallback);
  }

  const history = await model.fit(trainX, trainY, {
    epochs: epochs,
    batchSize: batchSize,
    validationData: [validationX, validationY],
    callbacks: callbacks
  });

  console.log('Model training completed.');

  // Return only the trained model, without disposing any tensors
  return { model };
}
