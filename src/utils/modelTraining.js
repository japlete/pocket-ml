import * as tf from '@tensorflow/tfjs';

// Custom metric functions
function calculateRMSE(actual, predicted) {
  return tf.sqrt(tf.metrics.mse(actual, predicted)).dataSync()[0];
}

function calculatePrecision(actual, predicted) {
  const roundedPredictions = tf.round(predicted);
  return tf.metrics.precision(actual, roundedPredictions).dataSync()[0];
}

function calculateRecall(actual, predicted) {
  const roundedPredictions = tf.round(predicted);
  return tf.metrics.recall(actual, roundedPredictions).dataSync()[0];
}

function calculateF1Score(actual, predicted) {
  const roundedPredictions = tf.round(predicted);
  const precision = tf.metrics.precision(actual, roundedPredictions).dataSync()[0];
  const recall = tf.metrics.recall(actual, roundedPredictions).dataSync()[0];
  return (2 * precision * recall) / (precision + recall);
}

function calculateROCAUC(actual, predicted) {
  return tf.tidy(() => {
    const thresholds = tf.linspace(0, 1, 100);
    const thresholdsArray = thresholds.arraySync();
    
    // Convert actual to boolean tensor
    const actualBool = actual.greater(0.5);
    const actualArray = actualBool.arraySync();
    const predictedArray = predicted.arraySync();
    
    const tpr = [];
    const fpr = [];

    thresholdsArray.forEach((threshold, index) => {
      let tp = 0, fn = 0, fp = 0, tn = 0;
      
      for (let i = 0; i < actualArray.length; i++) {
        if (actualArray[i]) {
          if (predictedArray[i] >= threshold) {
            tp++;
          } else {
            fn++;
          }
        } else {
          if (predictedArray[i] >= threshold) {
            fp++;
          } else {
            tn++;
          }
        }
      }

      const tprValue = tp / (tp + fn) || 0;
      const fprValue = fp / (fp + tn) || 0;
      tpr.push(tprValue);
      fpr.push(fprValue);

    });

    // Sort FPR and TPR arrays together, with FPR in ascending order
    const sortedPairs = fpr.map((f, index) => [f, tpr[index]]).sort((a, b) => a[0] - b[0]);
    const sortedFPR = sortedPairs.map(pair => pair[0]);
    const sortedTPR = sortedPairs.map(pair => pair[1]);

    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < sortedFPR.length; i++) {
      auc += (sortedFPR[i] - sortedFPR[i - 1]) * (sortedTPR[i] + sortedTPR[i - 1]) / 2;
    }

    return auc;
  });
}

function calculatePRAUC(actual, predicted) {
  return tf.tidy(() => {
    const thresholds = tf.linspace(0, 1, 100);
    const thresholdsArray = thresholds.arraySync();
    
    // Convert actual to boolean tensor
    const actualBool = actual.greater(0.5);
    const actualArray = actualBool.arraySync();
    const predictedArray = predicted.arraySync();
    
    const precision = [];
    const recall = [];

    thresholdsArray.forEach((threshold, index) => {
      let tp = 0, fn = 0, fp = 0;
      
      for (let i = 0; i < actualArray.length; i++) {
        if (actualArray[i]) {
          if (predictedArray[i] >= threshold) {
            tp++;
          } else {
            fn++;
          }
        } else {
          if (predictedArray[i] >= threshold) {
            fp++;
          }
        }
      }

      const precisionValue = tp / (tp + fp) || 0;
      const recallValue = tp / (tp + fn) || 0;
      precision.push(precisionValue);
      recall.push(recallValue);

    });

    // Sort Recall and Precision arrays together, with Recall in descending order
    const sortedPairs = recall.map((r, index) => [r, precision[index]]).sort((a, b) => b[0] - a[0]);
    const sortedRecall = sortedPairs.map(pair => pair[0]);
    const sortedPrecision = sortedPairs.map(pair => pair[1]);

    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < sortedRecall.length; i++) {
      auc += (sortedRecall[i - 1] - sortedRecall[i]) * (sortedPrecision[i] + sortedPrecision[i - 1]) / 2;
    }

    return auc;
  });
}

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

export async function trainModel(trainData, validationData, testData, targetColumn, featureColumns, targetType, classMapping, onProgressUpdate, 
  primaryMetric, secondaryMetrics, seed, learning_rate, l1_penalty, dropout_rate, batchSize, max_epochs, earlyStoppingEnabled, autoHiddenDim, hiddenDimInput) {
  console.log('Starting model training...');
  console.log('Target type:', targetType);
  console.log('Class mapping:', classMapping);
  console.log('Primary metric:', primaryMetric);
  console.log('Secondary metrics:', secondaryMetrics);

  // Prepare the data
  const trainX = tf.tensor2d(trainData.map(row => featureColumns.map(col => row[col])));
  const validationX = tf.tensor2d(validationData.map(row => featureColumns.map(col => row[col])));
  const testX = tf.tensor2d(testData.map(row => featureColumns.map(col => row[col])));

  let trainY, validationY, testY;
  if (targetType === 'regression') {
    trainY = tf.tensor2d(trainData.map(row => [row[targetColumn]]));
    validationY = tf.tensor2d(validationData.map(row => [row[targetColumn]]));
    testY = tf.tensor2d(testData.map(row => [row[targetColumn]]));
  } else {
    trainY = tf.tensor1d(trainData.map(row => row[targetColumn]));
    validationY = tf.tensor1d(validationData.map(row => row[targetColumn]));
    testY = tf.tensor1d(testData.map(row => row[targetColumn]));
  }

  console.log('Data shapes:');
  console.log('trainX:', trainX.shape);
  console.log('trainY:', trainY.shape);
  console.log('validationX:', validationX.shape);
  console.log('validationY:', validationY.shape);
  console.log('testX:', testX.shape);
  console.log('testY:', testY.shape);

  // Create the model. Determine first hidden layer size
  const hidden_dim = autoHiddenDim ? 2 ** Math.ceil(Math.log2(Math.ceil((trainX.shape[0] / trainX.shape[1]) ** 0.4))) : hiddenDimInput;
  console.log('First hidden layer size:', hidden_dim);
  const input = tf.input({shape: [featureColumns.length]});
  let first_layer_args = {units: hidden_dim, activation: 'gelu_new'};
  // Add L1 regularization if specified
  if (l1_penalty) {
    first_layer_args = {
      ...first_layer_args,
      kernelRegularizer: tf.regularizers.l1({l1: l1_penalty})
    };
  }
  const dense1 = tf.layers.dense(first_layer_args).apply(input);
  let residual_concat = [input, dense1];
  let next_hidden_dim = Math.ceil(hidden_dim / 4);
  let prev_layer = dense1;
  // Add more layers with decreasing size up to a minimum of 2 units
  while (next_hidden_dim >= 2) {
    // Add dropout after each dense layer if specified
    if (dropout_rate) {
      const dropout = tf.layers.dropout({rate: dropout_rate, seed: seed+next_hidden_dim}).apply(prev_layer);
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
  if (dropout_rate) {
    const dropout = tf.layers.dropout({rate: dropout_rate, seed: seed+next_hidden_dim}).apply(last_layer);
    last_layer = dropout;
  }

  // Add the output layer based on the target type
  let output;
  if (targetType === 'regression') {
    output = tf.layers.dense({units: 1}).apply(last_layer);
  } else if (targetType === 'binary') {
    output = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(last_layer);
  } else if (targetType === 'multiclass') {
    const numClasses = Object.keys(classMapping).length;
    output = tf.layers.dense({units: numClasses, activation: 'softmax'}).apply(last_layer);
  }

  const model = tf.model({inputs: input, outputs: output});

  // Print model summary and backend
  console.log("Current backend:", tf.getBackend());
  console.log('Model summary:');
  model.summary();

  // Compile the model
  const optimizer = tf.train.adam(learning_rate);
  let loss, metrics;
  const allMetrics = [primaryMetric, ...secondaryMetrics];

  if (targetType === 'regression') {
    loss = tf.losses.meanSquaredError;
    metrics = allMetrics.filter(metric => {
      return ['mae', 'mape', 'mse', 'r2'].includes(metric);
    }).map(metric => {
      switch (metric) {
        case 'mae': return tf.metrics.meanAbsoluteError;
        case 'mape': return tf.metrics.meanAbsolutePercentageError;
        case 'mse': return tf.metrics.meanSquaredError;
        case 'r2': return tf.metrics.r2Score;
        default: return null;
      }
    }).filter(Boolean);
  } else if (targetType === 'binary') {
    loss = 'binaryCrossentropy';
    metrics = allMetrics.filter(metric => {
      return ['accuracy'].includes(metric);
    }).map(metric => {
      switch (metric) {
        case 'accuracy': return tf.metrics.binaryAccuracy;
        default: return null;
      }
    }).filter(Boolean);
  } else if (targetType === 'multiclass') {
    loss = 'sparseCategoricalCrossentropy';
    metrics = allMetrics.filter(metric => {
      return ['accuracy'].includes(metric);
    }).map(metric => {
      switch (metric) {
        case 'accuracy': return tf.metrics.sparseCategoricalAccuracy;
        default: return null;
      }
    }).filter(Boolean);
  }
  model.compile({ optimizer, loss, metrics });
  console.log('Model compiled with loss:', loss, 'and metrics:', metrics);

  // Set up callbacks during training
  const loggingCallback = new LoggingCallback(onProgressUpdate, max_epochs);
  const callbacks = [loggingCallback];
  if (earlyStoppingEnabled) {
    const earlyStoppingCallback = tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      patience: 3
    });
    callbacks.push(earlyStoppingCallback);
  }

  // Train the model
  const history = await model.fit(trainX, trainY, {
    epochs: max_epochs,
    batchSize: batchSize,
    validationData: [validationX, validationY],
    callbacks: callbacks
  });

  console.log('Model training completed.');

  // Evaluate the model
  const evalResults = {
    train: model.evaluate(trainX, trainY),
    validation: model.evaluate(validationX, validationY),
    test: model.evaluate(testX, testY)
  };

  // Gather the metrics
  const results = {};
  ['train', 'validation', 'test'].forEach((set) => {
    const evalTensors = evalResults[set];
    const evalValues = Array.isArray(evalTensors) ? evalTensors.map(tensor => tensor.dataSync()[0]) : [evalTensors.dataSync()[0]];
    const [loss, ...metricValues] = evalValues;
    results[`${set}Loss`] = loss;

    const str_metrics = metrics.map(metric => {
      switch (metric) {
        case tf.metrics.meanAbsoluteError: return 'mae';
        case tf.metrics.meanAbsolutePercentageError: return 'mape';
        case tf.metrics.meanSquaredError: return 'mse';
        case tf.metrics.r2Score: return 'r2';
        case tf.metrics.binaryAccuracy: return 'accuracy';
        case tf.metrics.sparseCategoricalAccuracy: return 'accuracy';
        default: return null;
      }
    });
    str_metrics.forEach((metric, index) => {
      results[`${set}${metric.toUpperCase()}`] = metricValues[index];
    });

    // Compute custom metrics (not included in TensorFlow.js)
    const predicted = model.predict(eval(set + 'X'));
    const actual = eval(set + 'Y');

    if (allMetrics.includes('rmse') && !results[`${set}RMSE`]) {
      results[`${set}RMSE`] = calculateRMSE(actual, predicted);
    }

    if (targetType === 'binary') {
      if (allMetrics.includes('precision')) {
        results[`${set}PRECISION`] = calculatePrecision(actual, predicted);
      }
      if (allMetrics.includes('recall')) {
        results[`${set}RECALL`] = calculateRecall(actual, predicted);
      }
      if (allMetrics.includes('f1')) {
        results[`${set}F1`] = calculateF1Score(actual, predicted);
      }
      if (allMetrics.includes('roc_auc')) {
        results[`${set}ROC_AUC`] = calculateROCAUC(actual, predicted);
      }
      if (allMetrics.includes('pr_auc')) {
        results[`${set}PR_AUC`] = calculatePRAUC(actual, predicted);
      }
    }

    predicted.dispose(); // Dispose of the predicted tensor
  });

  // Clean up memory
  trainX.dispose();
  trainY.dispose();
  validationX.dispose();
  validationY.dispose();
  testX.dispose();
  testY.dispose();
  model.dispose();

  return results;
}
