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

      if (index % 10 === 0) {  // Log every 10th value to avoid console clutter
        console.log(`ROC AUC - Threshold: ${threshold.toFixed(2)}, TP: ${tp}, FN: ${fn}, FP: ${fp}, TN: ${tn}, TPR: ${tprValue.toFixed(4)}, FPR: ${fprValue.toFixed(4)}`);
      }
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

    console.log(`ROC AUC final value: ${auc}`);
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

      if (index % 10 === 0) {  // Log every 10th value to avoid console clutter
        console.log(`PR AUC - Threshold: ${threshold.toFixed(2)}, TP: ${tp}, FP: ${fp}, FN: ${fn}, Precision: ${precisionValue.toFixed(4)}, Recall: ${recallValue.toFixed(4)}`);
      }
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

    console.log(`PR AUC final value: ${auc}`);
    return auc;
  });
}

export async function trainModel(trainData, validationData, testData, targetColumn, featureColumns, scaler, targetType, classMapping, onProgressUpdate, primaryMetric, secondaryMetrics = []) {
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

  // Create the model
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [featureColumns.length] }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));

  // Add the output layer based on the target type
  if (targetType === 'regression') {
    model.add(tf.layers.dense({ units: 1 }));
  } else if (targetType === 'binary') {
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  } else if (targetType === 'multiclass') {
    const numClasses = Object.keys(classMapping).length;
    model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
  }

  console.log('Model summary:');
  model.summary();

  // Compile the model
  const optimizer = tf.train.adam();
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

  // Train the model
  const totalEpochs = 50;
  const history = await model.fit(trainX, trainY, {
    epochs: totalEpochs,
    validationData: [validationX, validationY],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss ? logs.val_loss.toFixed(4) : 'N/A'}`);
        onProgressUpdate(epoch + 1, totalEpochs);
      }
    }
  });

  console.log('Model training completed.');

  // Evaluate the model
  const evalResults = {
    train: model.evaluate(trainX, trainY),
    validation: model.evaluate(validationX, validationY),
    test: model.evaluate(testX, testY)
  };

  const results = {};
  ['train', 'validation', 'test'].forEach((set) => {
    const evalTensors = evalResults[set];
    const evalValues = Array.isArray(evalTensors) ? evalTensors.map(tensor => tensor.dataSync()[0]) : [evalTensors.dataSync()[0]];
    const [loss, ...metricValues] = evalValues;
    results[`${set}Loss`] = loss;

    allMetrics.forEach((metric, index) => {
      if (metric === 'accuracy') {
        results[`${set}${metric.toUpperCase()}`] = metricValues[index];
      }
    });

    // Compute custom metrics if requested
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
