import * as tf from '@tensorflow/tfjs';
import { result } from 'lodash';

export function calculateMetrics(predictions, labels, targetType, requestedMetrics) {
  const results = {};
  
  // Calculate all requested metrics
  requestedMetrics.forEach(metric => {
    switch(metric.toLowerCase()) {
      case 'mse':
        results.MSE = tf.metrics.mse(labels, predictions).dataSync()[0];
        break;
      case 'rmse':
        results.RMSE = tf.sqrt(tf.metrics.mse(labels, predictions)).dataSync()[0]
        break;
      case 'mae':
        results.MAE = tf.metrics.meanAbsoluteError(labels, predictions).dataSync()[0];
        break;
      case 'mape':
        results.MAPE = tf.metrics.MAPE(labels, predictions).dataSync()[0];
        break;
      case 'r2':
        results.R2 = tf.metrics.r2Score(labels, predictions).dataSync()[0];
        break;
      case 'accuracy':
        results.ACCURACY = calculateAccuracy(labels, predictions, targetType);
        break;
      case 'precision':
        results.PRECISION = calculatePrecision(labels, predictions);
        break;
      case 'recall':
        results.RECALL = calculateRecall(labels, predictions);
        break;
      case 'f1':
        results.F1 = calculateF1Score(labels, predictions);
        break;
      case 'roc_auc':
        results.ROC_AUC = calculateROCAUC(labels, predictions);
        break;
      case 'pr_auc':
        results.PR_AUC = calculatePRAUC(labels, predictions);
        break;
      default:
        console.warn(`Unrecognized metric: ${metric}`);
    }
  });
  
  console.log('Computed metrics:', requestedMetrics);
  console.log('Results:', results);
  
  return results;
}

export function calculateAccuracy(actual, predicted, targetType) {
  if (targetType === 'binary') {
    return tf.metrics.binaryAccuracy(actual, predicted).dataSync()[0];
  } else if (targetType === 'multiclass') {
    return tf.metrics.sparseCategoricalAccuracy(actual, predicted).dataSync()[0];
  }
  return null;
}

export function calculatePrecision(actual, predicted) {
  const roundedPredictions = tf.round(predicted);
  return tf.metrics.precision(actual, roundedPredictions).dataSync()[0];
}

export function calculateRecall(actual, predicted) {
  const roundedPredictions = tf.round(predicted);
  return tf.metrics.recall(actual, roundedPredictions).dataSync()[0];
}

export function calculateF1Score(actual, predicted) {
  const roundedPredictions = tf.round(predicted);
  const precision = tf.metrics.precision(actual, roundedPredictions).dataSync()[0];
  const recall = tf.metrics.recall(actual, roundedPredictions).dataSync()[0];
  return (2 * precision * recall) / (precision + recall);
}

export function calculateROCAUC(actual, predicted) {
  return tf.tidy(() => {
    const thresholds = tf.linspace(0, 1, 100);
    const thresholdsArray = thresholds.arraySync();
    
    // Convert actual to boolean tensor
    const actualBool = actual.greater(0.5);
    const actualArray = actualBool.arraySync();
    const predictedArray = predicted.arraySync();
    
    const tpr = [];
    const fpr = [];

    thresholdsArray.forEach(threshold => {
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

export function calculatePRAUC(actual, predicted) {
  return tf.tidy(() => {
    const thresholds = tf.linspace(0, 1, 100);
    const thresholdsArray = thresholds.arraySync();
    
    // Convert actual to boolean tensor
    const actualBool = actual.greater(0.5);
    const actualArray = actualBool.arraySync();
    const predictedArray = predicted.arraySync();
    
    const precision = [];
    const recall = [];

    thresholdsArray.forEach(threshold => {
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
