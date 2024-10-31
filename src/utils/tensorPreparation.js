import * as tf from '@tensorflow/tfjs';

export function prepareDataTensors(trainData, validationData, testData, targetColumn, featureColumns, targetType) {
  console.log('Preparing data tensors...');

  // Filter out 'split' from feature columns if it's present
  const modelFeatures = featureColumns.filter(col => col !== 'split');
  console.log('Model features:', modelFeatures);

  // Prepare feature tensors with explicit shape
  const trainX = tf.tensor2d(trainData.map(row => modelFeatures.map(col => row[col])), 
    [trainData.length, modelFeatures.length]);
  const validationX = tf.tensor2d(validationData.map(row => modelFeatures.map(col => row[col])),
    [validationData.length, modelFeatures.length]);
  const testX = tf.tensor2d(testData.map(row => modelFeatures.map(col => row[col])),
    [testData.length, modelFeatures.length]);

  // Prepare target tensors
  let trainY, validationY, testY;
  if (targetType === 'regression') {
    trainY = tf.tensor2d(trainData.map(row => [row[targetColumn]]), [trainData.length, 1]);
    validationY = tf.tensor2d(validationData.map(row => [row[targetColumn]]), [validationData.length, 1]);
    testY = tf.tensor2d(testData.map(row => [row[targetColumn]]), [testData.length, 1]);
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

  return {
    train: { x: trainX, y: trainY },
    validation: { x: validationX, y: validationY },
    test: { x: testX, y: testY }
  };
}
