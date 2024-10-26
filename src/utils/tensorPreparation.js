import * as tf from '@tensorflow/tfjs';

export function prepareDataTensors(trainData, validationData, testData, targetColumn, featureColumns, targetType) {
  console.log('Preparing data tensors...');

  // Prepare feature tensors
  const trainX = tf.tensor2d(trainData.map(row => featureColumns.map(col => row[col])));
  const validationX = tf.tensor2d(validationData.map(row => featureColumns.map(col => row[col])));
  const testX = tf.tensor2d(testData.map(row => featureColumns.map(col => row[col])));

  // Prepare target tensors
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

  return {
    train: { x: trainX, y: trainY },
    validation: { x: validationX, y: validationY },
    test: { x: testX, y: testY }
  };
}
