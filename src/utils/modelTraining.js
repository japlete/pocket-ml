import * as tf from '@tensorflow/tfjs';

export async function trainModel(trainData, validationData, testData, targetColumn, featureColumns, scaler, targetType, classMapping, onProgressUpdate) {
  console.log('Starting model training...');
  console.log('Target type:', targetType);
  console.log('Class mapping:', classMapping);

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

  // Create the model (1 fixed hidden layer for now)
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [featureColumns.length] }));

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
  if (targetType === 'regression') {
    loss = 'meanSquaredError';
    metrics = ['mse'];
  } else if (targetType === 'binary') {
    loss = 'binaryCrossentropy';
    metrics = ['accuracy'];
  } else if (targetType === 'multiclass') {
    loss = 'sparseCategoricalCrossentropy';
    metrics = ['accuracy'];
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
  console.log('Evaluating the model...');
  
  console.log('Train evaluation:');
  const trainEval = model.evaluate(trainX, trainY);
  console.log('Train loss:', trainEval[0].dataSync()[0]);
  console.log('Train metric:', trainEval[1].dataSync()[0]);

  console.log('Validation evaluation:');
  const validationEval = model.evaluate(validationX, validationY);
  console.log('Validation loss:', validationEval[0].dataSync()[0]);
  console.log('Validation metric:', validationEval[1].dataSync()[0]);

  console.log('Test evaluation:');
  const testEval = model.evaluate(testX, testY);
  console.log('Test loss:', testEval[0].dataSync()[0]);
  console.log('Test metric:', testEval[1].dataSync()[0]);

  let results;
  if (targetType === 'regression') {
    results = {
      trainRMSE: Math.sqrt(trainEval[0].dataSync()[0]),
      validationRMSE: Math.sqrt(validationEval[0].dataSync()[0]),
      testRMSE: Math.sqrt(testEval[0].dataSync()[0])
    };
  } else {
    results = {
      trainAccuracy: trainEval[1].dataSync()[0],
      validationAccuracy: validationEval[1].dataSync()[0],
      testAccuracy: testEval[1].dataSync()[0]
    };
  }

  console.log('Final results:', results);

  // Log the final losses
  console.log('Final losses:');
  console.log('Train loss:', history.history.loss[history.history.loss.length - 1]);
  console.log('Validation loss:', history.history.val_loss ? history.history.val_loss[history.history.val_loss.length - 1] : 'N/A');

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

function calculateRMSE(actual, predicted) {
  const mse = tf.losses.meanSquaredError(actual, predicted);
  return Math.sqrt(mse.dataSync()[0]);
}
