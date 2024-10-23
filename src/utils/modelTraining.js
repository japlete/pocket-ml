import * as tf from '@tensorflow/tfjs';

export async function trainModel(trainData, testData, targetColumn, featureColumns) {
  // Prepare the data
  const trainX = tf.tensor2d(trainData.map(row => featureColumns.map(col => row[col])));
  const trainY = tf.tensor2d(trainData.map(row => [row[targetColumn]]));
  const testX = tf.tensor2d(testData.map(row => featureColumns.map(col => row[col])));
  const testY = tf.tensor2d(testData.map(row => [row[targetColumn]]));

  // Create the model
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [featureColumns.length] }));

  // Compile the model
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  // Train the model
  await model.fit(trainX, trainY, {
    epochs: 50,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}`);
      }
    }
  });

  // Evaluate the model
  const trainPredictions = model.predict(trainX);
  const testPredictions = model.predict(testX);

  const trainRMSE = calculateRMSE(trainY, trainPredictions);
  const testRMSE = calculateRMSE(testY, testPredictions);

  return { trainRMSE, testRMSE };
}

function calculateRMSE(actual, predicted) {
  const mse = tf.losses.meanSquaredError(actual, predicted);
  return Math.sqrt(mse.dataSync()[0]);
}
