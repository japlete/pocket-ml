import * as tf from '@tensorflow/tfjs';
import { trainModel } from './modelTraining';
import { calculateMetrics } from './modelMetrics';
import { prepareDataTensors } from './tensorPreparation';

export class ModelTrainingManager {
  constructor(config) {
    this.config = {
      // Training configuration
      targetType: config.targetType,
      primaryMetric: config.primaryMetric,
      minIterations: config.minIterations,  // Changed from maxIterations
      maxTrainingTime: config.maxTrainingTime,
      numClasses: config.numClasses,
      
      // Model hyperparameters
      learningRate: config.learningRate,
      l1Penalty: config.l1Penalty,
      dropoutRate: config.dropoutRate,
      batchSize: config.batchSize,
      epochs: config.epochs,
      earlyStoppingEnabled: config.earlyStoppingEnabled,
      autoHiddenDim: config.autoHiddenDim,
      hiddenDimInput: config.hiddenDimInput,
      seed: config.seed
    };

    this.bestModel = null;
    this.bestMetrics = null;
    this.currentIteration = 0;
    this.allModels = []; // Store metrics for all trained models
    this.isTraining = false;
    this.startTime = null;
    this.tensorData = null;
  }

  async startTrainingCycle(trainData, validationData, testData, onProgressUpdate) {
    this.isTraining = true;
    this.startTime = Date.now();
    
    // Prepare tensors once at the start
    this.tensorData = prepareDataTensors(
      trainData, 
      validationData, 
      testData, 
      this.config.targetColumn, 
      this.config.featureColumns, 
      this.config.targetType
    );
    
    while (this.shouldContinueTraining()) {
      console.log(`Starting iteration ${this.currentIteration + 1}`);
      
      const { model } = await trainModel(
        this.tensorData,
        this.config.targetType,
        onProgressUpdate,
        this.getCurrentHyperparameters()
      );

      // During training, only compute primary metric, since it's the only one used for model selection
      const trainResults = await this.evaluateModel(model, this.tensorData.train.x, this.tensorData.train.y, false);
      const validationResults = await this.evaluateModel(model, this.tensorData.validation.x, this.tensorData.validation.y, false);

      const results = {
        train: trainResults,
        validation: validationResults
      };

      this.allModels.push({
        iteration: this.currentIteration,
        metrics: results,
        hyperparameters: this.getCurrentHyperparameters()
      });

      this.updateBestModel(model, results);
      
      this.currentIteration++;
      
      if (this.shouldStopTraining()) break;
      
      if (model !== this.bestModel) {
        model.dispose();
      }

      this.updateHyperparameters();
    }
    
    this.isTraining = false;
    
    // Clean up tensors at the end
    if (this.tensorData) {
      Object.values(this.tensorData).forEach(set => {
        Object.values(set).forEach(tensor => tensor.dispose());
      });
      this.tensorData = null;
    }
    
    // Compute all metrics for the best model
    if (this.bestModel) {
      const finalTrainResults = await this.evaluateModel(this.bestModel, this.tensorData.train.x, this.tensorData.train.y, true);
      const finalValidationResults = await this.evaluateModel(this.bestModel, this.tensorData.validation.x, this.tensorData.validation.y, true);
      const finalTestResults = await this.evaluateModel(this.bestModel, this.tensorData.test.x, this.tensorData.test.y, true);

      return {
        bestModel: this.bestModel,
        trainedModels: this.allModels.sort((a, b) => 
          this.compareMetrics(b.metrics.validation, a.metrics.validation)
        ),
        finalMetrics: {
          train: finalTrainResults,
          validation: finalValidationResults,
          test: finalTestResults
        }
      };
    }
  }

  async trainSingleModel(trainData, validationData, onProgressUpdate) {
    // Current training logic from modelTraining.js
  }

  async evaluateModel(model, data, labels, computeAllMetrics = false) {
    const predictions = model.predict(data);
    const loss = model.evaluate(data, labels).dataSync()[0];
    
    const metricsToCompute = computeAllMetrics ? 
      [this.config.primaryMetric, ...this.config.secondaryMetrics] : 
      [this.config.primaryMetric];
    
    const metricResults = calculateMetrics(
      model, 
      data, 
      labels, 
      this.config.targetType, 
      metricsToCompute
    );
    
    predictions.dispose();

    return {
      loss,
      ...metricResults
    };
  }

  updateBestModel(model, results) {
    if (!this.bestModel || this.isBetterModel(results)) {
      this.bestModel = model;
      this.bestMetrics = results;
    }
  }

  isBetterModel(newMetrics) {
    // Compare validation metrics
    const currentValidationMetric = this.bestMetrics[`validation${this.config.primaryMetric.toUpperCase()}`];
    const newValidationMetric = newMetrics[`validation${this.config.primaryMetric.toUpperCase()}`];
    
    // For some metrics, higher is better (accuracy, AUC)
    // For others, lower is better (RMSE, loss)
    const higherIsBetter = !['rmse', 'mse', 'mae', 'mape'].includes(this.config.primaryMetric);
    
    return higherIsBetter ? 
      newValidationMetric > currentValidationMetric :
      newValidationMetric < currentValidationMetric;
  }

  compareMetrics(metricsA, metricsB) {
    const metricA = metricsA[`validation${this.config.primaryMetric.toUpperCase()}`];
    const metricB = metricsB[`validation${this.config.primaryMetric.toUpperCase()}`];
    
    const higherIsBetter = !['rmse', 'mse', 'mae', 'mape'].includes(this.config.primaryMetric);
    return higherIsBetter ? metricA - metricB : metricB - metricA;
  }

  shouldContinueTraining() {
    const timeLimit = this.config.maxTrainingTime * 60 * 1000; // Convert minutes to ms
    const timeElapsed = Date.now() - this.startTime;
    const timeRemaining = timeElapsed < timeLimit;
    const minIterationsCompleted = this.currentIteration >= this.config.minIterations;

    // Continue if we haven't reached min iterations OR if we still have time
    return !minIterationsCompleted || timeRemaining;
  }

  shouldStopTraining() {
    const timeLimit = this.config.maxTrainingTime * 60 * 1000;
    const timeElapsed = Date.now() - this.startTime;
    const timeExceeded = timeElapsed >= timeLimit;
    const minIterationsCompleted = this.currentIteration >= this.config.minIterations;

    // Stop if we've exceeded time limit AND completed minimum iterations
    return timeExceeded && minIterationsCompleted;
  }

  getCurrentHyperparameters() {
    return {
      learningRate: this.config.learningRate,
      l1Penalty: this.config.l1Penalty,
      dropoutRate: this.config.dropoutRate,
      batchSize: this.config.batchSize,
      epochs: this.config.epochs,
      hiddenDim: this.config.autoHiddenDim ? 'auto' : this.config.hiddenDimInput,
      numClasses: this.config.numClasses // Include this in hyperparameters
    };
  }

  updateHyperparameters() {
    // This will be implemented later with the AutoML logic
    // For now, just make small random adjustments to hyperparameters
    console.log('Hyperparameter update will be implemented in future versions');
  }

  // ... other helper methods
}
