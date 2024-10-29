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
      secondaryMetrics: config.secondaryMetrics || [],
      minIterations: config.minIterations,
      maxTrainingTime: config.maxTrainingTime,
      numClasses: config.numClasses,
      
      // Add these two important configurations
      targetColumn: config.targetColumn,
      featureColumns: config.featureColumns,
      
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
    this.abortController = null;
  }

  async startTrainingCycle(trainData, validationData, testData, onProgressUpdate) {
    this.isTraining = true;
    this.startTime = Date.now();
    this.abortController = new AbortController();
    
    // Prepare tensors once at the start with the correct configuration
    this.tensorData = prepareDataTensors(
      trainData, 
      validationData, 
      testData, 
      this.config.targetColumn,
      this.config.featureColumns,
      this.config.targetType
    );
    
    const trainNextIteration = async () => {
      if (!this.shouldContinueTraining()) {
        await this.finishTraining();
        return;
      }

      console.log(`Starting hyper-tuning iteration ${this.currentIteration + 1}`);
      
      try {
        const { model } = await trainModel(
          this.tensorData,
          this.config.targetType,
          onProgressUpdate,
          this.getCurrentHyperparameters()
        );

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
        
        if (model !== this.bestModel) {
          model.dispose();
        }

        this.updateHyperparameters();
        
        // Continue to next iteration
        setTimeout(trainNextIteration, 0);
      } catch (error) {
        console.error('Training error:', error);
        await this.finishTraining();
      }
    };

    // Start the first iteration
    await trainNextIteration();
    
    // Wait for training to complete
    return new Promise((resolve) => {
      const checkCompletion = () => {
        if (!this.isTraining) {
          resolve({
            bestModel: this.bestModel,
            trainedModels: this.allModels.sort((a, b) => 
              this.compareMetrics(b.metrics.validation, a.metrics.validation)
            ),
            finalMetrics: this.finalMetrics,
            completedIterations: this.currentIteration
          });
        } else {
          setTimeout(checkCompletion, 100);
        }
      };
      checkCompletion();
    });
  }

  stopTraining() {
    if (this.abortController) {
      this.abortController.abort();
    }
  }

  async finishTraining() {
    if (!this.isTraining) return; // Prevent multiple calls
    
    console.log('Finishing training and computing final metrics...');
    
    // Clean up tensors
    if (this.tensorData) {
      // Compute all metrics for the best model before cleanup
      if (this.bestModel) {
        const finalTrainResults = await this.evaluateModel(this.bestModel, this.tensorData.train.x, this.tensorData.train.y, true);
        const finalValidationResults = await this.evaluateModel(this.bestModel, this.tensorData.validation.x, this.tensorData.validation.y, true);
        const finalTestResults = await this.evaluateModel(this.bestModel, this.tensorData.test.x, this.tensorData.test.y, true);
        
        this.finalMetrics = {
          train: finalTrainResults,
          validation: finalValidationResults,
          test: finalTestResults
        };
      }

      Object.values(this.tensorData).forEach(set => {
        Object.values(set).forEach(tensor => tensor.dispose());
      });
      this.tensorData = null;
    }

    this.isTraining = false;
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
    const timeExceeded = timeElapsed >= timeLimit;
    const minIterationsCompleted = this.currentIteration >= this.config.minIterations;

    // Continue training if either:
    // 1. We haven't completed minimum iterations OR
    // 2. We haven't exceeded the time limit (if a time limit is set)
    // AND
    // 3. We haven't been aborted
    return (!minIterationsCompleted || !timeExceeded) && 
           !this.abortController.signal.aborted;
  }

  getCurrentHyperparameters() {
    return {
      learningRate: this.config.learningRate,
      l1Penalty: this.config.l1Penalty,
      dropoutRate: this.config.dropoutRate,
      batchSize: this.config.batchSize,
      epochs: this.config.epochs,
      earlyStoppingEnabled: this.config.earlyStoppingEnabled,
      autoHiddenDim: this.config.autoHiddenDim,
      hiddenDimInput: this.config.hiddenDimInput,
      numClasses: this.config.numClasses,
      seed: this.config.seed
    };
  }

  updateHyperparameters() {
    // This will be implemented later with the AutoML logic
    // For now, just make small random adjustments to hyperparameters
    console.log('Hyperparameter update will be implemented in future versions');
  }

  // ... other helper methods
}
