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
    this.lastTrainedEpochs = 0;
    this.bestModelIteration = null;
  }

  async startTrainingCycle(trainData, validationData, testData, onProgressUpdate, onIterationComplete) {
    this.isTraining = true;
    this.startTime = Date.now();
    this.abortController = new AbortController();
    
    this.tensorData = prepareDataTensors(
      trainData, validationData, testData, 
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
        const { model, history } = await trainModel(
          this.tensorData,
          this.config.targetType,
          onProgressUpdate,
          this.getCurrentHyperparameters()
        );

        this.lastTrainedEpochs = history.epoch.length;

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
        
        // Call the callback with current models after each iteration
        if (onIterationComplete) {
          onIterationComplete(this.allModels, this.bestModelIteration);
        }
        
        this.currentIteration++;
        
        if (model !== this.bestModel) {
          model.dispose();
        }

        this.updateHyperparameters();
        
        setTimeout(trainNextIteration, 0);
      } catch (error) {
        console.error('Training error:', error);
        await this.finishTraining();
      }
    };

    await trainNextIteration();
    
    return new Promise((resolve) => {
      const checkCompletion = () => {
        if (!this.isTraining) {
          resolve({
            bestModel: this.bestModel,
            trainedModels: this.allModels,
            finalMetrics: this.finalMetrics,
            bestModelIteration: this.bestModelIteration,
            completedIterations: this.currentIteration
          });
        } else {
          setTimeout(checkCompletion, 100);
        }
      };
      checkCompletion();
    });
  };

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
        // Always compute all metrics in final evaluation
        const allMetrics = [this.config.primaryMetric, ...this.config.secondaryMetrics];
        console.log('Computing final metrics:', allMetrics);
        
        const finalTrainResults = await this.evaluateModel(this.bestModel, this.tensorData.train.x, this.tensorData.train.y, true);
        const finalValidationResults = await this.evaluateModel(this.bestModel, this.tensorData.validation.x, this.tensorData.validation.y, true);
        const finalTestResults = await this.evaluateModel(this.bestModel, this.tensorData.test.x, this.tensorData.test.y, true);
        
        this.finalMetrics = {
          train: finalTrainResults,
          validation: finalValidationResults,
          test: finalTestResults
        };
        
        console.log('Final metrics computed:', this.finalMetrics);
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
    const predictions = model.predict(data, {batchSize: 1024}).squeeze();
    const loss = model.evaluate(data, labels, {batchSize: 1024}).dataSync()[0];
    
    const metricsToCompute = computeAllMetrics ? 
      [this.config.primaryMetric, ...this.config.secondaryMetrics] : 
      [this.config.primaryMetric];
    
    const metricResults = calculateMetrics(
      predictions, 
      labels.squeeze(), 
      this.config.targetType, 
      metricsToCompute
    );
    
    predictions.dispose();

    const finalResults = {
      loss,
      ...metricResults
    };

    return finalResults;
  }

  updateBestModel(model, results) {
    if (!this.bestModel || this.isBetterModel(results)) {
      this.bestModel = model;
      this.bestMetrics = results;
      this.bestModelIteration = this.currentIteration + 1; // Store the iteration when we find a better model
    }
  }

  isBetterModel(newMetrics) {
    console.log('Comparing metrics:');  // Debug log
    console.log('Best metrics so far:', this.bestMetrics);  // Debug log
    console.log('New metrics:', newMetrics);  // Debug log
    
    // Compare validation metrics - capitalize the metric name
    const metricKey = this.config.primaryMetric.toUpperCase();
    const currentValidationMetric = this.bestMetrics?.validation[metricKey];
    const newValidationMetric = newMetrics?.validation[metricKey];
    
    console.log('Primary metric:', metricKey);  // Debug log
    console.log('Current validation metric:', currentValidationMetric);  // Debug log
    console.log('New validation metric:', newValidationMetric);  // Debug log
    
    // If we don't have valid metrics to compare, treat new model as better
    if (this.bestMetrics === null) {
      console.log(`Model comparison (${metricKey}):`);
      console.log('  Current best validation: none (first model)');
      console.log(`  New model validation: ${newValidationMetric}`);
      console.log('  Better? true (first model)');
      return true;
    }
    
    // For some metrics, higher is better (accuracy, AUC)
    // For others, lower is better (RMSE, loss)
    const higherIsBetter = !['rmse', 'mse', 'mae', 'mape'].includes(this.config.primaryMetric.toLowerCase());
    
    const isBetter = higherIsBetter ? 
      newValidationMetric > currentValidationMetric :
      newValidationMetric < currentValidationMetric;

    console.log(`Model comparison (${metricKey}):`);
    console.log(`  Current best validation: ${currentValidationMetric}`);
    console.log(`  New model validation: ${newValidationMetric}`);
    console.log(`  Better? ${isBetter}`);
    
    return isBetter;
  }

  compareMetrics(metricsA, metricsB) {
    const metricKey = this.config.primaryMetric.toUpperCase();
    const metricA = metricsA[metricKey];
    const metricB = metricsB[metricKey];
    
    const higherIsBetter = !['rmse', 'mse', 'mae', 'mape'].includes(this.config.primaryMetric.toLowerCase());
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
    // If the early stopping occurred early, halve the learning rate
    if (this.lastTrainedEpochs < this.config.epochs * 0.4) {
      console.log(`Early stopping occurred at ${this.lastTrainedEpochs}/${this.config.epochs} epochs. Reducing learning rate.`);
      this.config.learningRate /= 2;
      console.log(`New learning rate: ${this.config.learningRate}`);
    }

    // Get the latest model's metrics
    const latestModel = this.allModels[this.allModels.length - 1];
    if (!latestModel) return;

    const metricKey = this.config.primaryMetric.toUpperCase();
    const trainMetric = latestModel.metrics.train[metricKey];
    const valMetric = latestModel.metrics.validation[metricKey];
    const valSetSize = this.tensorData.validation.x.shape[0];
    
    // Calculate the overfitting threshold factor
    const thresholdFactor = 1 - 0.25 / Math.pow(valSetSize, 0.2);
    
    console.log('Overfitting detection metrics:');
    console.log(`  Training ${metricKey}: ${trainMetric}`);
    console.log(`  Validation ${metricKey}: ${valMetric}`);
    console.log(`  Threshold factor: ${thresholdFactor}`);
    
    // Check for overfitting based on metric type
    const isHigherBetterMetric = !['rmse', 'mse', 'mae', 'mape'].includes(this.config.primaryMetric.toLowerCase());
    console.log('isHigherBetter: ', isHigherBetterMetric)
    const hasOverfitting = isHigherBetterMetric ? 
      (valMetric < trainMetric * thresholdFactor - 0.01) :
      (valMetric > trainMetric / thresholdFactor);

    console.log(`  Overfitting detected: ${hasOverfitting}`);

    if (hasOverfitting) {
      console.log('Overfitting detected. Adjusting regularization...');
      
      // Increase dropout rate
      this.config.dropoutRate = Math.min(0.5, (this.config.dropoutRate || 0) + 0.1);
      console.log(`Increased dropout rate to ${this.config.dropoutRate}`);
      
      // Handle L1 regularization changes
      if (this.config.dropoutRate >= 0.3 && this.config.l1Penalty < 0.03) {
        if (!this.config.l1Penalty) {
          this.config.l1Penalty = 0.001;
          console.log('Initialized L1 penalty to 0.001');
        } else {
          this.config.l1Penalty *= 3;
          console.log(`Increased L1 penalty to ${this.config.l1Penalty}`);
        }
      }
    } else {
      // If no overfitting, increase hidden dimension
      if (this.config.autoHiddenDim) {
        const trainShape = this.tensorData.train.x.shape;
        this.config.hiddenDimInput = Math.min(512, 
          2 ** Math.ceil(Math.log2(Math.ceil((trainShape[0] / trainShape[1]) ** 0.4)))
        );
        this.config.autoHiddenDim = false;
      }
      // Double the hidden dimension, but cap at 512
      this.config.hiddenDimInput = Math.min(512, this.config.hiddenDimInput * 2);
      console.log(`Increased hidden dimension to ${this.config.hiddenDimInput}`);
    }

    // Log the current state of hyperparameters
    console.log('Current hyperparameters:', {
      learningRate: this.config.learningRate,
      dropoutRate: this.config.dropoutRate,
      l1Penalty: this.config.l1Penalty,
      hiddenDim: this.config.autoHiddenDim ? 'auto' : this.config.hiddenDimInput
    });
  }

  // ... other helper methods
}
