import React, { useState, useEffect } from 'react';
import CSVUploader from './components/CSVUploader.js';
import DataPreview from './components/DataPreview.js';
import TargetSelector from './components/TargetSelector.js';
import TargetTypeSelector from './components/TargetTypeSelector.js';
import AdvancedSettings from './components/AdvancedSettings.js';
import Accordion from './components/Accordion.js';
import { preprocessData } from './utils/dataPreprocessing.js';
import { trainModel } from './utils/modelTraining.js';
import { ModelTrainingManager } from './utils/modelTrainingManager';
import TuningParameters from './components/TuningParameters';
import SaveModelDialog from './components/SaveModelDialog';
import SavedModelsList from './components/SavedModelsList';
import ModelArchitectureDisplay from './components/ModelArchitectureDisplay';
import './index.css';
import { LinearProgress, Box, Typography } from '@mui/material';
import githubIcon from './assets/icons/github-mark-white.svg';

function App() {
  const [parsedData, setParsedData] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [targetType, setTargetType] = useState('');
  const [suggestedTargetType, setSuggestedTargetType] = useState('');
  const [processedData, setProcessedData] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [missingValueCount, setMissingValueCount] = useState(0);
  const [uniqueValueCount, setUniqueValueCount] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [splitRatios, setSplitRatios] = useState({ train: 0.7, validation: 0.2, test: 0.1 });
  const [seed, setSeed] = useState(42);
  const [primaryMetric, setPrimaryMetric] = useState(targetType === 'regression' ? 'rmse' : 'roc_auc');
  const [secondaryMetrics, setSecondaryMetrics] = useState([]);
  const [classDistribution, setClassDistribution] = useState(null);
  const [allowedTargetTypes, setAllowedTargetTypes] = useState([]);
  const [l1_penalty, setL1Penalty] = useState(0.0);
  const [dropout_rate, setDropoutRate] = useState(0.1);
  const [batchSize, setBatchSize] = useState(8);
  const [epochs, setEpochs] = useState(50);
  const [earlyStoppingEnabled, setEarlyStoppingEnabled] = useState(true);
  const [learningRate, setLearningRate] = useState(0.003);
  const [autoHiddenDim, setAutoHiddenDim] = useState(true);
  const [hiddenDimInput, setHiddenDimInput] = useState(32);
  const [fileName, setFileName] = useState('');
  const [minIterations, setMinIterations] = useState(5);
  const [maxTrainingTime, setMaxTrainingTime] = useState(5);
  const [trainingManager, setTrainingManager] = useState(null);
  const [trainedModels, setTrainedModels] = useState([]);
  const [bestModel, setBestModel] = useState(null);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [dataPreviewOpen, setDataPreviewOpen] = useState(true);
  const [processedDataPreviewOpen, setProcessedDataPreviewOpen] = useState(true);
  const [modelArchitectureOpen, setModelArchitectureOpen] = useState(true);
  const [isStopping, setIsStopping] = useState(false);

  const handleDataParsed = (data, name) => {
    setParsedData(data);
    setFileName(name);
    setTargetColumn('');
    setTargetType('');
    setSuggestedTargetType('');
    setProcessedData(null);
    setModelResults(null);
    setMissingValueCount(0);
    setUniqueValueCount(0);
    setIsTraining(false);
  };

  const handleTargetChange = (column) => {
    setTargetColumn(column);
    detectTargetType(column);
    checkTargetColumnValidity(column);
  };

  const handleTargetTypeChange = (type) => {
    setTargetType(type);
  };

  const detectTargetType = (column) => {
    if (!parsedData || !column) return;

    const values = parsedData.map(row => row[column]);
    const uniqueValues = [...new Set(values.filter(value => value !== null && value !== ''))];
    const isNumeric = uniqueValues.every(value => typeof value === 'number' && !isNaN(value));

    let detectedType;
    let allowedTypes = [];

    if (isNumeric) {
      if (uniqueValues.length === 2) {
        detectedType = 'binary';
        allowedTypes = ['binary', 'regression'];
      } else if (uniqueValues.length >= 3 && uniqueValues.length <= 10) {
        detectedType = 'multiclass';
        allowedTypes = ['multiclass', 'regression'];
      } else {
        detectedType = 'regression';
        allowedTypes = ['regression'];
      }
    } else {
      if (uniqueValues.length === 2) {
        detectedType = 'binary';
        allowedTypes = ['binary'];
      } else {
        detectedType = 'multiclass';
        allowedTypes = ['multiclass'];
      }
    }

    setSuggestedTargetType(detectedType);
    setTargetType(detectedType);
    setAllowedTargetTypes(allowedTypes);

    // Compute class distribution for classification problems
    if (detectedType !== 'regression') {
      const distribution = values.reduce((acc, val) => {
        acc[val] = (acc[val] || 0) + 1;
        return acc;
      }, {});
      const total = Object.values(distribution).reduce((sum, count) => sum + count, 0);
      const normalizedDistribution = Object.fromEntries(
        Object.entries(distribution).map(([key, value]) => [key, value / total])
      );
      setClassDistribution(normalizedDistribution);
    } else {
      setClassDistribution(null);
    }
  };

  const checkTargetColumnValidity = (column) => {
    if (!parsedData || !column) return;

    const values = parsedData.map(row => row[column]);
    const missingCount = values.filter(value => value === null || value === '').length;
    const uniqueValues = [...new Set(values.filter(value => value !== null && value !== ''))];

    setMissingValueCount(missingCount);
    setUniqueValueCount(uniqueValues.length);
  };

  const handleSplitChange = (newSplitRatios) => {
    setSplitRatios({
      train: newSplitRatios.train / 100,
      validation: newSplitRatios.validation / 100,
      test: newSplitRatios.test / 100
    });
  };

  const handleSeedChange = (newSeed) => {
    setSeed(newSeed);
  };

  const handleStartTraining = async () => {
    setIsTraining(true);
    setIsStopping(false);
    setTrainingProgress(0);
    setDataPreviewOpen(false);
    setProcessedDataPreviewOpen(true);
    
    if (parsedData && targetColumn) {
      try {
        const filteredData = parsedData.filter(row => row[targetColumn] !== null && row[targetColumn] !== '');
        const preprocessedData = preprocessData(filteredData, targetColumn, targetType, splitRatios, seed);
        setProcessedData(preprocessedData);

        const featureColumns = preprocessedData.updatedColumns.filter(col => col !== targetColumn);

        const config = {
          targetType,
          primaryMetric,
          secondaryMetrics,
          minIterations,
          maxTrainingTime,
          learningRate,
          l1Penalty: l1_penalty,
          dropoutRate: dropout_rate,
          batchSize,
          epochs,
          earlyStoppingEnabled: Boolean(earlyStoppingEnabled),
          autoHiddenDim,
          hiddenDimInput,
          seed,
          targetColumn,
          featureColumns,
          numClasses: targetType === 'multiclass' ? 
            [...new Set(preprocessedData.trainData.map(row => row[targetColumn]))].length : 
            (targetType === 'binary' ? 2 : 1)
        };

        const manager = new ModelTrainingManager(config);
        setTrainingManager(manager);

        const results = await manager.startTrainingCycle(
          preprocessedData.trainData,
          preprocessedData.validationData,
          preprocessedData.testData,
          (epoch, totalEpochs) => {
            setTrainingProgress(Math.round((epoch / totalEpochs) * 100));
          },
          (currentModels, bestIteration) => {
            setTrainedModels(currentModels);
          }
        );

        if (results) {
          setTrainedModels(results.trainedModels);
          setBestModel(results.bestModel);
          setModelResults({
            ...results.finalMetrics,
            bestModelIteration: results.bestModelIteration
          });
          setDataPreviewOpen(false);
          setProcessedDataPreviewOpen(false);
          setModelArchitectureOpen(true);
        }
      } catch (error) {
        console.error('Training error:', error);
        // Handle error appropriately
      } finally {
        setIsTraining(false);
        setIsStopping(false);
      }
    }
  };

  const handleReset = () => {
    setParsedData(null);
    setTargetColumn('');
    setTargetType('');
    setSuggestedTargetType('');
    setProcessedData(null);
    setModelResults(null);
    setMissingValueCount(0);
    setUniqueValueCount(0);
    setIsTraining(false);
    setBestModel(null);
  };

  const handlePrimaryMetricChange = (metric) => {
    setPrimaryMetric(metric);
  };

  const handleSecondaryMetricsChange = (metrics) => {
    setSecondaryMetrics(metrics);
  };

  const handleStopTraining = async () => {
    if (trainingManager) {
      setIsStopping(true);
      trainingManager.stopTraining();
    }
  };

  const handleSaveModel = () => {
    setShowSaveDialog(true);
  };

  const handleSaveDialogClose = (success) => {
    setShowSaveDialog(false);
    if (success) {
      // Optionally show a success message
      alert('Model saved successfully!');
    }
  };

  const columns = parsedData ? Object.keys(parsedData[0]) : [];

  const getTimeProgress = () => {
    if (!trainingManager?.startTime) return 0;
    const elapsed = (Date.now() - trainingManager.startTime) / 1000; // in seconds
    const maxTime = maxTrainingTime * 60; // convert minutes to seconds
    return Math.min((elapsed / maxTime) * 100, 100);
  };

  const getIterationProgress = () => {
    if (!trainedModels.length) return 0;
    return Math.min((trainedModels.length / minIterations) * 100, 100);
  };

  return (
    <div className="App">
      <h1>Pocket ML</h1>
      
      {/* Add welcome section when no data is loaded */}
      {!parsedData && (
        <div className="welcome-section">
          <p className="welcome-text">
            Train machine learning models directly in your browser
          </p>
          <div className="key-features">
            <div className="feature">
              <h3>No Setup Required</h3>
              <p>Start immediately without installation or programming skills</p>
            </div>
            <div className="feature">
              <h3>100% Private</h3>
              <p>Your data never leaves your device - all processing happens locally</p>
            </div>
            <div className="feature">
              <h3>Completely Free</h3>
              <p>Open source and free to use, no account needed</p>
            </div>
          </div>
        </div>
      )}

      <div className="panel-container">
        {/* Left Panel */}
        <div className="left-panel">
          {parsedData && <button onClick={handleReset} className="home-button">Home</button>}
          {!parsedData && <CSVUploader onDataParsed={handleDataParsed} />}
          
          {/* Show configuration UI only when data is loaded AND not training/trained */}
          {parsedData && !isTraining && !modelResults && (
            <>
              <TargetSelector
                columns={columns}
                selectedTarget={targetColumn}
                onTargetChange={handleTargetChange}
              />
              {targetColumn && (
                <>
                  {missingValueCount > 0 && (
                    <p className="warning">
                      Warning: {missingValueCount} rows with missing values detected in the target column. These rows will be discarded.
                    </p>
                  )}
                  {uniqueValueCount === 1 ? (
                    <p className="error">Error: The target column has only one unique value. Please select a different target column.</p>
                  ) : (
                    <TargetTypeSelector
                      targetType={targetType}
                      suggestedType={suggestedTargetType}
                      onTargetTypeChange={handleTargetTypeChange}
                      allowedTypes={allowedTargetTypes}
                    />
                  )}
                  <TuningParameters
                    maxTrainingTime={maxTrainingTime}
                    onMaxTrainingTimeChange={setMaxTrainingTime}
                    minIterations={minIterations}
                    onMinIterationsChange={setMinIterations}
                  />
                </>
              )}
              <AdvancedSettings
                onSplitChange={handleSplitChange}
                onSeedChange={handleSeedChange}
                onPrimaryMetricChange={handlePrimaryMetricChange}
                onSecondaryMetricsChange={handleSecondaryMetricsChange}
                onL1PenaltyChange={setL1Penalty}
                onDropoutRateChange={setDropoutRate}
                onBatchSizeChange={setBatchSize}
                onEpochsChange={setEpochs}
                onEarlyStoppingChange={setEarlyStoppingEnabled}
                onLearningRateChange={setLearningRate}
                onAutoHiddenDimChange={setAutoHiddenDim}
                onHiddenDimInputChange={setHiddenDimInput}
                autoHiddenDim={autoHiddenDim}
                hiddenDimInput={hiddenDimInput}
                targetType={targetType}
                classDistribution={classDistribution}
                l1_penalty={l1_penalty}
                dropout_rate={dropout_rate}
                batchSize={batchSize}
                epochs={epochs}
                earlyStoppingEnabled={earlyStoppingEnabled}
                learningRate={learningRate}
              />
              <button 
                onClick={handleStartTraining} 
                disabled={!targetColumn || !targetType || uniqueValueCount <= 1}
              >
                Start Training
              </button>
            </>
          )}

          {/* Show training progress */}
          {isTraining && (
            <div className="training-status">
              <h2>Model Training</h2>
              
              <div className="progress-section">
                <div className="progress-item">
                  <Typography className="progress-label">
                    Time: {Math.round(getTimeProgress())}% of max {maxTrainingTime} minutes
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={getTimeProgress()} 
                    className="progress-bar time"
                  />
                </div>

                <div className="progress-item">
                  <Typography className="progress-label">
                    Iterations: {trainedModels.length} of min {minIterations}
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={getIterationProgress()} 
                    className="progress-bar iterations"
                  />
                </div>

                <div className="progress-item">
                  <Typography className="progress-label">
                    Current Model: {trainingProgress}% complete
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={trainingProgress} 
                    className="progress-bar current"
                  />
                </div>

                <div className="status-info">
                  <Typography variant="body2" className="info-text">
                    Training model {trainedModels.length + 1}
                    {trainedModels.length > 0 && trainedModels[0].metrics?.validation && 
                      ` • Best validation ${primaryMetric}: ${
                        trainedModels
                          .reduce((best, current) => {
                            const metricKey = primaryMetric.toUpperCase();
                            const currentMetric = current.metrics.validation[metricKey];
                            const bestMetric = best.metrics.validation[metricKey];
                            
                            const higherIsBetter = !['rmse', 'mse', 'mae', 'mape'].includes(primaryMetric.toLowerCase());
                            
                            return higherIsBetter ? 
                              (currentMetric > bestMetric ? current : best) : 
                              (currentMetric < bestMetric ? current : best);
                          }, trainedModels[0])
                          .metrics.validation[primaryMetric.toUpperCase()]
                          .toFixed(4)
                      }`
                    }
                  </Typography>
                </div>
              </div>

              {isStopping ? (
                <div className="stopping-message">
                  Stopping training. Waiting for the last model to finish...
                </div>
              ) : (
                <button 
                  onClick={handleStopTraining}
                  className="stop-training-button"
                >
                  Stop Training
                </button>
              )}
            </div>
          )}

          {/* Show results */}
          {!isTraining && modelResults && (
            <div>
              <h2>Model Results</h2>
              <p className="iterations-info">
                Best of {trainedModels.length} models (iteration {modelResults.bestModelIteration})
              </p>
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Train</th>
                    <th>Validation</th>
                    <th>Test</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Loss</td>
                    <td>{modelResults.train.loss.toFixed(4)}</td>
                    <td>{modelResults.validation.loss.toFixed(4)}</td>
                    <td>{modelResults.test.loss.toFixed(4)}</td>
                  </tr>
                  {[primaryMetric, ...secondaryMetrics].map((metric) => {
                    const trainValue = modelResults.train[metric.toUpperCase()];
                    const validationValue = modelResults.validation[metric.toUpperCase()];
                    const testValue = modelResults.test[metric.toUpperCase()];
                    
                    if (trainValue !== undefined && validationValue !== undefined && testValue !== undefined) {
                      const formattedMetricName = metric
                        .split('_')
                        .map((word, index) => 
                          index === 0 
                            ? word.charAt(0).toUpperCase() + word.slice(1).toLowerCase() 
                            : ['auc', 'roc', 'pr'].includes(word.toLowerCase()) 
                              ? word.toUpperCase() 
                              : word.toLowerCase()
                        )
                        .join(' ');

                      return (
                        <tr key={metric}>
                          <td>{formattedMetricName}</td>
                          <td>{trainValue.toFixed(4)}</td>
                          <td>{validationValue.toFixed(4)}</td>
                          <td>{testValue.toFixed(4)}</td>
                        </tr>
                      );
                    }
                    return null;
                  })}
                </tbody>
              </table>
              <button 
                onClick={handleSaveModel}
                className="save-model-button"
              >
                Save Model
              </button>
              {showSaveDialog && (
                <SaveModelDialog
                  model={bestModel}
                  modelResults={modelResults}
                  config={{
                    targetType,
                    primaryMetric,
                    secondaryMetrics,
                    l1_penalty,
                    dropout_rate,
                    batchSize,
                    epochs,
                    earlyStoppingEnabled,
                    learningRate,
                    autoHiddenDim,
                    hiddenDimInput,
                    seed,
                    // Add any other relevant configuration
                  }}
                  onClose={handleSaveDialogClose}
                />
              )}
            </div>
          )}
        </div>

        {/* Right Panel */}
        <div className="right-panel">
          {parsedData ? (
            <>
              <Accordion 
                title="Data Preview" 
                defaultOpen={dataPreviewOpen}
                isOpen={dataPreviewOpen}
                onToggle={setDataPreviewOpen}
              >
                <DataPreview 
                  data={parsedData} 
                  columns={Object.keys(parsedData[0])}
                />
              </Accordion>
              {processedData && (
                <Accordion 
                  title="Preprocessed Data Preview"
                  defaultOpen={processedDataPreviewOpen}
                  isOpen={processedDataPreviewOpen}
                  onToggle={setProcessedDataPreviewOpen}
                >
                  <DataPreview 
                    data={[
                      ...processedData.trainData,
                      ...processedData.validationData,
                      ...processedData.testData
                    ]}
                    columns={processedData.updatedColumns}
                    showDownload={true}
                    originalFileName={fileName}
                  />
                </Accordion>
              )}
              {bestModel && (
                <Accordion 
                  title="Model Architecture"
                  defaultOpen={modelArchitectureOpen}
                  isOpen={modelArchitectureOpen}
                  onToggle={setModelArchitectureOpen}
                >
                  <ModelArchitectureDisplay
                    model={bestModel}
                    config={{
                      autoHiddenDim,
                      hiddenDimInput,
                      learningRate,
                      batchSize,
                      l1_penalty,
                      dropoutRate: dropout_rate,
                      earlyStoppingEnabled
                    }}
                    featureCount={processedData.updatedColumns.filter(col => 
                      col !== targetColumn && col !== 'split'
                    ).length}
                  />
                </Accordion>
              )}
            </>
          ) : (
            <SavedModelsList />
          )}
        </div>
      </div>

      {/* Add the footer at the bottom of the App component, after the panel-container div */}
      <footer className="footer">
        <p>
          © {new Date().getFullYear()} José A Poblete
          <a href="https://github.com/japlete/pocket-ml" target="_blank" rel="noopener noreferrer">
            <img src={githubIcon} alt="GitHub" className="footer-icon" />
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
