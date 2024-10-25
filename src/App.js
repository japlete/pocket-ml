import React, { useState, useEffect } from 'react';
import CSVUploader from './components/CSVUploader.js';
import DataPreview from './components/DataPreview.js';
import TargetSelector from './components/TargetSelector.js';
import TargetTypeSelector from './components/TargetTypeSelector.js';
import AdvancedSettings from './components/AdvancedSettings.js';
import Accordion from './components/Accordion.js';
import { preprocessData } from './utils/dataPreprocessing.js';
import { trainModel } from './utils/modelTraining.js';
import './index.css';

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

  const handleDataParsed = (data) => {
    setParsedData(data);
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
    setTrainingProgress(0);
    if (parsedData && targetColumn) {
      const filteredData = parsedData.filter(row => row[targetColumn] !== null && row[targetColumn] !== '');
      
      const { trainData, validationData, testData, updatedColumns, scaler, classMapping } = preprocessData(filteredData, targetColumn, targetType, splitRatios, seed);
      setProcessedData({ trainData, validationData, testData, updatedColumns });

      const featureColumns = updatedColumns.filter(col => col !== targetColumn);
      const results = await trainModel(
        trainData,
        validationData,
        testData,
        targetColumn,
        featureColumns,
        targetType,
        classMapping,
        (epoch, totalEpochs) => {
          setTrainingProgress(Math.round((epoch / totalEpochs) * 100));
        },
        primaryMetric,
        secondaryMetrics,
        seed,
        learningRate,
        l1_penalty,
        dropout_rate,
        batchSize,
        epochs,
        earlyStoppingEnabled,
        autoHiddenDim,
        hiddenDimInput
      );
      setModelResults(results);
      setIsTraining(false);
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
  };

  const handlePrimaryMetricChange = (metric) => {
    setPrimaryMetric(metric);
  };

  const handleSecondaryMetricsChange = (metrics) => {
    setSecondaryMetrics(metrics);
  };

  const columns = parsedData ? Object.keys(parsedData[0]) : [];

  return (
    <div className="App">
      <h1>Pocket ML</h1>
      {parsedData && <button onClick={handleReset} className="home-button">Home</button>}
      {!parsedData && <CSVUploader onDataParsed={handleDataParsed} />}
      {parsedData && !isTraining && !modelResults && (
        <>
          <p>{parsedData.length} rows loaded</p>
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
      {isTraining && (
        <div>
          <h2>Model Training</h2>
          <p>Model is training. Please wait... {trainingProgress}% completed</p>
        </div>
      )}
      {!isTraining && modelResults && (
        <div>
          <h2>Model Results</h2>
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
                <td>{modelResults.trainLoss.toFixed(4)}</td>
                <td>{modelResults.validationLoss.toFixed(4)}</td>
                <td>{modelResults.testLoss.toFixed(4)}</td>
              </tr>
              {[primaryMetric, ...secondaryMetrics].map((metric) => {
                const trainValue = modelResults[`train${metric.toUpperCase()}`];
                const validationValue = modelResults[`validation${metric.toUpperCase()}`];
                const testValue = modelResults[`test${metric.toUpperCase()}`];
                
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
        </div>
      )}
      {processedData && (
        <Accordion title="Data Preview (transformed features after preprocessing)">
          <DataPreview 
            data={processedData.trainData} 
            columns={processedData.updatedColumns}
          />
        </Accordion>
      )}
    </div>
  );
}

export default App;
