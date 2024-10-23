import React, { useState } from 'react';
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
    if (isNumeric) {
      if (uniqueValues.length === 2) {
        detectedType = 'binary';
      } else if (uniqueValues.length >= 3 && uniqueValues.length <= 10) {
        detectedType = 'multiclass';
      } else {
        detectedType = 'regression';
      }
    } else {
      detectedType = uniqueValues.length === 2 ? 'binary' : 'multiclass';
    }

    setSuggestedTargetType(detectedType);
    setTargetType(detectedType);
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
      const results = await trainModel(trainData, validationData, testData, targetColumn, featureColumns, scaler, targetType, classMapping, (epoch, totalEpochs) => {
        setTrainingProgress(Math.round((epoch / totalEpochs) * 100));
      });
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
                />
              )}
            </>
          )}
          <AdvancedSettings
            onSplitChange={handleSplitChange}
            onSeedChange={handleSeedChange}
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
          {targetType === 'regression' ? (
            <>
              <p>Train RMSE: {modelResults.trainRMSE.toFixed(4)}</p>
              <p>Validation RMSE: {modelResults.validationRMSE.toFixed(4)}</p>
              <p>Test RMSE: {modelResults.testRMSE.toFixed(4)}</p>
            </>
          ) : (
            <>
              <p>Train Accuracy: {(modelResults.trainAccuracy * 100).toFixed(2)}%</p>
              <p>Validation Accuracy: {(modelResults.validationAccuracy * 100).toFixed(2)}%</p>
              <p>Test Accuracy: {(modelResults.testAccuracy * 100).toFixed(2)}%</p>
            </>
          )}
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
