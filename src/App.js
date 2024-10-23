import React, { useState } from 'react';
import CSVUploader from './components/CSVUploader.js';
import DataPreview from './components/DataPreview.js';
import TargetSelector from './components/TargetSelector.js';
import { preprocessData } from './utils/dataPreprocessing.js';
import { trainModel } from './utils/modelTraining.js';
import './index.css';

function App() {
  const [parsedData, setParsedData] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [processedData, setProcessedData] = useState(null);
  const [modelResults, setModelResults] = useState(null);

  const handleDataParsed = (data) => {
    setParsedData(data);
    setTargetColumn('');
    setProcessedData(null);
    setModelResults(null);
  };

  const handleTargetChange = (column) => {
    setTargetColumn(column);
  };

  const handleStartTraining = async () => {
    if (parsedData && targetColumn) {
      const { trainData, testData, updatedColumns, scaler } = preprocessData(parsedData, targetColumn);
      setProcessedData({ trainData, testData, updatedColumns });

      const featureColumns = updatedColumns.filter(col => col !== targetColumn);
      const results = await trainModel(trainData, testData, targetColumn, featureColumns, scaler);
      setModelResults(results);
    }
  };

  const columns = parsedData ? Object.keys(parsedData[0]) : [];

  return (
    <div className="App">
      <h1>Pocket ML</h1>
      <CSVUploader onDataParsed={handleDataParsed} />
      {parsedData && (
        <>
          <TargetSelector
            columns={columns}
            selectedTarget={targetColumn}
            onTargetChange={handleTargetChange}
          />
          <button onClick={handleStartTraining} disabled={!targetColumn}>
            Start Training
          </button>
          {modelResults && (
            <div>
              <h2>Model Results</h2>
              <p>Train RMSE: {modelResults.trainRMSE.toFixed(4)}</p>
              <p>Test RMSE: {modelResults.testRMSE.toFixed(4)}</p>
            </div>
          )}
          <DataPreview 
            data={processedData ? processedData.trainData : parsedData} 
            columns={processedData ? processedData.updatedColumns : columns}
          />
        </>
      )}
    </div>
  );
}

export default App;
