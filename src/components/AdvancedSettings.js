import React, { useState, useEffect } from 'react';
import Accordion from './Accordion';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';

function AdvancedSettings({ 
  onSplitChange, 
  onSeedChange, 
  onPrimaryMetricChange, 
  onSecondaryMetricsChange, 
  onL1PenaltyChange,
  onDropoutRateChange,
  onBatchSizeChange,
  onEpochsChange,
  onEarlyStoppingChange,
  onLearningRateChange,
  targetType, 
  classDistribution,
  l1_penalty,
  dropout_rate,
  batchSize,
  epochs,
  earlyStoppingEnabled,
  learningRate,
  onAutoHiddenDimChange,
  onHiddenDimInputChange,
  autoHiddenDim,
  hiddenDimInput
}) {
  const [splitRatios, setSplitRatios] = useState([70, 90]);
  const [seed, setSeed] = useState(42);
  const [primaryMetric, setPrimaryMetric] = useState('');
  const [secondaryMetrics, setSecondaryMetrics] = useState([]);

  useEffect(() => {
    let defaultPrimaryMetric = '';
    let defaultSecondaryMetrics = [];

    if (targetType === 'regression') {
      defaultPrimaryMetric = 'rmse';
      defaultSecondaryMetrics = ['mae'];
    } else if (targetType === 'binary') {
      const minorityClassProportion = Math.min(...Object.values(classDistribution));
      if (minorityClassProportion < 0.3) {
        defaultPrimaryMetric = 'pr_auc';
        defaultSecondaryMetrics = ['roc_auc', 'precision', 'recall', 'accuracy'];
      } else {
        defaultPrimaryMetric = 'accuracy';
        defaultSecondaryMetrics = ['roc_auc', 'precision', 'recall'];
      }
    } else if (targetType === 'multiclass') {
      defaultPrimaryMetric = 'accuracy';
      defaultSecondaryMetrics = [];
    }

    setPrimaryMetric(defaultPrimaryMetric);
    setSecondaryMetrics(defaultSecondaryMetrics);
    onPrimaryMetricChange(defaultPrimaryMetric);
    onSecondaryMetricsChange(defaultSecondaryMetrics);
  }, [targetType, classDistribution]);

  const handleSplitChange = (values) => {
    setSplitRatios(values);
    const [trainVal, valTest] = values;
    const train = trainVal;
    const val = valTest - trainVal;
    const test = 100 - valTest;
    onSplitChange({ train, validation: val, test });
  };

  const handleSeedChange = (e) => {
    const newSeed = parseInt(e.target.value);
    setSeed(newSeed);
    onSeedChange(newSeed);
  };

  const handlePrimaryMetricChange = (e) => {
    const metric = e.target.value;
    setPrimaryMetric(metric);
    onPrimaryMetricChange(metric);
    setSecondaryMetrics(prevMetrics => prevMetrics.filter(m => m !== metric));
  };

  const handleSecondaryMetricChange = (metric) => {
    setSecondaryMetrics(prevMetrics => {
      const updatedMetrics = prevMetrics.includes(metric)
        ? prevMetrics.filter(m => m !== metric)
        : [...prevMetrics, metric];
      onSecondaryMetricsChange(updatedMetrics);
      return updatedMetrics;
    });
  };

  const regressionMetrics = ['rmse', 'mse', 'mae', 'mape', 'r2'];
  const binaryClassificationMetrics = ['accuracy', 'roc_auc', 'pr_auc', 'precision', 'recall', 'f1'];
  const multiclassMetrics = ['accuracy'];

  let metrics;
  if (targetType === 'regression') {
    metrics = regressionMetrics;
  } else if (targetType === 'binary') {
    metrics = binaryClassificationMetrics;
  } else if (targetType === 'multiclass') {
    metrics = multiclassMetrics;
  } else {
    metrics = [];
  }

  const handleL1PenaltyChange = (e) => {
    const value = parseFloat(e.target.value);
    onL1PenaltyChange(value);
  };

  const handleDropoutRateChange = (e) => {
    const value = parseFloat(e.target.value);
    onDropoutRateChange(value);
  };

  const handleBatchSizeChange = (e) => {
    const value = parseInt(e.target.value);
    if (value > 0) {
      onBatchSizeChange(value);
    } else {
      onBatchSizeChange(1); // Increase to 1 if zero
    }
  };

  const handleEpochsChange = (e) => {
    const value = parseInt(e.target.value);
    if (value > 0) {
      onEpochsChange(value);
    } else {
      onEpochsChange(1); // Increase to 1 if zero
    }
  };

  const handleEarlyStoppingChange = (e) => {
    onEarlyStoppingChange(e.target.checked);
  };

  const handleLearningRateChange = (e) => {
    const value = parseFloat(e.target.value);
    onLearningRateChange(value);
  };

  const handleAutoHiddenDimChange = (e) => {
    onAutoHiddenDimChange(e.target.checked);
  };

  const handleHiddenDimInputChange = (e) => {
    const value = parseInt(e.target.value);
    if (value > 0) {
      onHiddenDimInputChange(value);
    } else {
      onHiddenDimInputChange(1); // Minimum value of 1
    }
  };

  const formatMetricName = (metric) => {
    return metric
      .split('_')
      .map((word, index) => 
        index === 0 
          ? word.charAt(0).toUpperCase() + word.slice(1).toLowerCase() 
          : ['auc', 'roc', 'pr'].includes(word.toLowerCase()) 
            ? word.toUpperCase() 
            : word.toLowerCase()
      )
      .join(' ');
  };

  return (
    <Accordion title="Advanced">
      <Accordion title="Train-val-test split">
        <div style={{ margin: '20px 0' }}>
          <Slider
            range
            min={0}
            max={100}
            defaultValue={splitRatios}
            onChange={handleSplitChange}
          />
          <p>Training {splitRatios[0]}% Validation {splitRatios[1] - splitRatios[0]}% Test {100 - splitRatios[1]}%</p>
        </div>
        <div>
          <label htmlFor="seed">Random Seed: </label>
          <input
            type="number"
            id="seed"
            value={seed}
            onChange={handleSeedChange}
            min={0}
          />
        </div>
      </Accordion>
      <Accordion title="Metrics">
        <div>
          <label htmlFor="primary-metric">Primary Validation Metric: </label>
          <select id="primary-metric" value={primaryMetric} onChange={handlePrimaryMetricChange}>
            {metrics.map(metric => (
              <option key={metric} value={metric}>{formatMetricName(metric)}</option>
            ))}
          </select>
        </div>
        {targetType !== 'multiclass' && (
          <div>
            <p>Secondary Metrics:</p>
            {metrics.map(metric => (
              <div key={metric}>
                <input
                  type="checkbox"
                  id={`secondary-${metric}`}
                  checked={secondaryMetrics.includes(metric) && metric !== primaryMetric}
                  onChange={() => handleSecondaryMetricChange(metric)}
                  disabled={metric === primaryMetric}
                />
                <label htmlFor={`secondary-${metric}`}>{formatMetricName(metric)}</label>
              </div>
            ))}
          </div>
        )}
      </Accordion>
      <Accordion title="Model Hyperparameters">
      <div style={{ fontSize: '12px', color: '#888' }}>
        Some of these parameters will be adjusted during the tuning iterations.
      </div>
        <div>
          <label htmlFor="learning-rate">Learning Rate: </label>
          <input
            type="number"
            id="learning-rate"
            value={learningRate}
            onChange={handleLearningRateChange}
            step="0.0001"
            min="0.0001"
            max="1"
          />
        </div>
        <div>
          <label htmlFor="batch-size">Batch Size: </label>
          <input
            type="number"
            id="batch-size"
            value={batchSize}
            onChange={handleBatchSizeChange}
            step="8"
            min="0"
          />
        </div>
        <div>
          <label htmlFor="epochs">Epochs: </label>
          <input
            type="number"
            id="epochs"
            value={epochs}
            onChange={handleEpochsChange}
            step="10"
            min="0"
          />
        </div>
        <div>
          <input
            type="checkbox"
            id="early-stopping"
            checked={earlyStoppingEnabled}
            onChange={handleEarlyStoppingChange}
          />
          <label htmlFor="early-stopping">Enable Early Stopping</label>
        </div>
        <div>
          <label htmlFor="dropout-rate">Dropout Rate: </label>
          <input
            type="number"
            id="dropout-rate"
            value={dropout_rate}
            onChange={handleDropoutRateChange}
            step="0.05"
            min="0"
            max="1"
          />
        </div>
        <div>
          <label htmlFor="l1-penalty">L1 Penalty: </label>
          <input
            type="number"
            id="l1-penalty"
            value={l1_penalty}
            onChange={handleL1PenaltyChange}
            step="0.001"
            min="0"
          />
        </div>
        <div>
          <input
            type="checkbox"
            id="auto-hidden-dim"
            checked={autoHiddenDim}
            onChange={handleAutoHiddenDimChange}
          />
          <label htmlFor="auto-hidden-dim">Automatic model sizing</label>
        </div>
        <div>
          <label htmlFor="hidden-dim-input">Hidden Dimension: </label>
          <input
            type="number"
            id="hidden-dim-input"
            value={autoHiddenDim ? '' : hiddenDimInput}
            onChange={handleHiddenDimInputChange}
            disabled={autoHiddenDim}
            min="1"
            step="1"
          />
        </div>
      </Accordion>
    </Accordion>
  );
}

export default AdvancedSettings;
