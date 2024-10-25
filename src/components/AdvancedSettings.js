import React, { useState, useEffect } from 'react';
import Accordion from './Accordion';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';

function AdvancedSettings({ onSplitChange, onSeedChange, onPrimaryMetricChange, onSecondaryMetricsChange, targetType, classDistribution }) {
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
              <option key={metric} value={metric}>{metric.toUpperCase()}</option>
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
                <label htmlFor={`secondary-${metric}`}>{metric.toUpperCase()}</label>
              </div>
            ))}
          </div>
        )}
      </Accordion>
    </Accordion>
  );
}

export default AdvancedSettings;
