import React from 'react';

function ModelArchitectureDisplay({ model, config, featureCount }) {
  // Calculate total parameters
  const getTotalParams = () => {
    return model ? model.countParams() : 0;
  };

  // Get number of dense layers
  const getDenseLayerCount = () => {
    if (!model) return 0;
    return model.layers.filter(layer => layer.getClassName() === 'Dense').length;
  };

  // Get all hidden layer sizes
  const getHiddenLayerSizes = () => {
    if (!model) return 'NA';
    const denseLayers = model.layers.filter(layer => layer.getClassName() === 'Dense');
    // Skip first (input) and last (output) layers
    if (denseLayers.length <= 2) return 'NA';
    const hiddenLayers = denseLayers.slice(0, -1);
    return hiddenLayers
      .map(layer => layer.units || layer.getConfig().units)
      .join(' ');
  };

  // Get actual epochs trained
  const getActualEpochs = () => {
    if (!model || !model.history) return 'NA';
    return model.history.epoch ? model.history.epoch.length : 'NA';
  };

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(2)}M`;
    } else if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
  };

  return (
    <div className="model-architecture">
      <h3>Model Architecture</h3>
      <div className="architecture-grid">
        <div className="architecture-item">
          <label>Input Features:</label>
          <span>{featureCount}</span>
        </div>
        <div className="architecture-item">
          <label>Total Parameters:</label>
          <span>{formatNumber(getTotalParams())}</span>
        </div>
        <div className="architecture-item">
          <label>Dense Layers:</label>
          <span>{getDenseLayerCount()}</span>
        </div>
        <div className="architecture-item">
          <label>Hidden Layer Sizes:</label>
          <span>{getHiddenLayerSizes()}</span>
        </div>
        <div className="architecture-item">
          <label>Learning Rate:</label>
          <span>{config.learningRate}</span>
        </div>
        <div className="architecture-item">
          <label>Batch Size:</label>
          <span>{config.batchSize}</span>
        </div>
        <div className="architecture-item">
          <label>L1 Penalty:</label>
          <span>{config.l1_penalty}</span>
        </div>
        <div className="architecture-item">
          <label>Dropout Rate:</label>
          <span>{config.dropoutRate}</span>
        </div>
        <div className="architecture-item">
          <label>Epochs Trained:</label>
          <span>{getActualEpochs()}</span>
        </div>
      </div>
    </div>
  );
}

export default ModelArchitectureDisplay; 