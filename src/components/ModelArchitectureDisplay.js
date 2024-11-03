import React from 'react';

function ModelArchitectureDisplay({ model, config, featureCount }) {
  // Calculate total parameters
  const getTotalParams = () => {
    return model ? model.countParams() : 'N/A';
  };

  // Get number of dense layers
  const getDenseLayerCount = () => {
    if (!model) return 'N/A';
    return model.layers.filter(layer => layer.getClassName() === 'Dense').length;
  };

  // Get all hidden layer sizes including concat layer
  const getHiddenLayerSizes = () => {
    if (!model) return 'N/A';
    
    const layers = model.layers;
    const hiddenSizes = [];
    
    // Find all dense layers except the output layer
    const denseLayers = layers.filter(layer => 
      layer.getClassName() === 'Dense'
    ).slice(0, -1); // Exclude output layer
    
    // Add dense layer sizes
    hiddenSizes.push(...denseLayers.map(layer => layer.units));
    
    // Find concat layer (it should be after the dense layers)
    const concatLayer = layers.find(layer => layer.getClassName() === 'Concatenate');
    if (concatLayer) {
      // outputShape is like [null, size] where null is batch dimension
      const concatSize = concatLayer.outputShape[1];
      // Use + to denote concatenation
      return `${hiddenSizes.join(' → ')} (+) ${concatSize}`;
    }

    return hiddenSizes.join(' → ');
  };

  // Get actual epochs trained - now using metadata if available
  const getActualEpochs = () => {
    if (config.metadata?.epochsTrained !== undefined) {
      return config.metadata.epochsTrained;
    }
    if (!model || !model.history) return 'N/A';
    return model.history.epoch ? model.history.epoch.length : 'N/A';
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
          <label>Input Features</label>
          <span>{featureCount}</span>
        </div>
        <div className="architecture-item">
          <label>Total Parameters</label>
          <span>{formatNumber(getTotalParams())}</span>
        </div>
        <div className="architecture-item">
          <label>Dense Layers</label>
          <span>{getDenseLayerCount()}</span>
        </div>
        <div className="architecture-item">
          <label>Hidden Layer Sizes</label>
          <span>{getHiddenLayerSizes()}</span>
        </div>
        <div className="architecture-item">
          <label>Learning Rate</label>
          <span>{config.learningRate}</span>
        </div>
        <div className="architecture-item">
          <label>Batch Size</label>
          <span>{config.batchSize}</span>
        </div>
        <div className="architecture-item">
          <label>L1 Penalty</label>
          <span>{config.l1_penalty}</span>
        </div>
        <div className="architecture-item">
          <label>Dropout Rate</label>
          <span>{config.dropoutRate || config.dropout_rate || 'N/A'}</span>
        </div>
        <div className="architecture-item">
          <label>Epochs Trained</label>
          <span>{getActualEpochs()}</span>
        </div>
      </div>
    </div>
  );
}

export default ModelArchitectureDisplay; 