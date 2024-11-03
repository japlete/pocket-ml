import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { getAllModels, deleteModel } from '../utils/indexedDB';
import ModelArchitectureDisplay from './ModelArchitectureDisplay';

function SavedModelsList() {
  const [models, setModels] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loadedTfModel, setLoadedTfModel] = useState(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setIsLoading(true);
      const savedModels = await getAllModels();
      setModels(savedModels);
      setError(null);
    } catch (err) {
      console.error('Error loading models:', err);
      setError('Failed to load saved models');
    } finally {
      setIsLoading(false);
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

  const getMetricValue = (model) => {
    const metricKey = model.config.primaryMetric.toUpperCase();
    return model.results?.validation[metricKey]?.toFixed(4) || 'N/A';
  };

  const handleDelete = async (modelName) => {
    if (window.confirm(`Are you sure you want to delete the model "${modelName}"?`)) {
      try {
        setIsDeleting(true);
        await deleteModel(modelName);
        await loadModels();
      } catch (err) {
        console.error('Error deleting model:', err);
        setError('Failed to delete model');
      } finally {
        setIsDeleting(false);
      }
    }
  };

  const handleRowClick = async (model) => {
    if (selectedModel?.name === model.name) {
      setSelectedModel(null);
      setLoadedTfModel(null);
      return;
    }

    setSelectedModel(model);
    setIsLoadingModel(true);
    
    try {
      const tfModel = await tf.loadLayersModel(`indexeddb://${model.name}`);
      setLoadedTfModel(tfModel);
    } catch (err) {
      console.error('Error loading model:', err);
      setError(`Failed to load model architecture for ${model.name}`);
      setLoadedTfModel(null);
    } finally {
      setIsLoadingModel(false);
    }
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return (
      <div className="date-cell">
        <div>{date.toLocaleDateString()}</div>
        <div className="time">{date.toLocaleTimeString()}</div>
      </div>
    );
  };

  if (isLoading) {
    return <div>Loading saved models...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  if (models.length === 0) {
    return <div className="empty-models-message">Your saved models will appear here</div>;
  }

  return (
    <div className="saved-models-list">
      <h3>Saved Models</h3>
      <div className="models-container">
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Target Type</th>
                <th>Primary Metric</th>
                <th>Value</th>
                <th>Saved</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr 
                  key={model.name}
                  onClick={() => handleRowClick(model)}
                  className={selectedModel?.name === model.name ? 'selected-row' : ''}
                >
                  <td>{model.name}</td>
                  <td>{model.config.targetType}</td>
                  <td>{formatMetricName(model.config.primaryMetric)}</td>
                  <td className="metric-value">{getMetricValue(model)}</td>
                  <td>{formatDate(model.timestamp)}</td>
                  <td>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(model.name);
                      }}
                      className="delete-model-button"
                      disabled={isDeleting}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {selectedModel && (
          <div className="model-details">
            <h4>Model Details: {selectedModel.name}</h4>
            {isLoadingModel ? (
              <div className="loading-indicator">Loading model architecture...</div>
            ) : (
              <ModelArchitectureDisplay
                model={loadedTfModel}
                config={{
                  ...selectedModel.config,
                  metadata: selectedModel.metadata
                }}
                featureCount={selectedModel.metadata?.featureCount || 'N/A'}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default SavedModelsList; 