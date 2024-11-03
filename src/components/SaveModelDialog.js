import React, { useState } from 'react';
import { saveModel, checkModelNameExists } from '../utils/indexedDB';

function SaveModelDialog({ model, modelResults, config, onClose }) {
  const [modelName, setModelName] = useState('');
  const [error, setError] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async (e) => {
    e.preventDefault();
    
    if (!modelName.trim()) {
      setError('Please enter a model name');
      return;
    }

    try {
      setIsSaving(true);
      
      // Check if name already exists
      const exists = await checkModelNameExists(modelName);
      if (exists) {
        setError('A model with this name already exists');
        setIsSaving(false);
        return;
      }

      // Get the number of epochs trained from model history
      const epochsTrained = model.history?.epoch?.length || 0;

      // Prepare model data for saving
      const modelData = {
        name: modelName,
        model: await model.save(`indexeddb://${modelName}`),
        results: modelResults,
        config: {
          ...config,
          dropoutRate: config.dropoutRate || config.dropout_rate // Handle both naming conventions
        },
        metadata: {
          savedAt: new Date().toISOString(),
          targetType: config.targetType,
          primaryMetric: config.primaryMetric,
          featureCount: model.inputs[0].shape[1],
          epochsTrained: epochsTrained // Save epochs trained in metadata
        }
      };

      await saveModel(modelData);
      onClose(true);
    } catch (err) {
      console.error('Error saving model:', err);
      setError('Failed to save model: ' + err.message);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h3>Save Model</h3>
        <form onSubmit={handleSave}>
          <div>
            <label htmlFor="modelName">Model Name:</label>
            <input
              type="text"
              id="modelName"
              value={modelName}
              onChange={(e) => {
                setModelName(e.target.value);
                setError('');
              }}
              placeholder="Enter a unique name"
              disabled={isSaving}
            />
          </div>
          {error && <p className="error-message">{error}</p>}
          <div className="modal-buttons">
            <button type="submit" disabled={isSaving}>
              {isSaving ? 'Saving...' : 'Save'}
            </button>
            <button type="button" onClick={() => onClose(false)} disabled={isSaving}>
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default SaveModelDialog; 