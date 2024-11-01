import React, { useState, useEffect } from 'react';
import { getAllModels, deleteModel } from '../utils/indexedDB';

function SavedModelsList() {
  const [models, setModels] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

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

  const handleDelete = async (modelName) => {
    if (window.confirm(`Are you sure you want to delete the model "${modelName}"?`)) {
      try {
        setIsDeleting(true);
        await deleteModel(modelName);
        // Refresh the list after deletion
        await loadModels();
      } catch (err) {
        console.error('Error deleting model:', err);
        setError('Failed to delete model');
      } finally {
        setIsDeleting(false);
      }
    }
  };

  if (isLoading) {
    return <div>Loading saved models...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  if (models.length === 0) {
    return <div>No saved models found</div>;
  }

  return (
    <div className="saved-models-list">
      <h3>Saved Models</h3>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Saved Date</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.name}>
              <td>{model.name}</td>
              <td>{model.config.targetType}</td>
              <td>{new Date(model.timestamp).toLocaleString()}</td>
              <td>
                <button
                  onClick={() => handleDelete(model.name)}
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
  );
}

export default SavedModelsList; 