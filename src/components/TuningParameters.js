import React from 'react';

function TuningParameters({ 
  maxTrainingTime, 
  onMaxTrainingTimeChange, 
  minIterations, 
  onMinIterationsChange 
}) {
  const handleMaxTrainingTimeChange = (e) => {
    const value = parseInt(e.target.value);
    if (value > 0) {
      onMaxTrainingTimeChange(value);
    } else {
      onMaxTrainingTimeChange(30); // Default to 30 minutes if invalid
    }
  };

  const handleMinIterationsChange = (e) => {
    const value = parseInt(e.target.value);
    if (value > 0) {
      onMinIterationsChange(value);
    } else {
      onMinIterationsChange(10); // Default to 10 iterations if invalid
    }
  };

  return (
    <div className="tuning-parameters">
      <div>
        <label htmlFor="max-training-time">Maximum Training Time (minutes): </label>
        <input
          type="number"
          id="max-training-time"
          value={maxTrainingTime}
          onChange={handleMaxTrainingTimeChange}
          min="1"
          step="5"
        />
      </div>
      <div>
        <label htmlFor="min-iterations">Minimum Tuning Iterations: </label>
        <input
          type="number"
          id="min-iterations"
          value={minIterations}
          onChange={handleMinIterationsChange}
          min="1"
          step="1"
        />
      </div>
    </div>
  );
}

export default TuningParameters;
