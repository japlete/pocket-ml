import React from 'react';

function TargetSelector({ columns, selectedTarget, onTargetChange }) {
  return (
    <div>
      <label htmlFor="target-selector">Select target column: </label>
      <select
        id="target-selector"
        value={selectedTarget}
        onChange={(e) => onTargetChange(e.target.value)}
      >
        <option value="">-- Select a column --</option>
        {columns.map((column) => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </select>
    </div>
  );
}

export default TargetSelector;
