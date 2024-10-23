import React from 'react';

function TargetTypeSelector({ targetType, suggestedType, onTargetTypeChange }) {
  return (
    <div>
      <label htmlFor="target-type-selector">Target type: </label>
      <select
        id="target-type-selector"
        value={targetType}
        onChange={(e) => onTargetTypeChange(e.target.value)}
      >
        <option value="regression">Regression</option>
        <option value="binary">Binary Classification</option>
        <option value="multiclass">Multiclass Classification</option>
      </select>
      {targetType === suggestedType && <span> (autodetected)</span>}
    </div>
  );
}

export default TargetTypeSelector;
