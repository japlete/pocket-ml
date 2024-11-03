import React from 'react';

function TargetTypeSelector({ targetType, suggestedType, onTargetTypeChange, allowedTypes }) {
  return (
    <div className="target-type-info">
      <div>
        <label htmlFor="target-type-selector">Target type: </label>
        <select
          id="target-type-selector"
          value={targetType}
          onChange={(e) => onTargetTypeChange(e.target.value)}
        >
          {allowedTypes.includes('regression') && <option value="regression">Regression</option>}
          {allowedTypes.includes('binary') && <option value="binary">Binary Classification</option>}
          {allowedTypes.includes('multiclass') && <option value="multiclass">Multiclass Classification</option>}
        </select>
        {targetType === suggestedType && <span className="autodetected">(autodetected)</span>}
      </div>
    </div>
  );
}

export default TargetTypeSelector;
