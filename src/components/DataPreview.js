import React from 'react';
import _ from 'lodash';

function DataPreview({ data, columns }) {
  if (!data || data.length === 0) {
    return <p>No data to display</p>;
  }

  const getColumnType = (columnName) => {
    const sampleValues = data.slice(0, 100).map(row => row[columnName]);
    const nonNullValues = sampleValues.filter(val => val !== null && val !== '');
    
    if (nonNullValues.length === 0) return 'unknown (all null/empty)';
    
    const firstNonNullValue = nonNullValues[0];
    
    if (typeof firstNonNullValue === 'number') return 'number';
    if (typeof firstNonNullValue === 'string') return 'string';
    if (typeof firstNonNullValue === 'boolean') return 'boolean';
    
    return 'unknown';
  };

  const formatValue = (value) => {
    if (typeof value === 'number') {
      return value.toPrecision(3); // Display 3 significant digits for numbers
    }
    return value === null ? 'null' : value;
  };

  return (
    <div>
      <table>
        <thead>
          <tr>
            {columns.map((header) => (
              <th key={header}>
                {header}
                <br />
                <small>({getColumnType(header)})</small>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 5).map((row, index) => (
            <tr key={index}>
              {columns.map((header) => (
                <td key={`${index}-${header}`}>{formatValue(row[header])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default DataPreview;
