import React from 'react';
import _ from 'lodash';

function DataPreview({ data, columns, showDownload = false, originalFileName = '' }) {
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

  const handleDownload = () => {
    // Generate download filename based on original filename if available
    const downloadFileName = originalFileName ? 
      originalFileName.replace('.csv', '_preprocessed.csv') : 
      'preprocessed_data.csv';

    // Convert data to CSV
    const csvContent = [
      columns.join(','), // Header row
      ...data.map(row => columns.map(col => {
        const value = row[col];
        // Handle values that might need escaping
        if (typeof value === 'string' && value.includes(',')) {
          return `"${value}"`;
        }
        return value;
      }).join(','))
    ].join('\n');

    // Create and trigger download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', downloadFileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="data-preview-container">
      {showDownload && (
        <button onClick={handleDownload} className="download-button">
          Download CSV
        </button>
      )}
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
