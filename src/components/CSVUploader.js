import React, { useState } from 'react';
import { parseCSV } from '../utils/csvParser.js';

function CSVUploader({ onDataParsed }) {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (file) {
      try {
        const parsedData = await parseCSV(file);
        onDataParsed(parsedData);
      } catch (error) {
        console.error('Error parsing CSV:', error);
        // TODO: Add error handling UI
      }
    }
  };

  return (
    <div>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={!file}>
        Upload CSV
      </button>
    </div>
  );
}

export default CSVUploader;
