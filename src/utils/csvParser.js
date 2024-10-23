import Papa from 'papaparse';

export function parseCSV(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      complete: (results) => {
        if (results.errors.length > 0) {
          reject(results.errors);
        } else {
          resolve(results.data);
        }
      },
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
    });
  });
}
