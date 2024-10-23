import _ from 'lodash';

export function preprocessData(data, targetColumn) {
  // 1. Apply train-test split
  const { train, test } = trainTestSplit(data, 0.8);
  
  // 2. Analyze and preprocess the training data
  let updatedColumns = Object.keys(data[0]);
  
  // 2.1 Discard columns with all same values
  updatedColumns = updatedColumns.filter(col => {
    const uniqueValues = _.uniq(train.map(row => row[col]));
    return uniqueValues.length > 1;
  });
  
  // 2.2 & 2.3 Process string columns
  const stringColumns = updatedColumns.filter(col => {
    const nonNullValues = train.map(row => row[col]).filter(val => val !== null && val !== '');
    return typeof nonNullValues[0] === 'string' && col !== targetColumn;
  });
  
  stringColumns.forEach(col => {
    // Fill missing values with 'Unknown'
    train.forEach(row => { if (row[col] === null || row[col] === '') row[col] = 'Unknown'; });
    test.forEach(row => { if (row[col] === null || row[col] === '') row[col] = 'Unknown'; });
    
    // Check for granularity
    const valueCounts = _.countBy(train, col);
    const totalCount = train.length;
    const isGranular = Object.values(valueCounts).every(count => count / totalCount < 0.1);
    
    if (isGranular) {
      updatedColumns = updatedColumns.filter(c => c !== col);
    } else {
      // 2.4 Apply target encoding
      const targetEncoding = _.mapValues(
        _.groupBy(train, col),
        group => _.meanBy(group, targetColumn)
      );
      
      train.forEach(row => {
        row[`${col}_encoded`] = targetEncoding[row[col]] || 0;
        delete row[col]; // Remove the original string column
      });
      test.forEach(row => {
        row[`${col}_encoded`] = targetEncoding[row[col]] || 0;
        delete row[col]; // Remove the original string column
      });
      
      // Update the columns list
      updatedColumns = updatedColumns.filter(c => c !== col);
      updatedColumns.push(`${col}_encoded`);
    }
  });
  
  // 2.5 Fill missing values for numeric columns
  const numericColumns = updatedColumns.filter(col => {
    const nonNullValues = train.map(row => row[col]).filter(val => val !== null && val !== '' && !isNaN(val));
    return typeof nonNullValues[0] === 'number' && col !== targetColumn;
  });
  
  numericColumns.forEach(col => {
    const validValues = train.map(row => row[col]).filter(val => val !== null && val !== '' && !isNaN(val));
    const mean = _.mean(validValues);
    
    train.forEach(row => { 
      if (row[col] === null || row[col] === '' || isNaN(row[col])) {
        row[col] = mean; 
      }
    });
    test.forEach(row => { 
      if (row[col] === null || row[col] === '' || isNaN(row[col])) {
        row[col] = mean; 
      }
    });
  });
  
  // 2.6 Apply Standard Scaler to numeric columns
  const scaler = {};
  numericColumns.forEach(col => {
    const values = train.map(row => row[col]);
    const mean = _.mean(values);
    const std = Math.sqrt(_.sum(values.map(v => Math.pow(v - mean, 2))) / values.length);
    scaler[col] = { mean, std };
    
    train.forEach(row => {
      row[col] = (row[col] - mean) / std;
    });
    test.forEach(row => {
      row[col] = (row[col] - mean) / std;
    });
  });
  
  return {
    trainData: train,
    testData: test,
    updatedColumns,
    scaler
  };
}

function trainTestSplit(data, trainRatio) {
  const shuffled = _.shuffle(data);
  const trainSize = Math.floor(shuffled.length * trainRatio);
  return {
    train: shuffled.slice(0, trainSize),
    test: shuffled.slice(trainSize)
  };
}
