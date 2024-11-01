import _lodash from 'lodash';
import seedrandom from 'seedrandom';

export function preprocessData(data, targetColumn, targetType, splitRatios, seed) {
  // Set the random seed
  seedrandom(seed, { global: true });
  const _ = _lodash.runInContext();

  // Apply stratified train-validation-test split
  const { train, validation, test } = stratifiedTrainValidationTestSplit(data, targetColumn, targetType, splitRatios.train, splitRatios.validation, seed);
  
  // Add split information to each row
  train.forEach(row => row.split = 'train');
  validation.forEach(row => row.split = 'validation');
  test.forEach(row => row.split = 'test');
  
  // Combine all data for preprocessing
  const allData = [...train, ...validation, ...test];
  
  // Analyze and preprocess the training data
  let updatedColumns = Object.keys(data[0]);
  
  // Discard columns with all same values
  updatedColumns = updatedColumns.filter(col => {
    const uniqueValues = _.uniq(allData.map(row => row[col]));
    return uniqueValues.length > 1;
  });
  
  // Detect string columns
  const stringColumns = updatedColumns.filter(col => {
    const nonNullValues = allData.map(row => row[col]).filter(val => val !== null && val !== '');
    return typeof nonNullValues[0] === 'string' && col !== targetColumn;
  });

  // Get unique target classes for classification tasks
  let targetClasses = [];
  if (targetType !== 'regression') {
    targetClasses = [...new Set(allData.map(row => row[targetColumn]))].sort();
  }
  
  const K = targetType === 'regression' ? 2 : targetClasses.length;

  // Check for granularity and filter out highly granular columns
  const nonGranularColumns = stringColumns.filter(col => {
    const valueCounts = _.countBy(allData, col);
    const totalCount = allData.length;
    return !Object.values(valueCounts).every(count => count / totalCount < 0.1);
  });

  // Update updatedColumns to remove granular columns
  updatedColumns = updatedColumns.filter(col => !stringColumns.includes(col) || nonGranularColumns.includes(col));

  // Process non-granular string columns
  nonGranularColumns.forEach(col => {
    // Skip the split column
    if (col === 'split') return;
    
    // Fill missing values with 'Unknown'
    allData.forEach(row => { if (row[col] === null || row[col] === '') row[col] = 'Unknown'; });
    
    // Get unique categories for the column
    const categories = [...new Set(allData.map(row => row[col]))];
    const d = categories.length;

    if (d < K + 5) {
      // Apply one-hot encoding
      const encodedColumns = applyOneHotEncoding(col, categories, allData);
      updatedColumns = updatedColumns.filter(c => c !== col);
      updatedColumns.push(...encodedColumns);
    } else {
      // Apply target encoding
      if (targetType === 'regression') {
        applyRegressionTargetEncoding(col, allData, targetColumn, _);
        updatedColumns = updatedColumns.filter(c => c !== col);
        updatedColumns.push(`${col}_encoded`);
      } else {
        applyClassificationTargetEncoding(col, allData, targetColumn, targetClasses, _);
        updatedColumns = updatedColumns.filter(c => c !== col);
        targetClasses.slice(1).forEach(cls => {
          updatedColumns.push(`${col}_encoded_${cls}`);
        });
      }
    }
  });
  
  // Also exclude 'split' from numeric columns
  const numericColumns = updatedColumns.filter(col => {
    if (col === 'split') return false;
    const nonNullValues = allData.map(row => row[col]).filter(val => val !== null && val !== '' && !isNaN(val));
    return typeof nonNullValues[0] === 'number' && col !== targetColumn;
  });
  
  numericColumns.forEach(col => {
    const validValues = allData.map(row => row[col]).filter(val => val !== null && val !== '' && !isNaN(val));
    const mean = _.mean(validValues);
    
    allData.forEach(row => { 
      if (row[col] === null || row[col] === '' || isNaN(row[col])) {
        row[col] = mean; 
      }
    });
  });
  
  // Apply Standard Scaler to numeric columns
  const scaler = {};
  numericColumns.forEach(col => {
    const values = allData.map(row => row[col]);
    const mean = _.mean(values);
    const std = Math.sqrt(_.sum(values.map(v => Math.pow(v - mean, 2))) / values.length);
    scaler[col] = { mean, std };
    
    allData.forEach(row => {
      row[col] = (row[col] - mean) / std;
    });
  });
  
  // Process target column for classification tasks
  let classMapping = null;
  if (targetType === 'binary' || targetType === 'multiclass') {
    const uniqueClasses = [...new Set(allData.map(row => row[targetColumn]))];
    classMapping = Object.fromEntries(uniqueClasses.map((cls, index) => [cls, index]));
    
    // Apply class mapping to all datasets
    allData.forEach(row => {
      row[targetColumn] = classMapping[row[targetColumn]] ?? -1;
    });
  }
  
  // Make sure 'split' is included in updatedColumns
  if (!updatedColumns.includes('split')) {
    updatedColumns.push('split');
  }

  // Split the processed data back into train/validation/test
  const processedTrain = allData.filter(row => row.split === 'train');
  const processedValidation = allData.filter(row => row.split === 'validation');
  const processedTest = allData.filter(row => row.split === 'test');

  return {
    trainData: processedTrain,
    validationData: processedValidation,
    testData: processedTest,
    updatedColumns,
    scaler,
    classMapping
  };
}

function applyOneHotEncoding(col, categories, allData) {
  const categoriesToEncode = categories.slice(0, -1); // Omit the last category
  allData.forEach(row => {
    categoriesToEncode.forEach(category => {
      row[`${col}_${category}`] = row[col] === category ? 1 : 0;
    });
    delete row[col];
  });
  return categoriesToEncode.map(category => `${col}_${category}`);
}

function applyRegressionTargetEncoding(col, allData, targetColumn, _) {
  const globalAverage = _.mean(allData.map(row => row[targetColumn]));
  const encodedColumn = `${col}_encoded`;
  
  const shuffledAllData = _.shuffle(allData);
  const encodingMap = {};
  const categoryCounts = {};
  
  shuffledAllData.forEach(row => {
    const category = row[col];
    if (!encodingMap[category]) {
      encodingMap[category] = globalAverage;
      categoryCounts[category] = 1;
    } else {
      const count = categoryCounts[category];
      const currentSum = encodingMap[category] * count;
      encodingMap[category] = (currentSum + row[targetColumn] + globalAverage) / (count + 2);
      categoryCounts[category]++;
    }
    row[encodedColumn] = encodingMap[category];
  });
  
  allData.forEach(row => {
    row[encodedColumn] = encodingMap[row[col]] || globalAverage;
  });
}

function applyClassificationTargetEncoding(col, allData, targetColumn, targetClasses, _) {
  const globalProportions = targetClasses.reduce((acc, cls) => {
    acc[cls] = allData.filter(row => row[targetColumn] === cls).length / allData.length;
    return acc;
  }, {});

  const shuffledAllData = _.shuffle(allData);
  const encodingMaps = {};
  const categoryCounts = {};

  targetClasses.slice(1).forEach(cls => {
    encodingMaps[cls] = {};
    categoryCounts[cls] = {};
  });

  shuffledAllData.forEach(row => {
    const category = row[col];
    targetClasses.slice(1).forEach(cls => {
      const encodedColumn = `${col}_encoded_${cls}`;
      if (!encodingMaps[cls][category]) {
        encodingMaps[cls][category] = globalProportions[cls];
        categoryCounts[cls][category] = 1;
      } else {
        const count = categoryCounts[cls][category];
        const currentSum = encodingMaps[cls][category] * count;
        const isTargetClass = row[targetColumn] === cls ? 1 : 0;
        encodingMaps[cls][category] = (currentSum + isTargetClass + globalProportions[cls]) / (count + 2);
        categoryCounts[cls][category]++;
      }
      row[encodedColumn] = encodingMaps[cls][category];
    });
  });

  allData.forEach(row => {
    targetClasses.slice(1).forEach(cls => {
      const encodedColumn = `${col}_encoded_${cls}`;
      row[encodedColumn] = encodingMaps[cls][row[col]] || globalProportions[cls];
    });
  });
}

function stratifiedTrainValidationTestSplit(data, targetColumn, targetType, trainRatio, validationRatio, seed) {
  let strata;
  let auxColumn = null;

  // Create a new lodash instance with seeded random
  seedrandom(seed, { global: true });
  const _ = _lodash.runInContext();

  if (targetType === 'regression') {
    // Create 10 bins for regression targets
    const targetValues = data.map(row => row[targetColumn]).filter(val => val !== null && !isNaN(val));
    const min = Math.min(...targetValues);
    const max = Math.max(...targetValues);
    const binSize = (max - min) / 10;
    
    auxColumn = targetColumn + '_temp_strat_bin';
    data.forEach(row => {
      const value = row[targetColumn];
      if (value === null || isNaN(value)) {
        row[auxColumn] = -1; // Assign -1 for missing or invalid values
      } else {
        row[auxColumn] = Math.min(Math.floor((value - min) / binSize), 9);
      }
    });
    strata = auxColumn;
  } else {
    // For classification, use the target column directly
    strata = targetColumn;
  }

  // Group data by strata
  const groupedData = _.groupBy(data, strata);

  // Initialize result sets
  const train = [];
  const validation = [];
  const test = [];

  // Stratified sampling
  Object.values(groupedData).forEach(stratum => {
    const shuffled = _.shuffle(stratum);
    const trainSize = Math.floor(shuffled.length * trainRatio);
    const validationSize = Math.floor(shuffled.length * validationRatio);

    train.push(...shuffled.slice(0, trainSize));
    validation.push(...shuffled.slice(trainSize, trainSize + validationSize));
    test.push(...shuffled.slice(trainSize + validationSize));
  });

  // Remove auxiliary column if it was created
  if (auxColumn) {
    train.forEach(row => delete row[auxColumn]);
    validation.forEach(row => delete row[auxColumn]);
    test.forEach(row => delete row[auxColumn]);
  }

  return { train, validation, test };
}
