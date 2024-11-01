const DB_NAME = 'PocketMLDB';
const DB_VERSION = 1;
const MODELS_STORE = 'models';

export async function initDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(MODELS_STORE)) {
        const store = db.createObjectStore(MODELS_STORE, { keyPath: 'name' });
        store.createIndex('name', 'name', { unique: true });
        store.createIndex('timestamp', 'timestamp', { unique: false });
      }
    };
  });
}

export async function saveModel(modelData) {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([MODELS_STORE], 'readwrite');
    const store = transaction.objectStore(MODELS_STORE);

    // Add timestamp to the model data
    const modelToSave = {
      ...modelData,
      timestamp: new Date().toISOString()
    };

    const request = store.add(modelToSave);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

export async function checkModelNameExists(name) {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([MODELS_STORE], 'readonly');
    const store = transaction.objectStore(MODELS_STORE);
    const request = store.get(name);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(!!request.result);
  });
}

// Add this function to get all saved models
export async function getAllModels() {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([MODELS_STORE], 'readonly');
    const store = transaction.objectStore(MODELS_STORE);
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      // Sort models by timestamp, most recent first
      const models = request.result.sort((a, b) => 
        new Date(b.timestamp) - new Date(a.timestamp)
      );
      resolve(models);
    };
  });
}

// Add this function to delete a model
export async function deleteModel(modelName) {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([MODELS_STORE], 'readwrite');
    const store = transaction.objectStore(MODELS_STORE);
    const request = store.delete(modelName);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
} 