import React, { useState } from 'react';
import Accordion from './Accordion';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';

function AdvancedSettings({ onSplitChange, onSeedChange }) {
  const [splitRatios, setSplitRatios] = useState([70, 90]);
  const [seed, setSeed] = useState(42);

  const handleSplitChange = (values) => {
    setSplitRatios(values);
    const [trainVal, valTest] = values;
    const train = trainVal;
    const val = valTest - trainVal;
    const test = 100 - valTest;
    onSplitChange({ train, validation: val, test });
  };

  const handleSeedChange = (e) => {
    const newSeed = parseInt(e.target.value);
    setSeed(newSeed);
    onSeedChange(newSeed);
  };

  return (
    <Accordion title="Advanced">
      <Accordion title="Train-val-test split">
        <div style={{ margin: '20px 0' }}>
          <Slider
            range
            min={0}
            max={100}
            defaultValue={splitRatios}
            onChange={handleSplitChange}
          />
          <p>Training {splitRatios[0]}% Validation {splitRatios[1] - splitRatios[0]}% Test {100 - splitRatios[1]}%</p>
        </div>
        <div>
          <label htmlFor="seed">Random Seed: </label>
          <input
            type="number"
            id="seed"
            value={seed}
            onChange={handleSeedChange}
            min={0}
          />
        </div>
      </Accordion>
    </Accordion>
  );
}

export default AdvancedSettings;
