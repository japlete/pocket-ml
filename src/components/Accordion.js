import React, { useState } from 'react';

function Accordion({ title, children }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="accordion">
      <button onClick={() => setIsOpen(!isOpen)} className="accordion-toggle">
        {title} {isOpen ? '▲' : '▼'}
      </button>
      {isOpen && <div className="accordion-content">{children}</div>}
    </div>
  );
}

export default Accordion;
