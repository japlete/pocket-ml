import React, { useState } from 'react';

function Accordion({ title, children, defaultOpen = false }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

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
