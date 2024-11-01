import React from 'react';

function Accordion({ title, children, defaultOpen = false, isOpen, onToggle }) {
  // Use controlled state if provided, otherwise use internal state
  const [internalIsOpen, setInternalIsOpen] = React.useState(defaultOpen);
  
  const isControlled = isOpen !== undefined && onToggle !== undefined;
  const shown = isControlled ? isOpen : internalIsOpen;
  
  const handleToggle = () => {
    if (isControlled) {
      onToggle(!isOpen);
    } else {
      setInternalIsOpen(!internalIsOpen);
    }
  };

  return (
    <div className="accordion">
      <button onClick={handleToggle} className="accordion-toggle">
        {title} {shown ? '▲' : '▼'}
      </button>
      {shown && <div className="accordion-content">{children}</div>}
    </div>
  );
}

export default Accordion;
