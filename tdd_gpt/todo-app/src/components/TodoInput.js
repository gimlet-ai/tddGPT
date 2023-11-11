import React, { useState } from 'react';

function TodoInput({ addTodo }) {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    if (inputValue.trim()) {
      addTodo(inputValue);
      setInputValue('');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        placeholder='Add a new task here...'
        aria-label='Todo input field'
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
      />
      <button type='submit'>Add</button>
    </form>
  );
}

export default TodoInput;