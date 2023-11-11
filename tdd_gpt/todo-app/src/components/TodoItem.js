import React from 'react';

function TodoItem({ todo, deleteTodo }) {
  const [completed, setCompleted] = React.useState(false);

  const handleCheckboxChange = () => {
    setCompleted(!completed);
  };

  return (
    <div>
      <input
        type='checkbox'
        checked={completed}
        onChange={handleCheckboxChange}
        aria-label={`Mark ${todo} as completed`}
      />
      <span style={{ textDecoration: completed ? 'line-through' : 'none' }}>
        {todo}
      </span>
      <button onClick={() => deleteTodo(todo)}>Delete</button>
    </div>
  );
}

export default TodoItem;