import React, { useState } from 'react';
import TodoInput from './components/TodoInput';
import TodoList from './components/TodoList';

function App() {
  const [todos, setTodos] = useState([]);

  const addTodo = (newTodo) => {
    setTodos([...todos, newTodo]);
  };

  const deleteTodo = (todoToDelete) => {
    setTodos(todos.filter((todo) => todo !== todoToDelete));
  };

  return (
    <div className='App'>
      <TodoInput addTodo={addTodo} />
      <TodoList todos={todos} deleteTodo={deleteTodo} />
    </div>
  );
}

export default App;
