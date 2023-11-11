import { render, screen } from '@testing-library/react';
import TodoList from '../components/TodoList';
import TodoItem from '../components/TodoItem';

test('renders an empty list when no todos are provided', () => {
  render(<TodoList todos={[]} />);
  const listElement = screen.getByRole('list');
  expect(listElement).toBeInTheDocument();
  expect(listElement).toBeEmptyDOMElement();
});

test('renders a list of todos', () => {
  const todos = ['Buy milk', 'Walk the dog'];
  render(<TodoList todos={todos} />);
  const todoItems = screen.getAllByRole('listitem');
  expect(todoItems.length).toBe(todos.length);
  todos.forEach((todo, index) => {
    expect(screen.getByText(todo)).toBeInTheDocument();
  });
});