import { render, screen, fireEvent } from '@testing-library/react';
import TodoItem from '../components/TodoItem';

test('renders the todo text', () => {
  const todo = 'Buy milk';
  render(<TodoItem todo={todo} />);
  const todoElement = screen.getByText(todo);
  expect(todoElement).toBeInTheDocument();
});

test('provides a way to mark a todo as completed', () => {
  const todo = 'Buy milk';
  render(<TodoItem todo={todo} />);
  const checkbox = screen.getByRole('checkbox');
  fireEvent.click(checkbox);
  expect(checkbox).toBeChecked();
});

test('provides a way to delete a todo', () => {
  const todo = 'Buy milk';
  const mockDeleteTodo = jest.fn();
  render(<TodoItem todo={todo} deleteTodo={mockDeleteTodo} />);
  const deleteButton = screen.getByRole('button', { name: /delete/i });
  fireEvent.click(deleteButton);
  expect(mockDeleteTodo).toHaveBeenCalledWith(todo);
});