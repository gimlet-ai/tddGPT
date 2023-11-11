import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';

test('allows users to add items to the todo list', () => {
  render(<App />);
  const inputElement = screen.getByPlaceholderText(/add a new task here.../i);
  const buttonElement = screen.getByRole('button', { name: /add/i });
  fireEvent.change(inputElement, { target: { value: 'New Todo' } });
  fireEvent.click(buttonElement);
  const todoElement = screen.getByText('New Todo');
  expect(todoElement).toBeInTheDocument();
});

test('allows users to delete items from the todo list', () => {
  render(<App />);
  const inputElement = screen.getByPlaceholderText(/add a new task here.../i);
  const buttonElement = screen.getByRole('button', { name: /add/i });
  fireEvent.change(inputElement, { target: { value: 'New Todo' } });
  fireEvent.click(buttonElement);
  const deleteButtonElement = screen.getByRole('button', { name: /delete/i });
  fireEvent.click(deleteButtonElement);
  expect(screen.queryByText('New Todo')).toBeNull();
});