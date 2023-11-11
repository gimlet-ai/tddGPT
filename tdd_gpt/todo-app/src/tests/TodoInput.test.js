import { render, screen, fireEvent } from '@testing-library/react';
import TodoInput from '../components/TodoInput';

test('renders the todo input field', () => {
  render(<TodoInput />);
  const inputElement = screen.getByPlaceholderText(/add a new task here.../i);
  expect(inputElement).toBeInTheDocument();
});

test('allows typing in the todo input field', () => {
  render(<TodoInput />);
  const inputElement = screen.getByPlaceholderText(/add a new task here.../i);
  fireEvent.change(inputElement, { target: { value: 'New Todo' } });
  expect(inputElement.value).toBe('New Todo');
});

test('clears the input field when a todo is submitted', () => {
  const mockAddTodo = jest.fn();
  render(<TodoInput addTodo={mockAddTodo} />);
  const inputElement = screen.getByPlaceholderText(/add a new task here.../i);
  const buttonElement = screen.getByRole('button', { name: /add/i });
  fireEvent.change(inputElement, { target: { value: 'New Todo' } });
  fireEvent.click(buttonElement);
  expect(inputElement.value).toBe('');
  expect(mockAddTodo).toHaveBeenCalledWith('New Todo');
});