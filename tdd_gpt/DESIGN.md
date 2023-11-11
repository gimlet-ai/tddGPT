# Application Design for Todo App

## Components

- `App`: The main application component that renders the todo list and input form.
- `TodoList`: A component that displays the list of todos.
- `TodoItem`: A component for each todo item in the list, with a checkbox to toggle completion and a delete button.
- `AddTodoForm`: A form component to input and submit new todos.

## State Management

- The `App` component will hold the state for the list of todos and pass down the necessary props to child components.

## Pseudocode

### App Component

- State: `todos` array
- Render `AddTodoForm` and `TodoList`

### AddTodoForm Component

- Input field for new todo
- Submit button to add todo
- OnSubmit: Add new todo to the `App` state

### TodoList Component

- Accepts `todos` array as prop
- Maps over `todos` array and renders `TodoItem` components

### TodoItem Component

- Accepts `todo` object and `onDelete` callback as props
- Checkbox to toggle completion status
- Delete button to remove todo

## File Structure

- src/
  - components/
    - App.js
    - TodoList.js
    - TodoItem.js
    - AddTodoForm.js
  - tests/
    - TodoList.test.js
    - TodoItem.test.js
    - AddTodoForm.test.js

## Notes

- Use functional components and React hooks for state and lifecycle management.
- Ensure components are reusable and maintainable.
- Follow TDD by writing tests before implementing functionality.