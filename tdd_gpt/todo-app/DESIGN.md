# Todo App Design Document

## Components:

### App
- The main container for the todo application.
- Manages the state of the todo list.
- Renders the TodoList and TodoInput components.

### TodoList
- Displays a list of todos.
- Accepts an array of todo items as props.
- Renders individual TodoItem components.

### TodoItem
- Represents a single todo item.
- Accepts a todo object as a prop.
- Displays the todo text.
- Provides a way to mark a todo as completed and to delete a todo.

### TodoInput
- Allows the user to input a new todo.
- Contains a form with an input field and a submit button.

## State Management:
- The App component will hold the state of the todo list in an array.
- TodoInput will have local state for the input field.

## Data Flow:
- Unidirectional data flow from App to TodoList and TodoInput.
- Callbacks will be passed to TodoItem and TodoInput to handle events like adding or removing todos.

## Styling:
- The application will be styled to be visually appealing and user-friendly.
- Responsive design to accommodate different screen sizes.

## Testing:
- Unit tests will be written for each component before implementation.
- Tests will be located in the src/tests/ directory, except for the main App tests which will be in the src/ directory.

This document will be updated as the project evolves.