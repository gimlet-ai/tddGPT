# AutoGPT for ReactJS apps

"Programmers have programmed themselves out of jobs" - Unknown

## Project Status

This project is in the very early alpha stage. Contributions are welcome!

## About

AutoGPT for ReactJS apps is an autonomous agent that builds applications just from prompts, following the Test-Driven Development (TDD) approach. It is capable of building an entire application step by step, iteratively. 

The agent has demonstrated its capability by successfully building a simple todo application all by itself, without any human intervention. However, like any AI agent, it does get lost once in a while and goes off the rails. If it gets stuck, you can simply modify the prompt and restart the run. The prompts can probably be improved a lot to guide the agent more effectively.

With the system prompt, we are trying to teach GPT-4 how to build ReactJS applications the correct way. This is a work in progress and there is a lot of room for improvement.

This project is in a very early stage and serves as a basic demo. The idea is to illustrate the potential of autonomous coding agents. Within a couple of years, we should have code generation models much better than GPT-4. These agents will probably be able to write entire applications just with simple prompts one day. Imagine on-the-fly application generation! The possibilities are endless.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run `pip install .` to install the package.
4. Set up your GPT-4 and Anthropic API keys as environment variables.
5. Run the command `dev-gpt --prompt "Your prompt here"` to start the program.

## Usage Examples

Try it with "build a simple todo app" prompt.

```shell
dev-gpt --prompt "Build a simple todo app"
```

## Contributing

We welcome contributions to this project. Please feel free to submit issues and pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source, under the [MIT license](LICENSE).

## Contact

If you have any questions or comments, please feel free to reach out to us on GitHub.
