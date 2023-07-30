# AutoGPT for ReactJS apps

"Programmers have programmed themselves out of jobs" - Unknown

AutoGPT for ReactJS apps is an autonomous agent that builds applications just from prompts, following the Test-Driven Development (TDD) approach. It starts by writing tests for a feature, then implements the code to pass the tests, runs the tests and fixes any issues. It is capable of building an entire application step by step, iteratively. 

The agent has demonstrated its capability by successfully building a simple todo application all by itself, without any human intervention. However, like any AI agent, it does get lost once in a while and goes off the rails. The TDD approach mostly is able to keep it on track though. If it gets stuck, you can simply modify the prompt and restart the run.

With the system prompt, we are trying to teach GPT-4 how to build ReactJS applications the correct way. This is a work in progress and there is a lot of room for improvement.

This project is in a very early stage and serves as a basic demo. The idea is to illustrate the potential of autonomous coding agents. Within a couple of years, we should have code generation models much better than GPT-4. These agents will probably be able to write entire applications just with simple prompts one day. Imagine on-the-fly application generation! The possibilities are endless.

## Project Status

This project is in the very early alpha stage. Contributions are welcome!

## Setup Instructions

1. Setup a virtual environment:
```
python3 -m venv env
```

2. Activate the virtual environment:
- On macOS and Linux:
  ```
  source env/bin/activate
  ```
- On Windows:
  ```
  .\env\Scripts\activate
  ```
3. Clone the repository to your local machine:
```
git clone https://github.com/gimlet-ai/dev-gpt.git

```
4. Navigate to the project directory:
```
cd dev-gpt
```

5. Run the following command to install the package and its dependencies:
```
pip install -r requirements.txt
python setup.py install
```

6. Set up your GPT-4 and Anthropic API keys as environment variables.
```
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

7. Run dev-gpt
```
dev-gpt --prompt 'build a simple todo app'
```

The app should be built in the todo-app in the current directory.


## Contributing

We welcome contributions to this project. Please feel free to submit issues and pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source, under the [MIT license](LICENSE).

## Contact

If you have any questions or comments, please feel free to reach out to us on GitHub.
