# AutoGPT for Web App Development 

"Programmers have programmed themselves out of jobs" - Unknown

AutoGPT for web app development is an autonomous agent that builds ReactJS, Flask, Express, etc. applications just from prompts, following the Test-Driven Development (TDD) approach. It starts by writing tests for a feature, then implements the code to pass the tests, runs the tests and fixes any issues. It is capable of building an entire application step by step, iteratively. 

The agent has demonstrated its capability by successfully building a simple todo application all by itself, without any human intervention. However, like any AI agent, it does get lost once in a while and goes off the rails. The TDD approach mostly is able to keep it on track though. If it gets stuck, you can simply modify the prompt and restart the run.

This project is in early alpha stage. The idea is to illustrate the potential of autonomous coding agents. Soon, we should have code generation models much better than GPT-4 and improved planning abilities. These agents will probably be able to write complex applications just with simple prompts within hours! 

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
dev-gpt --prompt 'build a ReactJS todo app with the following features: add a todo, mark it as complete and rename it'
```

The app should be built in the todo-app in the current directory.

## Similar Projects

- [Aider](https://github.com/paul-gauthier/aider)
- [Smol Developer](https://github.com/smol-ai/developer)
- [GPT-Engineer](https://github.com/AntonOsika/gpt-engineer)
- [Mentat](https://github.com/biobootloader/mentat)


## Contributing

We welcome contributions to this project. Please feel free to submit issues and pull requests. For major changes, please open an issue first to discuss what you would like to change.


## License

This project is open source, under the [MIT license](LICENSE).

## Contact

If you have any questions or comments, please feel free to reach out to us on GitHub.
