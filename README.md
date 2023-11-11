# Autonomous Agent for Test-Driven Development (TDD)

"Programmers have programmed themselves out of jobs" - Unknown

[![tddGPT: ReactJS Counter App with GPT3.5](https://cdn.loom.com/sessions/thumbnails/7f56ab1b478049baa299813c223526bd-with-play.gif)](https://www.loom.com/share/7f56ab1b478049baa299813c223526bd)

tddGPT is an autonomous coding agent that builds applications in ReactJS, Flask, Express, and more, all while adhering to the Test-Driven Development (TDD) methodology. It operates entirely without human intervention. Beginning with a project plan, tddGPT translates requirements into tests, develops code based on those tests, and debugs until all tests pass. Currently, it can build simple CRUD apps. The TDD framework keeps the agent focused and goal-oriented.

The core architecture is elegantly simple, utilizing just three tools: CLI, ReadFile, and WriteFile. It has been adpated from Langchain's AutoGPT example. Most enhancements were performed by GPT-4 itself over the course of a month-long chat interaction. I initially aimed to test the boundaries of GPT-4's capabilities in building ReactJS apps, and was successful in teaching it to construct applications step by step. In the process, it gained an understanding of temporal concepts like past, present, and future, as well as cause and effect.

The agent is not just a code generator; it’s also a learner. It evaluates its mistakes and areas for improvement as a final step, and some of these insights have already been incorporated into its operating prompts.

This project is in early alpha stage. GPT-4 API key is required. 

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
git clone https://github.com/sankethchebbi/tddGPT.git

```
4. Navigate to the project directory:
```
cd tddGPT
```

5. Run the following command to install the package and its dependencies:
```
python setup.py install
```

6. Set up your GPT-4 API keys as environment variables.
```
export OPENAI_API_KEY="sk-..."
```

7. Run the main.py
```
cd tdd-gpt
python main.py --model gpt-4-1106-preview or gpt-3.5-turbo --prompt "Your prompt here" --temperature 0.2 --context_window 128000
```

Check the counter-app directory for the generated app.

## Example apps

The following are some apps have been built by this agent.

- [Todo App](https://todo-gpt4r2.netlify.app/) - built with GPT4 Turbo
- [Task Tracker](https://brilliant-biscotti-3f9e48.netlify.app/) - built with GPT4
- [Counter App](https://counter-app-tddgpt.netlify.app/) - built with fine-tuned GPT3.5

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
