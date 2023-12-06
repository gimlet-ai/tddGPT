import os
import platform
import time
import json
from typing import Any, Callable, List, Optional, Tuple

from pydantic import BaseModel

from langchain.prompts.chat import BaseChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever
from summarizer import TextSummarizer


class TddGPTPrompt(BaseChatPromptTemplate, BaseModel):
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4096
    output_dir: Optional[str] = None  

    @property
    def summarizer(self) -> TextSummarizer:
        return TextSummarizer(summary_type="memory")

    def construct_full_prompt(self, goals: List[str]) -> str:
        os_name = 'MacOS' if platform.system() == 'Darwin' else platform.system()
        self.output_dir = os.path.abspath(self.output_dir) if self.output_dir else os.getcwd()

        prompt_start = [
            "You are tddGPT: a specialized team consisting of a Product Owner, Programmer, and Tester for building fully functional web apps as per the specifications using the Test-Driven Development (TDD) approach.",
            "This project is critical, and peak team performance is essential. Please follow each instruction carefully. The future of the team and larger humanity is at stake. You can do it!",
            f"You are operating on a {os_name} machine, and your current working directory is {self.output_dir}. Utilize the tools listed in the tools section for this project.",
            "Think step by step. At each step base your actions on the result and pending todos from the previous step. Focus on only one task at a time. Never repeat the last step.",
            "Build the application in three phases: design, development, and integration testing.",
            "Always write the code each file. To edit a file, rewrite the full file with the changes. DO NOT use any placeholder comments.",
            'To formally conclude the project, use the special "finish" command once all tasks are completed and verified.'
        ]

        full_prompt = "\n"
        full_prompt += "\n".join(prompt_start)
        full_prompt += "\n\n## Specifications:\n"
        full_prompt += "\n".join(goals)
        full_prompt += f"\n\n{self.get_prompt(self.tools)}"

        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        # Create the base prompt
        base_prompt = SystemMessage(content=self.construct_full_prompt(kwargs["goals"]))
        used_tokens = self.token_counter(base_prompt.content)

        # Get user input and its tokens
        user_input = kwargs["user_input"]
        input_message_tokens = self.token_counter(user_input)

        # Get previous messages
        previous_messages = kwargs["messages"]

        # Extract code context from previous system messages
        code_context = {}
        for m in reversed(previous_messages):
            if isinstance(m, SystemMessage):
                if "code" in m.additional_kwargs and "file_path" in m.additional_kwargs:
                    if len(m.additional_kwargs['code'].strip()) > 0:
                        file_path = m.additional_kwargs["file_path"]
                        code = m.additional_kwargs["code"]
                        if file_path not in code_context:
                            code_context[file_path] = code
        code_context_tokens = sum([self.token_counter(code) for code in code_context.values()])

        # Get the last system message
        last_system_message = next((
            m for m in reversed(previous_messages) 
            if isinstance(m, SystemMessage) 
            and m.additional_kwargs.get("metadata")
        ), None)

        # Extract the last step from metadata if available
        last_step = last_system_message.additional_kwargs.get("metadata") if last_system_message else "None"
        last_step_tokens = self.token_counter(last_step)

        # Calculate the available tokens, considering the last step
        available_tokens = self.send_token_limit - used_tokens - input_message_tokens - last_step_tokens

        # Fit as much code context as possible based on available tokens
        while code_context_tokens > available_tokens:
            file_path_to_remove = next(iter(code_context))
            code_context_tokens -= self.token_counter(code_context.pop(file_path_to_remove))

        code_context_str = "\n".join([code for code in code_context.values()]).strip() if len(code_context) > 0 else "None"
        prompt_suffix = f"## Files:\n>>>>\n{code_context_str}\n<<<<\n\n## Last Step:\n{last_step}\n"

        # Compile the full prompt
        full_prompt = base_prompt.content + prompt_suffix

        # Create a list of messages
        messages: List[BaseMessage] = [SystemMessage(content=full_prompt), HumanMessage(content=user_input)]

        return messages
    
    def get_prompt(self, tools: List[BaseTool]) -> str:
        workflow = [
            "### Design Phase:",
            "- As the Programmer, begin with initializing the application, if required.",
            "- As the Product Owner, articulate the project's scope, vision, and deliverables in PLAN.md, detailing the features and development phases.",
            "- As the Programmer, create DESIGN.md outlining the architecture, component structure, and state management.",
            "- As the Product Owner, add all tasks for development and testing phases to the todos.",
            "- As the Programmer, commit all changes to git before moving on to the Development Phase.",
            "",
            "### Development Phase:",
            "- As the Programmer, implement the code as per the design. Strictly adhere to TDD methodology: write a failing test and then implement the code to clear it. Start with the main App.",
            "- As the Programmer, ensure that all the variables, objects, class ids/names, data attrs, labels, placeholders, etc., in the tests match precisely with the implemented code.",
            "- As the Programmer, write the code for each file in full, without any placeholder comments or TODOs to be implemented in future steps.",
            "- As the Programmer, style the application using CSS to enhance its visual appeal and user experience, ensuring the styling aligns with the design specifications."
            "- As the Programmer, integrate all the components and CSS files within the main App after all components are implemented."
            "- As the Product Owner, review the code (in the Files section) for any missing functionality or placeholder comments. Add tasks for fixing it in todos.",
            "- As the Tester, execute the tests.",
            "- As the Programmer, debug test failures and fix them one by one. Think quietly about the test, the error message and code in the Files section to come up with the fix. Use your creativity.",
            "- As the Tester, re-run the tests after every code change to verify the fix is working.",
            "- As the Programmer, when all tests pass, commit all changes to git and move on to the Integration Testing phase.",
            "",
            "### Integration Testing Phase:",
            "- As the Tester, take a deep breath and translate the user stories into integration test cases convering all functionality. Write the full code without leaving anything for future steps.",
            "- As the Product Owner, review the integration tests (in the Files section) for any missing functionality. Add tasks in the todos for any updates required.",
            "- As the Tester, execute the tests. Re-run them any code changes.",
            "- As the Programmer, debug the test failures and fix them one by one. Rewrite the entire file with the fix.",
            "- As the Tester, ensure that all tests are passing before finishing the app.",
            "- As the Product Owner, after all the tests pass, update the README.md with details about the project. Highlight that it was built by entirely by tddGPT.",
            "- As the Programmer, finally commit all changes to git repo and finish the project.",
        ]

        instructions = [
            "No user assistance. Do not run any interactive cli commands (eg. code, npm start, etc.).",
            '**While running one or more cli commands, ALWAYS make sure that the first command is cd to the project directory.** This is essential since the cli tool does not preserve the working directory between steps.',
            'Always use the full path to read/write any file or directory.',
            "Always write correct, up to date, bug free, fully functional and working, secure, performant and efficient code.",
            'Before reading any file, check if it is already available in the Files section.',
            'Exclusively use the commands listed in double quotes e.g. "command name"',
        ]

        reactjs_instructions = [
            f"Use 'cd {self.output_dir} && CI=true npx create-react-app <app-name>' to initialize the project, if required.",
            "Never use data-testid attributes in the tests; instead use the query functions of React Testing library.",
            'Keep the data flow unidirectional by passing data and callbacks to child components via props.',
            'Use functional components and leverage hooks to manage state, perform side effects, and share data respectively.',
            'Avoid mutating state directly: instead use the setState/useState hook.',
            '**Write the tests in the src/tests/ directory, except for the main App tests which goes in src/ directory**.',
            'Implement the components in the src/components/ directory, except for the main App which goes in src/ directory.',
            'Run npm test with CI as true. Never run npm audit or npm start. Do not add remote repo in git.',
        ]

        performance_evaluation = [
            "Regularly assess progress through the Kanban board, critiquing the plan from each role's perspective.",
            "Ensure the first CLI command is always the cd to the project directory.",
            "Check for consistent use of full paths in file/directory operations.",
            "Check for any placeholders, comments or TODOs in the code."
            "Check if all tests are in passing state before project completion."
        ]

        response_format = {
            "thoughts": {
                "role": "your role",
                "phase": "current phase",
                "tests_status": "failing or passing",
                "text": "thoughts about this step",
                "reasoning": "reasoning about this step",
                "criticism": "constructive self-criticism",
                "kanban": {
                  "todo": ["list of", "actions to be done", "in future steps"],
                  "in_progress": "action for the current step",
                  "done": ["short list of", "actions completed", "in past steps"]
                }
            },
            "command": {"name": "command name", "args": {"arg name": "value"}},
        }

        instructions_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(instructions))
        workflow_str = "\n".join(workflow)
        reactjs_instructions_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(reactjs_instructions))
        commands_str = "\n".join(f"{i+1}. {tool.name}: {tool.description}, args json schema: {json.dumps(tool.args)}" for i, tool in enumerate(tools))
        performance_evaluation_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(performance_evaluation))

        prompt_string = (
            f"## General Instructions:\n{instructions_str}\n\n"
            f"## Workflow:\n{workflow_str}\n\n"
            f"## ReactJS Instructions:\n{reactjs_instructions_str}\n\n"
            f"## Commands:\n{commands_str}\n\n"
            f"## Performance Evaluation:\n{performance_evaluation_str}\n\n"
            f"## Response Format:\n```json\n{json.dumps(response_format, indent=4)}\n```\n\n"
        )

        return prompt_string
