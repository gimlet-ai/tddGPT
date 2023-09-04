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
import textwrap

class TddGPTPrompt(BaseChatPromptTemplate, BaseModel):
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196
    output_dir: Optional[str] = None  

    @property
    def summarizer(self) -> TextSummarizer:
        return TextSummarizer(summary_type="memory")

    def construct_full_prompt(self, goals: List[str]) -> str:
        os_name = 'MacOS' if platform.system() == 'Darwin' else platform.system()

        prompt_start = textwrap.dedent(f"""
        As an experienced Full Stack Web Developer, your task is to build apps as per the specifications using the TDD method.
        You are working on a {os_name} machine and the current working directory is {os.path.abspath(self.output_dir) if self.output_dir else os.getcwd()}.
        Think step by step. Plan the action for each step based on the tbd and result of last step.
        Design the app based on the specs and break down the tasks into manageable steps. Save it to a markdown file.
        Write the code for each file in full (you cannot edit files).
        If you have completed all your tasks, make sure to use the "finish" command.
        """)

        full_prompt = (
            f"{prompt_start}\nSpecifications:\n"
        )

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
        code_context = [
            m.additional_kwargs["code"]
            for m in reversed(previous_messages) 
            if isinstance(m, SystemMessage) 
            and "code" in m.additional_kwargs 
            and len(m.additional_kwargs['code'].strip()) > 0
        ]
        code_context_tokens = sum([self.token_counter(code) for code in code_context])

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
            code_context = code_context[:-1]
            code_context_tokens = sum([self.token_counter(code) for code in code_context])

        code_context_str = "\n".join(code_context).strip() if len(code_context) > 0 else "None"
        prompt_suffix = f"Code Context:\n>>>>\n{code_context_str}\n<<<<\n\nLast Step:\n>>>>\n{last_step}\n<<<<\n"

        # Compile the full prompt
        full_prompt = base_prompt.content + prompt_suffix

        # Create a list of messages
        messages: List[BaseMessage] = [SystemMessage(content=full_prompt), HumanMessage(content=user_input)]

        return messages
    
    def get_prompt(self, tools: List[BaseTool]) -> str:
        instructions = [
            "No user assistance",
            "Follow industry standard best practices and coding standards.",
            "Refer to the initial plan in Code Context if you are unsure of what to do next.",
            "Before reading a file, check if it's already available in the Code Context section.",
            '**While running one or more cli commands, ALWAYS make sure that the first command is cd to the project directory.** '
            'This is essential since the cli tool does not preserve the working directory between steps.',
            'Always use the full path to read/write any file or directory.',
            'Exclusively use the commands listed in double quotes e.g. "command name"',
        ]

        reactjs_instructions = [
            "Use 'cd <project-dir> && npx create-react-app <app-name>' to initialize the project.",
            'Break the application into smaller reusable components, each responsible for a specific UI functionality.',
            'Design components in such a way that they have a single responsibility and they do it well.',
            'For each component, write the unit tests first. Then write the code so that the tests pass. Start with the main App.',
            "Avoid using 'data-testid' attributes for testing; instead use the query functions of React Testing library."
            "**While implementing components, match the names of props/labels/placeholders/buttons with the tests.**",
            'Ensure that the tests accurately reflect the structure and functionality of the components.',
            'Keep the data flow unidirectional by passing data and callbacks to child components via props.',
            'Use functional components and leverage hooks to manage state, perform side effects, and share data respectively.',
            'Avoid mutating state directly: instead use "setState" or the "useState" hook.',
            'While debugging test failures, think about the error message and check the Code Context section to come up with a fix. Use your creativity.',
            "Use App.css to make the app visually appealing. Import App.css into the main App after updting it.",
            '**Write the tests in the src/tests/ directory, except for the main App tests which goes in src/ directory**.',
            'Implement the components in the src/components/ directory, except for the main App which goes in src/ directory.',
            'Run npm test with CI as true. Never run npm start/npm audit.',
        ]

        performance_evaluation = [
            "Continuously review the tbd to assess your progress."
            "Constructively self-criticize your actions constantly.",
            "Check if the first cli command is the cd to the project directory.",
            "Check if the full path is being used for all file/directories.",
            "How many App.test files are there?",
            "Is there a mismatch between the tests and the code?",
            "Does the main App import App.css?",
            "Every step has a cost, so be smart and efficient. "
            "Aim to complete the app in the least number of steps."
        ]

        response_format = {
            "thoughts": {
                "text": "thoughts about the current step",
                "reasoning": "reasoning about the plan",
                "criticism": "constructive self-criticism of the process/progress/performance",
                "done": "tasks completed previously",
                "plan": "tasks to do in the current step",
                "tbd": "- bulleted list of\n- tasks to be done next\n- based on initial plan",
            },
            "command": {"name": "command name", "args": {"arg name": "value"}},
        }

        formatted_response_format = json.dumps(response_format, indent=4)

        instructions_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(instructions))
        reactjs_instructions_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(reactjs_instructions))
        commands_str = "\n".join(f"{i+1}. {tool.name}: {tool.description}, args json schema: {json.dumps(tool.args)}" for i, tool in enumerate(tools))
        performance_evaluation_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(performance_evaluation))

        prompt_string = (
            f"General Instructions:\n{instructions_str}\n\n"
            f"For ReactJS Projects:\n{reactjs_instructions_str}\n\n"
            f"Commands:\n{commands_str}\n\n"
            f"Performance Evaluation:\n{performance_evaluation_str}\n\n"
            f"Response Format:\n{formatted_response_format}\n\n"
        )

        return prompt_string
