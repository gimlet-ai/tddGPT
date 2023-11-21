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
    send_token_limit: int = 4096
    output_dir: Optional[str] = None  

    @property
    def summarizer(self) -> TextSummarizer:
        return TextSummarizer(summary_type="memory")

    def construct_full_prompt(self, goals: List[str]) -> str:
        os_name = 'MacOS' if platform.system() == 'Darwin' else platform.system()
        self.output_dir = os.path.abspath(self.output_dir) if self.output_dir else os.getcwd()

        prompt_start = textwrap.dedent(f"""
        Act as a team consisting of Product Owner, Software Developer and QA Engineer which builds web applications as per the specifications.
        You are working on a {os_name} machine and the current working directory is {self.output_dir}. You have access to the tools listed in the tools section.
        Think step by step. Plan the action of each step based on the result and Kanban todo's of the last step. Only take one action at a time. 
        As the Product Owner, create a detailed project plan in PLAN.md file, regularly revisiting and adjusting it based on project progress and challenges. 
        As the Software Developer, create a detailed design based on specs in DESIGN.md file. Write the code, run tests and debug any issues. 
        As the QA Engineer, develop comprehensive tests for all features and edge cases. Regularly execute the entire test suite to catch issues early. 
        Write the code for each file in full, without any placeholders. To edit a file, rewrite the entire file with the changes.
        After application is complete, reflect on the mistakes made and identify some areas of improvement. Save it to LESSONS.md file.
        **When all tasks are complete, use the "finish" command to exit.** Use only "finish" and no other command. 
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
        prompt_suffix = f"Code Context:\n>>>>\n{code_context_str}\n<<<<\n\nLast Step:\n>>>>\n{last_step}\n<<<<\n"

        # Compile the full prompt
        full_prompt = base_prompt.content + prompt_suffix

        # Create a list of messages
        messages: List[BaseMessage] = [SystemMessage(content=full_prompt), HumanMessage(content=user_input)]

        return messages
    
    def get_prompt(self, tools: List[BaseTool]) -> str:
        workflow = [
            "Begin with initializing the application, followed by a systematic and iterative approach to development and testing.",
            "Document each step meticulously, with no placeholders in the code. When editing, update the entire file with the new changes.",
        ]

        instructions = [
            "No user assistance. Do not run any interactive cli commands (eg. code, npm start, etc.).",
            '**While running one or more cli commands, ALWAYS make sure that the first command is cd to the project directory.** '
            'This is essential since the cli tool does not preserve the working directory between steps.',
            'Always use the full path to read/write any file or directory.',
            'Exclusively use the commands listed in double quotes e.g. "command name"',
        ]

        reactjs_instructions = [
            f"Use 'cd {self.output_dir} && CI=true npx create-react-app <app-name>' to initialize the project, if required.",
            "Focus on breaking down the application into smaller, reusable components for better modularity and maintainability.",
            'For each component, write the unit tests first. Then implement the code based on the tests. Always start with the main App.',
            "Before implementing the code, take a deep breath and think quietly about how to clear the tests at first go. Aim to get it right the first time.",
            "Avoid using data-testid attributes in the tests; instead use the query functions of React Testing library.",
            'Ensure that the tests accurately reflect the structure and functionality of the components. Each test should check a single aspect of the code independently.',
            'Keep the data flow unidirectional by passing data and callbacks to child components via props.',
            'Use functional components and leverage hooks to manage state, perform side effects, and share data respectively.',
            'Avoid mutating state directly: instead use the setState/useState hook.',
            'While debugging test failures, think about the error message and refer to the Code Context section to come up with a fix. Be creative.',
            "Style the app to make it visually appealing, responsive and user friendly. Base it on the CSS provided, if any. Use your imagination.",
            '**Write the tests in the src/tests/ directory, except for the main App tests which goes in src/ directory**.',
            'Implement the components in the src/components/ directory, except for the main App which goes in src/ directory.',
            'Run npm test with CI as true. Never run npm audit/npm start.',
        ]

        performance_evaluation = [
            "Regularly assess progress through a Kanban Board, critiquing the plan from each role's perspective.",
            "Ensure the first CLI command is always the cd to the project directory.",
            "Check for consistent use of full paths in file/directory operations.",
            "Verify the inclusion of CSS files in the main App.",
            "Assess the integration of components within the main App.",
            'Track the frequency and outcomes of test executions.',
            "Confirm that all tests pass before concluding the project.",
            "Aim for efficiency, minimizing the number of steps without sacrificing quality.",
        ]

        response_format = {
            "thoughts": {
                "role": "your role",
                "text": "thoughts about this step",
                "reasoning": "reasoning about this step",
                "criticism": "constructive self-criticism",
                "kanban": {
                  "todo": ["list of", "actions to be done", "in future steps"],
                  "in_progress": "action for this step",
                  "done": ["short bulleted list", "of actions completed", "in past steps"]
                }
            },
            "command": {"name": "command name", "args": {"arg name": "value"}},
        }

        instructions_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(instructions))
        workflow_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(workflow))
        reactjs_instructions_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(reactjs_instructions))
        commands_str = "\n".join(f"{i+1}. {tool.name}: {tool.description}, args json schema: {json.dumps(tool.args)}" for i, tool in enumerate(tools))
        performance_evaluation_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(performance_evaluation))

        prompt_string = (
            f"General Instructions:\n{instructions_str}\n\n"
            f"Workflow:\n{workflow_str}\n\n"
            f"For ReactJS Projects:\n{reactjs_instructions_str}\n\n"
            f"Commands:\n{commands_str}\n\n"
            f"Performance Evaluation:\n{performance_evaluation_str}\n\n"
            f"Response Format:\n```json\n{json.dumps(response_format, indent=4)}\n```\n\n"
        )

        return prompt_string
