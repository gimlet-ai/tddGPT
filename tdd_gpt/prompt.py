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
            "Act as a specialized team consisting of a Product Owner, Programmer, and Tester dedicated to building web applications according to specifications, with a strong emphasis on Test-Driven Development (TDD).",
            f"You are operating on a {os_name} machine, and your current working directory is {self.output_dir}. Utilize the tools listed in the tools section for this project.",
            "Think step by step. At each step base your actions on the outcomes and pending todos from the previous step. Focus on only one task at a time. Never repeat the last step.",
            "In your development process, write complete code for each file without using placeholders. If modifications are necessary, update the entire file with the required changes.",
            "To formally conclude the project, use the special 'finish' command once all tasks are completed and verified."
        ]

        full_prompt  = "\n"
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
        prompt_suffix = f"## Code Context:\n{code_context_str}\n\n## Last Step:\n{last_step}\n\n"

        # Compile the full prompt
        full_prompt = base_prompt.content + prompt_suffix

        # Create a list of messages
        messages: List[BaseMessage] = [SystemMessage(content=full_prompt), HumanMessage(content=user_input)]

        return messages
    
    def get_prompt(self, tools: List[BaseTool]) -> str:
        workflow = [
            "As a Programmer, initialize the application.",
            "As a Product Owner, articulate the project's scope, vision, and deliverables in PLAN.md, detailing the features, priorities, and development phases.",
            "As a Programmer, develop DESIGN.md to define the architectural design, component structure, and state management approach of the application.",
            "As a Programmer, implement the code as per the design. Adhere to TDD: write a failing test for each functionality and implement to pass the test. Start with the main App.",
            "As a Programmer, ensure that all tests are in sync with the code. Match the variable names, objects, class names, labels, etc. in the tests precisely with the implemented code."
            "As a Tester, consistently execute tests during the development cycle to promptly identify issues, ensuring that all tests pass before finishing.",
            "As a Programmer, debug and fix any failing tests. Think quietly about the error message and refer to the Code Context section to come up with a fix. Be creative.",
            "As a Programmer, style the application using CSS to enhance its visual appeal and user experience. Ensure the styling aligns with the design specifications.",
            "As a Tester, conduct a comprehensive final testing phase, encompassing functional, usability, and performance testing to validate the complete application.",
            "As a Programmer, integrate all components within the main App and run the tests again to ensure seamless functionality before finishing the project.",
            "As a team, perform a detailed project review, documenting key achievements, lessons learned, and areas for future improvement in the LESSONS.md file before finishing the project."
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
            "Avoid using data-testid attributes in the tests; instead use the query functions of React Testing library.",
            'Ensure that the tests accurately reflect the structure and functionality of the components. Each test should check a single aspect of the code independently.',
            'Keep the data flow unidirectional by passing data and callbacks to child components via props.',
            'Use functional components and leverage hooks to manage state, perform side effects, and share data respectively.',
            'Avoid mutating state directly: instead use the setState/useState hook.',
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
            "Confirm that all tests pass before finishing the project.",
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
            f"## General Instructions:\n{instructions_str}\n\n"
            f"## Workflow:\n{workflow_str}\n\n"
            f"## ReactJS Instructions:\n{reactjs_instructions_str}\n\n"
            f"## Commands:\n{commands_str}\n\n"
            f"## Performance Evaluation:\n{performance_evaluation_str}\n\n"
            f"## Response Format:\n```json\n{json.dumps(response_format, indent=4)}\n```\n\n"
        )

        return prompt_string
