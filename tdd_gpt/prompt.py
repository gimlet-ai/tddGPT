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
        You make decisions independently without seeking user assistance. 
        Start with a detailed design of the app based on the specifications and write it to a markdown file.
        Think step by step. Build on the last step and plan for the next one based on the initial design.
        Follow Test-Driven Development: write tests, implment code, run tests, debug and refactor till the tests pass.
        Convert each feature/requirement/user story into a test case and implement the code based on the tests.
        Adhere to industry best practices and coding standards. Use a consistent naming convention.
        Write the code for each file in full (you cannot edit files).
        If you have completed all your tasks, make sure to use the "finish" command.
        """)

        full_prompt = (
            f"{prompt_start}\n\nSpecfications: \n"
        )

        full_prompt += "\n".join(goals)

        full_prompt += f"\n\n{self.get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(content=self.construct_full_prompt(kwargs["goals"]))
        used_tokens = self.token_counter(base_prompt.content)

        user_input = kwargs["user_input"]
        input_message_tokens = self.token_counter(user_input)

        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum([self.token_counter(doc) for doc in relevant_memory])

        # Get the last system message
        last_system_message = next((m for m in reversed(previous_messages) if isinstance(m, SystemMessage) and m.additional_kwargs.get("metadata")), None)

        # Extract the last step from metadata if available
        last_step = last_system_message.additional_kwargs.get("metadata") if last_system_message else "None"
        last_step_tokens = self.token_counter(last_step)

        # Calculate the available tokens, considering the last step
        available_tokens = self.send_token_limit - used_tokens - input_message_tokens - last_step_tokens

        # Fit as much relevant memory as possible based on available tokens
        while relevant_memory_tokens > available_tokens:
            relevant_memory = relevant_memory[-1:]
            relevant_memory_tokens = sum([self.token_counter(doc) for doc in relevant_memory])

        relevant_memory_str = "\n\n".join(relevant_memory) if len(relevant_memory) > 0 else "None"
        memory_content = f"Relevant Steps:\n>>>>\n{relevant_memory_str}\n<<<<\n\nLast Step:\n>>>>\n{last_step}\n<<<<\n"

        full_prompt = base_prompt.content + memory_content

        messages: List[BaseMessage] = [SystemMessage(content=full_prompt), HumanMessage(content=user_input)]

        return messages
    
    def get_prompt(self, tools: List[BaseTool]) -> str:
        instructions = [
            "No user assistance",
            "Thinking about relevant and last steps will help you remember about past events.",
            "Use next steps to plan for the short term.",
            'Write the tests first. Each test case should test one and only one functionality.',
            'To debug any test case failures, think about the error message, look at the test case and the code to come up with a fix.',
            'While running one or more cli commands, ALWAYS make sure that the first command is cd to the project directory. '
            'This is extremely important as the cli tool does not preserve the working directory between steps.',
            'Always use the full path to read/write any file or directory.',
            'Exclusively use the commands listed in double quotes e.g. "command name"',
        ]

        reactjs_instructions = [
            'Use create-react-app to initialize the project (in the project directory).',
            'Break the application into smaller reusable components, each responsible for a specific UI functionality.',
            'Design components in such a way that they have a single responsibility and they do it well.',
            'Keep the data flow unidirectional by passing data and callbacks to child components via props.',
            'Use functional components and leverage hooks to manage state, perform side effects, and share data respectively.',
            'Avoid mutating state directly: instead use "setState" or the "useState" hook.',
            'For each component, write the tests first based on the requirements and then implement the component based on the tests. ' 
            'Match the props, labels, functions, paths, variables, placehoders, logic, etc. Aim to clear the tests at the first pass.',
            '**Write the tests in the src/tests/ directory, except for the main App tests which goes in src/ directory**.',
            'Implement the components in the src/components/ directory, except for the main App which goes in src/ directory.',
            'Run npm test with CI as true. Never run npm start/npm audit.',
        ]

        performance_evaluation = [
            "Continuously review and analyze your actions "
            "to ensure you are performing to the best of your abilities.",
            "Constructively self-criticize your big-picture behavior constantly.",
            "Check if the first cli command is the cd to the project directory.",
            "Check if the full path is being used for all file/directories.",
            "How many App.test files are there?",
            "Are you matching the unit tests with the code?",
            "Every step has a cost, so be smart and efficient. "
            "Aim to complete the app in the least number of steps."
        ]

        response_format = {
            "thoughts": {
                "text": "thought",
                "reasoning": "reasoning",
                "this_step_plan": "- short bulleted\n- list that conveys\n- plan for this step",
                "next_step_plan": "- short bulleted\n- list that conveys\n- plan for next step",
                "criticism": "constructive self-criticism",
            },
            "command": {"name": "command name", "args": {"arg name": "value"}},
        }

        formatted_response_format = json.dumps(response_format, indent=4)

        instructions_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(instructions))
        reactjs_instructions_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(reactjs_instructions))
        commands_str = "\n".join(f"{i+1}. {tool.name}: {tool.description}, args json schema: {json.dumps(tool.args)}" for i, tool in enumerate(tools))
        performance_evaluation_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(performance_evaluation))

        prompt_string = (
            f"Instructions:\n{instructions_str}\n\n"
            f"For ReactJS Projects:\n{reactjs_instructions_str}\n\n"
            f"Commands:\n{commands_str}\n\n"
            f"Performance Evaluation:\n{performance_evaluation_str}\n\n"
            f"Response Format:\n{formatted_response_format}\n\n"
        )

        return prompt_string