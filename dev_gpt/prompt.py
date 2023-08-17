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

class DevGPTPrompt(BaseChatPromptTemplate, BaseModel):
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
        As an experienced Full Stack Web Developer, your task is to build apps as per the specifications provided in the goals.
        You are working on a {os_name} machine and the current working directory is {os.path.abspath(self.output_dir) if self.output_dir else os.getcwd()}.
        You make decisions independently without seeking user assistance. 
        You are talented: use your creativity to overcome technical challenges.
        Think step by step and build the app iteratively. Take into account the steps already completed.
        Follow Test Driven Development: write tests first, run tests, implement, refactor, re-test and repeat. 
        If the test fails, start by addressing the first error. Fix erorrs one by one.
        Each feature/requirement/user story should have at least one unit test corresponding to it.
        Stick to industry standard best practices and coding standards.
        Write the code for each file in full.
        If you have completed all your tasks, make sure to use the "finish" command.
        """)

        full_prompt = (
            f"{prompt_start}\n\nGOAL: \n"
        )

        full_prompt += "\n".join(goals)

        full_prompt += f"\n\n{self.get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        full_prompt = self.construct_full_prompt(kwargs["goals"])
        used_tokens = self.token_counter(full_prompt)

        user_input = kwargs["user_input"]
        input_message_tokens = self.token_counter(user_input)

        memory = kwargs["memory"]
        last_elements_list = memory[-10:]

        # Combine the last 10 elements
        if len(last_elements_list) > 0:
            last_elements = "\n".join([element.page_content for element in last_elements_list])

            available_tokens = self.send_token_limit - used_tokens - input_message_tokens

            # Call summarization with the available tokens
            summary = self.summarizer.summarize(last_elements, token_max=available_tokens)

            memory_content = f"Completed Steps:\n{summary}\n"
            print(f"\n\033[36mSteps Summary:\033[0m\n{summary}\n")

            full_prompt += memory_content

        messages: List[BaseMessage] = [SystemMessage(content=full_prompt), HumanMessage(content=user_input)]

        return messages

    def get_prompt(self, tools: List[BaseTool]) -> str:
        constraints = [
            "~4000 word limit for short term memory. "
            "If you are unsure how you previously did something "
            "or want to recall past events, "
            "thinking about completed steps will help you remember.",
            "No user assistance",
            'Exclusively use the commands listed in double quotes e.g. "command name"',
            'While running one or more cli commands, ALWAYS make sure that the first command is cd to the project directory. '
            'This is extremely important as the cli tool does not preserve the working directory between steps.',
            'Always use the full path to read/write any file or directory.',
            'For ReactJS projects, always use create-react-app to initialize the project (in the project directory), '
            'use function components (in src/components directory), run npm test with CI as true and never run npm start/npm audit.'
        ]

        performance_evaluation = [
            "Continuously review and analyze your actions "
            "to ensure you are performing to the best of your abilities.",
            "Constructively self-criticize your big-picture behavior constantly.",
            "Assess the completed steps to evaluate if TDD process is being correctly followed.",
            "Check if the first cli command is the cd to the project directory.",
            "Check if the full path is being used for all file/directories.",
            "Evaluate critically how many steps it took to clear a test for a particular feature.",
            "Every step has a cost, so be smart and efficient. "
            "Aim to complete tasks in the least number of steps."
        ]

        response_format = {
            "thoughts": {
                "text": "thought",
                "reasoning": "reasoning",
                "plan": "- short bulleted\n- list that conveys\n- long-term plan",
                "criticism": "constructive self-criticism",
                "speak": "thoughts summary to say to user",
            },
            "command": {"name": "command name", "args": {"arg name": "value"}},
        }

        formatted_response_format = json.dumps(response_format, indent=4)

        constraints_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(constraints))
        commands_str = "\n".join(f"{i+1}. {tool.name}: {tool.description}, args json schema: {json.dumps(tool.args)}" for i, tool in enumerate(tools))
        performance_evaluation_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(performance_evaluation))

        prompt_string = (
            f"Constraints:\n{constraints_str}\n\n"
            f"Commands:\n{commands_str}\n\n"
            f"Performance Evaluation:\n{performance_evaluation_str}\n\n"
            f"Response Format:\n{formatted_response_format}\n\n"
        )

        return prompt_string
