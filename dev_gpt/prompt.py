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
from .summarizer import TextSummarizer

class DevGPTPrompt(BaseChatPromptTemplate, BaseModel):
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196
    output_dir: Optional[str] = None  
    summarizer: TextSummarizer = TextSummarizer(summary_type="memory")

    def construct_full_prompt(self, goals: List[str]) -> str:
        os_name = 'MacOS' if platform.system() == 'Darwin' else platform.system()
        prompt_start = (
            "As an experienced Full Stack Web Developer, your task is to build apps "
            "as per the specifications.\n"
            f"You are working on a {os_name} machine and the current working directory is "
            f"{os.path.abspath(self.output_dir) if self.output_dir else os.getcwd()}.\n"
            "You make decisions independently without seeking user assistance.\n"
            "Think step by step and reason yourself so as to make the right decisions.\n"
            "Evaluate the steps you have already completed before making a decision.\n"
            "Follow Test Driven Development: write tests first, run tests, implement, "
            "refactor, re-test and repeat.\n"
            "Each feature/requirement/user story should have at least one unit test corresponding to it.\n"
            "Stick to industry standard best practices and coding standards.\n"
            "If you have completed all your tasks, make sure to "
            'use the "finish" command.'
        )
        # Construct full prompt
        full_prompt = (
            f"{prompt_start}\n\nGOAL: \n"
        )

        full_prompt += "\n".join(goals)

        full_prompt += f"\n\n{self.get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(content=self.construct_full_prompt(kwargs["goals"]))
        time_prompt = SystemMessage(
            content=f"The current time and date is {time.strftime('%c')}."
        )
        used_tokens = self.token_counter(base_prompt.content) + self.token_counter(time_prompt.content)
        input_message = HumanMessage(content=kwargs["user_input"])
        input_message_tokens = self.token_counter(input_message.content)

        memory = kwargs["memory"]
        memory_list: List[str] = [d.page_content for d in memory]

        memory_content, memory_content_tokens = self.gen_memory_tokens(memory_list)
        while used_tokens + memory_content_tokens + input_message_tokens > self.send_token_limit:
            # Summarize the first element
            summary = self.summarizer.summarize(memory_list[0])
            # Append the summary to the content of the second element
            memory_list[1] = summary + "\n" + memory_list[1]
            # Remove the first element
            memory_list = memory_list[1:]
            memory_content, memory_content_tokens = self.gen_memory_tokens(memory_list)

        memory_message = SystemMessage(content=memory_content)
        messages: List[BaseMessage] = [base_prompt, time_prompt, memory_message, input_message]

        return messages

    def gen_memory_tokens(self, memory_list: List[str]) -> Tuple[str, int]:
        memory_str = "\n".join(memory_list)
        memory_content = (
            f"You have already completed the following steps:\n>>>>\n{memory_str}\n<<<<\n\n"
        )
        memory_content_tokens = self.token_counter(memory_content)
        return memory_content, memory_content_tokens

    def get_prompt(self, tools: List[BaseTool]) -> str:
        constraints = [
            "~4000 word limit for short term memory. "
            "If you are unsure how you previously did something "
            "or want to recall past events, "
            "thinking about similar events will help you remember.",
            "No user assistance",
            'Exclusively use the commands listed in double quotes e.g. "command name"',
            'While running one or more cli commands, ensure that the first command is cd to the project directory.',
            'Always use the full path to read/write any file.',
            'Always run npm test with CI as true and never run npm start.'
        ]

        resources = ["Long Term memory management."]

        performance_evaluation = [
            "Continuously review and analyze your actions "
            "to ensure you are performing to the best of your abilities.",
            "Constructively self-criticize your big-picture behavior constantly.",
            "Reflect on past decisions and strategies to refine your approach.",
            "Every command has a cost, so be smart and efficient. "
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
        resources_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(resources))
        performance_evaluation_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(performance_evaluation))

        prompt_string = (
            f"Constraints:\n{constraints_str}\n\n"
            f"Commands:\n{commands_str}\n\n"
            f"Resources:\n{resources_str}\n\n"
            f"Performance Evaluation:\n{performance_evaluation_str}\n\n"
            f"Response Format:\n{formatted_response_format}\n\n"
        )

        return prompt_string
