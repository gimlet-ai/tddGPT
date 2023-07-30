import os
import platform
import time
import json
from typing import Any, Callable, List, Optional

from pydantic import BaseModel

from langchain.prompts.chat import BaseChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever

class DevGPTPrompt(BaseChatPromptTemplate, BaseModel):
    tools: List[BaseTool]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196
    output_dir: Optional[str] = None  

    def construct_full_prompt(self, goals: List[str]) -> str:
        os_name = 'MacOS' if platform.system() == 'Darwin' else platform.system()
        prompt_start = (
            "As an experienced Full Stack Web Developer, your task is to build apps \n"
            "given the requirements (or user stories) using the TDD methodology.\n"
            f"You are working on a {os_name} machine and the current working directory is "
            f"{os.path.abspath(self.output_dir) if self.output_dir else os.getcwd()}.\n"
            "You make decisions independently without seeking user assistance.\n"
            "think step by step and reason yourself so as to make the right decisions.\n"
            "Follow the Test Driven Development methodology. Start by writing the test, \n"
            "then run the test, write the code, then run the test again, refactor the code \n"
            "and repeat till the tests pass. Each feaure should correspond to more or more test cases.\n"
            "Follow an incremental development process: start with the simplest possible \n"
            "version of a feature, make sure it works, then build on it adding complexity \n"
            "and additional features in small manageable steps.\n"
            "Stick to industry standard best practices: emphasize simplicity and efficiency.\n"
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
        used_tokens = self.token_counter(base_prompt.content) + self.token_counter(
            time_prompt.content
        )
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [self.token_counter(doc) for doc in relevant_memory]
        )

        while used_tokens + relevant_memory_tokens > 2500:
            relevant_memory = relevant_memory[-1:]
            relevant_memory_tokens = sum(
                [self.token_counter(doc) for doc in relevant_memory]
            )

        relevant_memory_str = "\n".join(relevant_memory)
        content_format = (
            f"This reminds you of these events "
            f"from your past:\n>>>>\n{relevant_memory_str}\n<<<<\n\n"
        )

        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens
        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[BaseMessage] = [base_prompt, time_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages

    def get_prompt(self, tools: List[BaseTool]) -> str:
        constraints = [
            "~4000 word limit for short term memory. "
            "Your short term memory is short, "
            "so immediately save important information to files.",
            "If you are unsure how you previously did something "
            "or want to recall past events, "
            "thinking about similar events will help you remember.",
            "No user assistance",
            'Exclusively use the commands listed in double quotes e.g. "command name"',
            'While running one or more cli commands, ensure that the first command is cd to the project directory. '
            'This is very important as the cli tool is not persistent and directories are not preserved across steps.',
            'Always use the full path to read/write any file.',
            'Always run npm test with CI as true. Ignore any npm warnings/audit/vulnerability issues.'
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
