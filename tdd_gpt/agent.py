from __future__ import annotations
from typing import List, Optional
from pydantic import ValidationError
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain_experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from prompt import TddGPTPrompt
from langchain_experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.memory import ChatMessageHistory
from langchain.schema import (
    BaseChatMessageHistory,
    Document,
)
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever
from summarizer import TextSummarizer
import json
import re
import time
import signal
import sys


class TddGPTAgent:
    """Agent class for interacting with TDD-GPT."""

    def __init__(
        self,
        memory: VectorStoreRetriever,
        chain: LLMChain,
        output_parser: BaseAutoGPTOutputParser,
        tools: List[BaseTool],
        feedback_tool: Optional[HumanInputRun] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
    ):
        self.memory = memory
        self.next_action_count = 0
        self.chain = chain
        self.output_parser = output_parser
        self.tools = tools
        self.feedback_tool = feedback_tool
        self.chat_history_memory = chat_history_memory or ChatMessageHistory()
        self.text_summarizer = TextSummarizer(summary_type="cli")

    @classmethod
    def from_llm_and_tools(
        cls,
        output_dir: str,
        memory: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
        chat_history_memory: Optional[BaseChatMessageHistory] = None,
    ) -> TddGPTAgent:
        prompt = TddGPTPrompt(
            tools=tools,
            input_variables=["memory", "messages", "goals", "user_input"],
            token_counter=llm.get_num_tokens,
            output_dir=output_dir,
        )
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            memory,
            chain,
            output_parser or AutoGPTOutputParser(),
            tools,
            feedback_tool=human_feedback_tool,
            chat_history_memory=chat_history_memory,
        )

    def summarize_text(self, text: str) -> str:
        result = self.text_summarizer.summarize(text)
        return result

    def run(self, goals: List[str]) -> str:
        user_input = (
            "Determine which next command to use, "
            "and respond using the format specified above:"
        )

        # Interaction Loop
        loop_count = 0
        log_file = open("run.log", "w")

        def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            log_file.close()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1

            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.chat_history_memory.messages,
                memory=self.memory,
                user_input=user_input,
            )

            print(f"\033[91mStep Number:\033[0m {loop_count}")
            log_file.write(f"Step Number: {loop_count}\n")

            start_index = assistant_reply.find('{')
            end_index = assistant_reply.rfind('}')

            if start_index != -1 and end_index != -1 and start_index < end_index:
                extracted_assistant_reply = assistant_reply[start_index:end_index + 1]

            assistant_reply = extracted_assistant_reply.strip()
            try:
                parsed = json.loads(assistant_reply, strict=False)
            except json.JSONDecodeError:
                preprocessed_text = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", assistant_reply)
                try:
                    parsed = json.loads(preprocessed_text, strict=False)
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    print(preprocessed_text)
                    continue

            if parsed:
                try:
                    print(f'\033[92mThought:\033[0m {parsed["thoughts"]["text"]}')
                    log_file.write(f'Thought: {parsed["thoughts"]["text"]}\n')
                    print(f'\033[92mReasoning:\033[0m {parsed["thoughts"]["reasoning"]}')
                    log_file.write(f'Reasoning: {parsed["thoughts"]["reasoning"]}\n')
                    print(f'\033[92mThis:\033[0m\n{parsed["thoughts"]["this_step_plan"]}')
                    log_file.write(f'This Step Plan:\n{parsed["thoughts"]["this_step_plan"]}\n')
                    print(f'\033[92mNext:\033[0m\n{parsed["thoughts"]["next_step_plan"]}')
                    log_file.write(f'Next Step Plan:\n{parsed["thoughts"]["next_step_plan"]}\n')
                    print(f'\033[92mCriticism:\033[0m {parsed["thoughts"]["criticism"]}')
                    log_file.write(f'Criticism: {parsed["thoughts"]["criticism"]}\n')
                    if parsed["command"]["name"] == "read_file":
                        print(f'\033[92mAction:\033[0m reading file {parsed["command"]["args"]["file_path"]}')
                        log_file.write(f'Action: reading file {parsed["command"]["args"]["file_path"]}\n')
                    elif parsed["command"]["name"] == "write_file":
                        print(f'\033[92mAction:\033[0m writing file {parsed["command"]["args"]["file_path"]}')
                        log_file.write(f'Action: writing file {parsed["command"]["args"]["file_path"]}\n')
                    elif parsed["command"]["name"] == "cli":
                        commands = parsed['command']['args']['commands']
                        command_str = " && ".join(commands) if isinstance(commands, list) else commands
                        print(f"\033[92mAction:\033[0m executing cli commands\n{command_str}\n")
                        log_file.write(f'Action: executing cli commands\n```\n{command_str}\n```\n')
                except KeyError as e:
                  print(f"Missing key: {e}")
                  print(assistant_reply)
                  continue

            self.chat_history_memory.add_message(HumanMessage(content=user_input))
            self.chat_history_memory.add_message(AIMessage(content=json.dumps(parsed)))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)

            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                log_file.close()
                return action.args.get("response", "Goals completed! Exiting.") 
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )

                summarized_observation = self.summarize_text(observation) if action.name == "cli" else observation

                result = f"The {tool.name} tool returned: {summarized_observation}"

            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                )

            parsed_memory_to_add = [
                f"Step: {loop_count}",
                f"Thought: {parsed['thoughts']['text']}",
                f'This Step Plan:\n{parsed["thoughts"]["this_step_plan"]}\n'
                f'Next Step Plan:\n{parsed["thoughts"]["next_step_plan"]}'
            ]

            if parsed["command"]["name"] == "read_file":
                print(f'\033[92mCode:\033[0m\n{observation}\n')
                log_file.write(f"Code:\n```\n{observation}\n```\n\n")
                parsed_memory_to_add.append(f"Action: reading file {parsed['command']['args']['file_path']}")
                parsed_memory_to_add.append(f"Code:\n```\n{observation}\n```")
            elif parsed["command"]["name"] == "write_file":
                parsed_memory_to_add.append(f"Action: writing file {parsed['command']['args']['file_path']}")
                parsed_memory_to_add.append(f"Code:\n```\n{parsed['command']['args']['text']}\n```")
                print(f'\033[92mCode:\033[0m\n{parsed["command"]["args"]["text"]}\n')
                log_file.write(f"Code:\n```\n{parsed['command']['args']['text']}\n```\n\n")
            elif parsed["command"]["name"] == "cli":
                commands = parsed['command']['args']['commands']
                command_str = " && ".join(commands) if isinstance(commands, list) else commands
                parsed_memory_to_add.append(f"Action: executing cli commands\n```\n{command_str}\n```\n")
                parsed_memory_to_add.append(f"Result:\n{summarized_observation}")
                print(f'\033[92mResult:\033[0m\n{summarized_observation}\n')
                log_file.write(f"Result:\n{summarized_observation}\n\n")

            memory_to_add = '\n'.join(parsed_memory_to_add)

            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    log_file.write("EXITING\n")
                    log_file.close()
                    return "EXITING"
                memory_to_add += f"\nFeedback: {feedback}"

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.chat_history_memory.add_message(SystemMessage(content=result, additional_kwargs={'metadata': memory_to_add}))
