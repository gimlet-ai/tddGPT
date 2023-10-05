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

    def parse_npm_test_output(self, test_output):
        lines = test_output.strip().split("\n")
        parsed_output = []
        inside_console_error_block = False
        inside_console_details = False
        console_line_count = 0
        in_expect_block = False
        after_received = False

        for line in lines:
            if line.startswith("PASS") or line.startswith("FAIL"):
                parsed_output.append(line + ("\n" if line.startswith("PASS") else ""))
                inside_console_error_block = False
                inside_console_details = False
                in_expect_block = False  # Reset the expect block flag
                after_received = False  # Reset the after_received flag

            elif line.startswith("  ●"):
                parsed_output.append(line)
                inside_console_error_block = "Console" in line
                if inside_console_error_block:
                    inside_console_details = True
            
            elif "Error: " in line:
                parsed_output.append(f"    {line.strip()}")
            
            elif line.startswith("    expect("):
                parsed_output.append(line.strip())
                in_expect_block = True  # Set flag to capture the next few lines
            
            elif in_expect_block and ("Expected element" in line or "Received:" in line or "Test todo" in line or "Number of calls:" in line):
                parsed_output.append(f"    {line.strip()}")
                if "Received:" in line:
                    after_received = True  # Flag to capture the line after "Received:"
            
            elif after_received:  # If the flag is True, capture the next line
                parsed_output.append(f"    {line.strip()}")
                after_received = False  # Reset the flag
            
            elif "Expected:" in line or "Received:" in line:
                if not inside_console_error_block:
                    parsed_output.append(f"    {line.strip()}")
            
            elif line.startswith("    >"):
                parsed_output.append(f"    {line}\n")
            
            elif inside_console_error_block and inside_console_details:
                if "console.error" in line:
                    parsed_output.append("  ● console.error")
                elif "Warning:" in line or "Error:" in line: 
                    parsed_output.append(f"    {line.strip()}")
                elif console_line_count < 1:
                    parsed_output.append(f"    {line.strip()}")
                    console_line_count += 1
            
            elif "Test Suites:" in line or "Tests:" in line:
                parsed_output.append(line)

        return "\n".join(parsed_output)

    def run(self, goals: List[str]) -> str:
        user_input = (
            "<</SYS>>\n\nYou are at the first step. Determine which next command to use, "
            "and respond using the json as specified in Response Format section.[/INST]\n\nResponse:"
        )

        # Interaction Loop
        loop_count = 0

        while True:
            human_message = ""

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

            start_index = assistant_reply.find('{')
            end_index = assistant_reply.rfind('}')

            extracted_assistant_reply = assistant_reply.strip()
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
                    user_input = (
                        f"<</SYS>>\n\n{assistant_reply}\n"
                        f"The response is not a valid json. Determine the next step "
                        f"and respond using the json format as specified in Response Format section.[/INST]\n\nResonse:"
                    )
                    continue

            if parsed:
                try:
                    print(f'\033[92mThought:\033[0m {parsed["thoughts"]["text"]}')
                    print(f'\033[92mReasoning:\033[0m {parsed["thoughts"]["reasoning"]}')
                    print(f'\033[92mCriticism:\033[0m {parsed["thoughts"]["criticism"]}')
                    print(f'\033[92mDone:\033[0m\n{parsed["thoughts"]["done"]}')
                    print(f'\033[92mPlan:\033[0m {parsed["thoughts"]["plan"]}')
                    print(f'\033[92mTBDs:\033[0m\n{parsed["thoughts"]["tbds"]}')
                    if parsed["command"]["name"] == "cli":
                      commands = parsed['command']['args']['commands']
                      command_str = " && ".join(commands) if isinstance(commands, list) else commands
                      print(f"\033[92mAction:\033[0m executing cli commands '{command_str}'")

                except KeyError as e:
                  print(f"Missing key: {e}")
                  print(assistant_reply)
                  user_input = (
                      f"<</SYS>>\n\n{assistant_reply}\n"
                      f"The response is missing the key '{e}'. Determine the next step "
                      f"and respond using the json format as specified in Response Forwat section.[/INST]\n\nResponse:"
                  )
                  continue

            self.chat_history_memory.add_message(HumanMessage(content=user_input))
            self.chat_history_memory.add_message(AIMessage(content=json.dumps(parsed)))

            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)

            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
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

                if action.name == "cli":
                    if 'npm test' in command_str:
                        summarized_observation = self.parse_npm_test_output(observation)

                        # print(f'-----------\n{observation}\n---------')

                        if 'FAIL' in summarized_observation:
                            human_message = "However, the tests have failed. You can fix it. "
                        else:
                            human_message = "All tests have passed. Good job! "
                    else:
                        summarized_observation = self.summarize_text(observation)
                else:
                    summarized_observation = observation

                result = f"The {tool.name} tool returned: {summarized_observation}"

            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                )

            parsed_memory_to_add = {
                "Step": loop_count,
                "Thought": parsed['thoughts']['text'],
                "Reasoning": parsed['thoughts']['reasoning'],
                "Criticism": parsed['thoughts']['criticism'],
                "Done": f'\n{parsed["thoughts"]["done"]}',
                "Plan": f'{parsed["thoughts"]["plan"]}',
                "TBDs": f'\n{parsed["thoughts"]["tbds"]}',
            }

            code_str = ""
            if parsed["command"]["name"] == "read_file":
                code_str = f"\n```\n// {parsed['command']['args']['file_path']}\n{observation}\n```"

                parsed_memory_to_add["Action"] = f'reading file {parsed["command"]["args"]["file_path"]}'

                print(f'\033[92mAction:\033[0m reading file {parsed["command"]["args"]["file_path"]}')
                print(f'\033[92mCode:\033[0m{code_str}\n')
            elif parsed["command"]["name"] == "write_file":
                code_str = f"\n```\n// {parsed['command']['args']['file_path']}\n{parsed['command']['args']['text']}\n```"

                parsed_memory_to_add["Action"] = f'writing file {parsed["command"]["args"]["file_path"]}'

                print(f'\033[92mAction:\033[0m writing file {parsed["command"]["args"]["file_path"]}')
                print(f'\033[92mCode:\033[0m{code_str}\n')
            elif parsed["command"]["name"] == "cli":
                parsed_memory_to_add["Action"] = f"executing cli commands '{command_str}'"
                parsed_memory_to_add["Result"] = f"\n{summarized_observation}"

                print(f'\033[92mResult:\033[0m\n{summarized_observation}\n')

            memory_to_add = '\n'.join([f'{k}: {v}' for k, v in parsed_memory_to_add.items()])

            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                memory_to_add += f"\nFeedback: {feedback}"

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.chat_history_memory.add_message(SystemMessage(content=result, additional_kwargs={'metadata': memory_to_add, 'code': code_str}))

            user_input = (
                f"<</SYS>>\n\nYou have completed step {loop_count}. {human_message}"
                f"Determine the next step and respond using the json specified in Response Format section.[/INST]\n\nResponse:"
            )
