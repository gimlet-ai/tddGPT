from __future__ import annotations
from typing import List, Optional
from pydantic import ValidationError
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain_experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from prompt import DevGPTPrompt
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


class DevGPTAgent:
    """Agent class for interacting with Dev-GPT."""

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
        self.memory_list = []
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
    ) -> DevGPTAgent:
        prompt = DevGPTPrompt(
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

        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1

            # Send message to AI, get response
            assistant_reply = self.chain.run(
                goals=goals,
                messages=self.chat_history_memory.messages,
                memory=self.memory_list,
                user_input=user_input,
            )

            print(f"\033[91mStep Number:\033[0m {loop_count}")

            try:
                parsed = json.loads(assistant_reply, strict=False)
            except json.JSONDecodeError:
                preprocessed_text = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", assistant_reply)
                try:
                    parsed = json.loads(preprocessed_text, strict=False)
                except Exception:
                    print(assistant_reply)

            if parsed:
                try:
                    print(f'\033[92mText:\033[0m {parsed["thoughts"]["text"]}')
                    print(f'\033[92mReasoning:\033[0m {parsed["thoughts"]["reasoning"]}')
                    print(f'\033[92mPlan:\033[0m\n{parsed["thoughts"]["plan"]}')
                    print(f'\033[92mCriticism:\033[0m {parsed["thoughts"]["criticism"]}')
                    print(f'\033[92mSpeak:\033[0m {parsed["thoughts"]["speak"]}')
                    if parsed["command"]["name"] == "read_file":
                        print(f'\033[92mAction:\033[0m reading file {parsed["command"]["args"]["file_path"]}')
                    elif parsed["command"]["name"] == "write_file":
                        print(f'\033[92mAction:\033[0m writing file {parsed["command"]["args"]["file_path"]}\n')
                        print(f"{parsed['command']['args']['text']}\n")
                    elif parsed["command"]["name"] == "cli":
                        commands = parsed['command']['args']['commands']
                        command_str = "\n".join(commands) if isinstance(commands, list) else commands
                        print(f'\033[92mAction:\033[0m executing cli commands\n' + command_str + "\n")
                except KeyError as e:
                  print(f"Missing key: {e}")

            self.chat_history_memory.add_message(HumanMessage(content=user_input))
            self.chat_history_memory.add_message(AIMessage(content=assistant_reply))

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

                summarized_observation = self.summarize_text(observation) if action.name == "cli" else observation

                result = f"The {tool.name} tool returned: {summarized_observation}"

                print(f'\033[92mResult:\033[0m {result}')

            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                )

            parsed_memory_to_add = [
                f"Step: {loop_count}",
                f"Thought: {parsed['thoughts']['speak']}",
                f"Action: {action.name}",
            ]

            if parsed["command"]["name"] == "read_file":
                parsed_memory_to_add.append(f"Action: reading file {parsed['command']['args']['file_path']}")
            elif parsed["command"]["name"] == "write_file":
                parsed_memory_to_add.append(f"Action: writing file {parsed['command']['args']['file_path']}")
                parsed_memory_to_add.append(f"{parsed['command']['args']['text']}")
            elif parsed["command"]["name"] == "cli":
                commands = parsed['command']['args']['commands']
                command_str = " && ".join(commands) if isinstance(commands, list) else commands
                parsed_memory_to_add.append(f"Action: executing cli commands {command_str}")

            parsed_memory_to_add_str = '\n'.join(parsed_memory_to_add)
            memory_to_add = f"\n{parsed_memory_to_add_str}\nResult: {result}\n"

            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                memory_to_add += f"\nFeedback: {feedback}"

            self.memory.add_documents([Document(page_content=memory_to_add)])
            self.memory_list.append(Document(page_content=memory_to_add, metadata={"tool": tool.name, "observation": observation}))
            self.chat_history_memory.add_message(SystemMessage(content=result))

