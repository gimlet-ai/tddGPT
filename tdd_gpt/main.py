from agent import TddGPTAgent
from cli import CLITool
from langchain.chat_models import ChatOpenAI
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools import ShellTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory.chat_message_histories import FileChatMessageHistory
import faiss
import argparse
import os

def parse_args():
    default_prompt = "build a flask app with a form to record the name, address, dob, height and weight and store it in a sqlite db" 

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Generate code based on user stories')
    parser.add_argument('--prompt', type=str, default=default_prompt, help='User stories for the app or file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--chat_history_file', type=str, help='Path to chat history file')
    parser.add_argument('--output_dir', type=str, default=os.getcwd(), help='Output directory for the files generated by the agent')

    # Parse the arguments
    return parser.parse_args()

def main():
    # Parse the arguments
    args = parse_args()

    chat_history_memory = None
    if args.chat_history_file:
        chat_history_memory = FileChatMessageHistory(args.chat_history_file)

    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

    tools = [
        CLITool(),
        WriteFileTool(),
        ReadFileTool(),
    ]

    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()

    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # Initialize the agent
    agent = TddGPTAgent.from_llm_and_tools(
        output_dir=args.output_dir,
        tools=tools,
        llm=ChatOpenAI(model='gpt-4-0613', temperature=0),
        memory=vectorstore.as_retriever(),
        chat_history_memory=chat_history_memory,
    )

    # Set verbose to be true if debug argument is passed
    agent.chain.verbose = args.debug

    prompt = args.prompt
    if os.path.isfile(prompt):
        with open(prompt, 'r') as file:
            prompt = file.read()

    print(f'{prompt}\n')
    agent.run([prompt])

if __name__ == "__main__":
    main()
