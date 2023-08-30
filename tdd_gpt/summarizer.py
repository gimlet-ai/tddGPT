from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.document import Document
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import StuffDocumentsChain, LLMChain
import textwrap

class TextSummarizer:
    CLI_TEMPLATE = textwrap.dedent("""Summarize the output in the following format.

            <format>
            The commands <succeded/failed> with the message <output summary>.
            </format>

            If, **and only if**, the output is from an 'npm test' command, follow the instructions below to summarize instead.
                                   
            Summarize each test failure in the following format.

            <format>
            Test Case: <test case description>
            Error: <error message>
            File name: <file name>
            Code Snippet: 
            ```
            <code snippet which caused the failure, if any>
            ```
            </format>

            Include test stats at the end (only once) in the following format. 

            <format>
            <num> tests failed out of <num> total tests
            </format>
            
            Ignore any warnings, security vulnerabilities, dependency/audit issues. 
            """)

    MEMORY_TEMPLATE = textwrap.dedent("""Summarize each step concisely and highlight the result.
            For npm test output, describe the error in detail including the file name, line number and code snippets.
            Ignore any suggestions, warnings, security vulnerabilities, dependency/audit issues. 
            Start with '- I successfully executed the <command> ' """)

    def __init__(self, summary_type: str):
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo-16k")

        # Define the prompt based on the summary_type
        prompt_template = self.get_prompt_template(summary_type)

        # MapReduce chain for longer texts
        map_prompt = PromptTemplate.from_template(prompt_template.replace("{text}", "{docs}"))
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        reduce_template = self.get_reduce_template(summary_type)
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="doc_summaries"
        )

        self.reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )

        self.map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=self.reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

    def get_prompt_template(self, summary_type: str) -> str:
        if summary_type == "cli":
            return textwrap.dedent(f"""The following is the output of a cli command:

            <output>
            {{text}}
            </output>

            {self.CLI_TEMPLATE}""")
        elif summary_type == "memory":
            return textwrap.dedent(f"""The following is the log of steps already completed. 

            <output>
            {{text}}
            </output>

            {self.MEMORY_TEMPLATE}""")
        else:
            raise ValueError("Invalid summary type")

    def get_reduce_template(self, summary_type: str) -> str:
        if summary_type == "cli":
            return textwrap.dedent(f"""The following is a set of summaries from CLI commands:

            <summaries>
            {{doc_summaries}}
            </summaries>

            {self.CLI_TEMPLATE}""")
        elif summary_type == "memory":
            return textwrap.dedent(f"""The following is a set of summaries from memory logs:

            <summaries>
            {{doc_summaries}}
            </summaries>

            {self.MEMORY_TEMPLATE}""")
        else:
            raise ValueError("Invalid summary type")

    def summarize(self, text: str, token_max: int = 4000) -> str:
        docs = [Document(page_content=text)]
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)
        self.reduce_documents_chain.token_max = token_max
        result = self.map_reduce_chain.run(split_docs)
        return result
