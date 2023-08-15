from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
import textwrap

class TextSummarizer:
    def __init__(self, summary_type: str):
        # Define the prompt based on the summary_type
        if summary_type == "cli":
            prompt_template = textwrap.dedent("""The following is the output of a cli command:

            <output>
            {text}
            </output>

            Please summarize it in one paragraph and highlight the errors. Skip any warnings, security
            vulnerabilities, dependencies or audit issues. For npm test output, describe the error in detail 
            and include the exact lines of code where the error occurred in the summary. For file reads, 
            include the relevant lines of code as is. Skip any disclaimers about original summary/context. 
            Start with 'The cli command was <status> ' where status is success/failure.
            """)

        elif summary_type == "memory":
            prompt_template = textwrap.dedent("""The following is the log of steps already completed. 

            <output>
            {text}
            </output>

            Summarize the steps one by one. Make it progressive: include more details for the later steps.
            Skip any warnings, vulnerabilities, dependencies or audit issues. For npm test output, 
            describe the error in detail and include the exact lines of code where the error occurred. 
            Preserve Thought and File Content as is for the read file action. Include all the steps in summary. 
            Start with 'Step <num> (<status>): ' where status is success/failure.
            """)

        else:
            raise ValueError("Invalid summary type")

        prompt = PromptTemplate.from_template(prompt_template)
        refine_template = textwrap.dedent("""Your job is to produce a final summary.
        We have provided an existing summary up to a certain point: 
                                          
        <original>
        {existing_answer}
        </original>

        We have the opportunity to refine the existing summary (only if needed) with some more context below.

        <context> 
        {text}
        </context>

        Given the new context, refine the original summary.  If the context isn't useful, return the original summary.
        Do not include any meta comments about original summary/context.
        """)

        refine_prompt = PromptTemplate.from_template(refine_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        self.llm_chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=False,
            input_key="input_documents",
            output_key="output_text",
        )

    def summarize(self, text: str) -> str:
        docs = [Document(page_content=text)]
        if len(text) > 3000:
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
            split_docs = text_splitter.split_documents(docs)
        else:
            split_docs = docs
        result = self.llm_chain.run({"input_documents": split_docs})
        return result
