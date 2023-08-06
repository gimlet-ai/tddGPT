from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.document import Document

class TextSummarizer:
    def __init__(self, summary_type: str):
        # Define the prompt based on the summary_type
        if summary_type == "cli":
            prompt_template = """The following is the output of a cli command:

            <output>
            {text}
            </output>

            Please summarize it in one paragraph and highlight the errors. Ignore any warnings, security
            vulnerabilities, dependencies or audit issues. Start with 'The cli command was <status>'.
            """

        elif summary_type == "memory":
            prompt_template = """Please summarize the following log in one paragraph. 

            <log>
            {text}
            </log>

            Start with 'Step <num>: ' and separate multiple steps with a newline.
            """

        else:
            raise ValueError("Invalid summary type")

        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)
        self.stuff_chain = StuffDocumentsChain(llm_chain=self.llm_chain, document_variable_name="text")

    def summarize(self, text: str) -> str:
        docs = [Document(page_content=text)]
        result = self.stuff_chain.run(docs)
        return result
