from langchain.text_splitter import TextSplitter
from typing import List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import (
    ChatAnthropic,
)  # https://python.langchain.com/v0.2/docs/integrations/chat/anthropic/
from langchain_core.output_parsers import StrOutputParser
import re
import tiktoken


class LLMTextSplitter(TextSplitter):
    """
    Splitting text using a LLM with options for different types of topic splitting.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet",
        prompt_type: str = "wide",
        count_tokens: bool = False,
        encoding_name: str = "cl100k_base",  # need to use the encoder available
        **kwargs: Any,
    ) -> None:
        """
        Initialise the LLM splitter with a model chain, a prompt type, and an optional token counter.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.count_tokens = count_tokens
        self.encoding_name = encoding_name
        self.model = ChatAnthropic(model_name=self.model_name)
        self.output_parser = StrOutputParser()

        # define two types of promt templates
        wide_topic_template = "Split the text according to the broad topics it deals with and add >>> and <<< around chunk: '{text}'"
        granular_topic_template = "Split the textinto detailed, granular topics and add >>> and <<< around each chunk: '{text}'"

        # Choose the prompt template based on the provided type
        if prompt_type == "wide":
            self.prompt_template = ChatPromptTemplate.from_template(wide_topic_template)
        elif prompt_type == "granular":
            self.prompt_template = ChatPromptTemplate.from_template(
                granular_topic_template
            )
        else:
            raise ValueError(
                "Invalid prompt type specified. Choose 'wide' or 'granular'."
            )

        self.chain = self.prompt_template | self.model | self.output_parser

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def split_text(self, text: str) -> List[str]:

        if self.count_tokens:
            token_count = self.num_tokens_from_string(text)
            print(f"Toekn count of input texst: {text_count}")

        response = self.chain.invoke({"text": text})
        return self._format_chunks(response)

    def _format_chuns(self, text: str) -> List[str]:
        pattern = r">>>(.*?)<<<"
        chunks = re.findall(pattern, text, re.DOTALL)
        formatted_chunks = [chunk.strip() for chunk in chunks]
        return formatted_chunks


###
if __name__ == "__main__":

    #######
    with open("./data.txt") as f:
        data = f.read()
    #######

    splitter = LLMTextSplitter(count_tokens=True, prompt_type="wide")
    chunks = splitter.split_text(data)

    docs = splitter.create_documents([data])
