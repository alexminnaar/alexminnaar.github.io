---
layout: post
title: "RepoGPT: Improving Question Answering Over Code Repositories by using Contextual Chunking"
date: 2020-08-07
comments: false
categories: 
---

tl;dr [RepoGPT](https://github.com/alexminnaar/RepoGPT) is an LLM-based project for question answering over code repositories that exploits contextual chunking for improved QA performance
chunking

You may have seen [LangChain's code understanding demo](https://python.langchain.com/docs/use_cases/code/code-analysis-deeplake) 
where 

1. A code repository is crawled.
2. The files are split into chunks and converted to embeddings. 
3. The chunks and embeddings are stored in a vector database.  

Then questions can be asked by 
1. Retrieving code chunks that are similar to the query from the vector database.
2. Creating a prompt containing the query and the similar code chunks.
3. Submitting the prompt to the LLM and get a (hopefully correct) result.

In the demo the LangChain repo itself is used as the repo to query against and you see some impressively correct responses to questions like

* "What is the class hierarchy?"
* "What classes are derived from the Chain class?"
* "What classes and functions in the ./langchain/utilities/ forlder are not covered by unit tests?"

However, if you have experimented with this demo yourself you know that it can yield some low quality responses as well. If you
look into the queries that produce low quality responses you will notice the following general causes.

1.  The similar chunks coming from the vector database do not contain the required information to answer the query.
2.  The similar chunks do contain the required information however the LLM does not understand how the chunks relate to each other.

These two causes essentially describe a search problem.  [RepoGPT](https://github.com/alexminnaar/RepoGPT) attempts to tackle this search problem and in this way
improve question answering over code repositories.  

## The Search Problem

Let's take the following example query for the LangChain code repo
```text
show the embed_documents method in the AlephAlphaAsymmetricSemanticEmbedding class
```

The 
actual `embed_documents` method in the `AlephAlphaAsymmetricSemanticEmbedding` class in the LangChain repo at the time it 
was cloned is

```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """Call out to Aleph Alpha's asymmetric Document endpoint.

    Args:
        texts: The list of texts to embed.

    Returns:
        List of embeddings, one for each text.
    """
    try:
        from aleph_alpha_client import (
            Prompt,
            SemanticEmbeddingRequest,
            SemanticRepresentation,
        )
    except ImportError:
        raise ValueError(
            "Could not import aleph_alpha_client python package. "
            "Please install it with `pip install aleph_alpha_client`."
        )
    document_embeddings = []

    for text in texts:
        document_params = {
            "prompt": Prompt.from_text(text),
            "representation": SemanticRepresentation.Document,
            "compress_to_size": self.compress_to_size,
            "normalize": self.normalize,
            "contextual_control_threshold": self.contextual_control_threshold,
            "control_log_additive": self.control_log_additive,
        }

        document_request = SemanticEmbeddingRequest(**document_params)
        document_response = self.client.semantic_embed(
            request=document_request, model=self.model
        )

        document_embeddings.append(document_response.embedding)

    return document_embeddings
```
so the above code snippet is what should be returned by the query _"show the embed_documents method in the AlephAlphaAsymmetricSemanticEmbedding class"_.
If you enter this query into the setup defined by the demo mentioned earlier, the LLM response is 

```
The provided context does not include the specific implementation of the `embed_documents` method for the 
`AlephAlphaAsymmetricSemanticEmbedding` class. However, it does show the abstract method from the `Embeddings` 
interface that `AlephAlphaAsymmetricSemanticEmbedding` would need to implement:

'''python
from abc import ABC, abstractmethod
from typing import List

class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        ...
'''

In this abstract class, `embed_documents` is a method that takes a list of strings (`texts`) and returns a list of lists of floats. The specific implementation would depend on how `AlephAlphaAsymmetricSemanticEmbedding` class embeds documents. 
```
which is a somewhat useful answer but obviously not correct in the sense that it did not show the actual method we were
looking for.  As the answer suggests, it is incorrect because the chunks corresponding to the `embed_documents` method 
in the `AlephAlphaAsymmetricSemanticEmbedding` class were not present in the chunks retrieved from the vector store for 
the given query.  Therefore, the LLM did not have the information required to be able to answer the question correctly.
This is where [RepoGPT](https://github.com/alexminnaar/RepoGPT) comes in.

## Adding Contextual Chunking

The problem with the demo setup is that once the code is chunked and indexed into the vector store, all the context around
the chunk is lost.  For example, here is a code chunk that the LLM might see.

```
        Returns:
            List of embeddings, one for each text.
        """
        try:
            from aleph_alpha_client import (
                Prompt,
                SemanticEmbeddingRequest,
                SemanticRepresentation,
            )
        except ImportError:
            raise ValueError(
                "Could not import aleph_alpha_client python package. "
                "Please install it with `pip install aleph_alpha_client`."
            )
        document_embeddings = []

        for text in texts:
            document_params = {
                "prompt": Prompt.from_text(text),
                "representation": SemanticRepresentation.Document,
                "compress_to_size": self.compress_to_size,
                "normalize": self.normalize,
                "contextual_control_threshold": self.contextual_control_threshold,
                "control_log_additive": self.control_log_additive,
            }
```

From the LLM's perspective it's just looking at a random code chunk having no understanding where 
it came from or how it fits together with the larger repo (it's not even explicitly told that it's a chunk of python code).  But what if the LLM had access to both the code chunks and some contextual 
information around them?  This would allow the LLM to answer more complicated queries.  Furthermore, if the additional 
context gets indexed along with the code chunk this could help with the retrieval step as well.  For example, what if a 
code chunk has no clear semantic similarity to the query but the context of that code chunk does?  In this case the relevant
code chunk would now be retrieved whereas before, without the context, it would not.  But what pieces of context should we
add to each code chunk?

* __The File Path:__ If the LLM has access to the file path of the code chunk then it could potentially answer questions related to the repo directory structure.
* __The Chunk Line Numbers:__  If the LLM knows the starting and ending lines of the code chunk then it can know where the chunk appears in the file and could also potentially have the ability to combine adjacent chunks.
* __The Methods Contained in the Chunk:__  If the LLM knows which methods are contained in the code chunk then it could potentially understand full methods that span multiple chunks.  This could also help with vector store retrieval as a query may mention a method name however a code chunk implementing that method may not have the method name explicitly mentioned - adding the method name context would fix that.
* __The Classes Contained in the Chunk:__  Adding the class names defined in the code chunks would also be helpful for the same reasons as adding the method names.

Here is an example of the same code chunk with the contextual information added by [RepoGPT](https://github.com/alexminnaar/RepoGPT).
```
The following code snippet is from a file at location /langchain/langchain/embeddings/aleph_alpha.py 
starting at line 74 and ending at line 98.   The last class defined before this snippet was called 
`AlephAlphaAsymmetricSemanticEmbedding` starting at line 9 and ending at line 142.  The last method starting before this 
snippet is called `embed_documents` which starts on line 68 and ends at line 107. The code snippet starting at line 
74 and ending at line 98 is 
'''
        Returns:
            List of embeddings, one for each text.
        """
        try:
            from aleph_alpha_client import (
                Prompt,
                SemanticEmbeddingRequest,
                SemanticRepresentation,
            )
        except ImportError:
            raise ValueError(
                "Could not import aleph_alpha_client python package. "
                "Please install it with `pip install aleph_alpha_client`."
            )
        document_embeddings = []

        for text in texts:
            document_params = {
                "prompt": Prompt.from_text(text),
                "representation": SemanticRepresentation.Document,
                "compress_to_size": self.compress_to_size,
                "normalize": self.normalize,
                "contextual_control_threshold": self.contextual_control_threshold,
                "control_log_additive": self.control_log_additive,
            }
'''            
``` 
With all this new information, the LLM now knows
where the chunk lives in the repo, its starting and ending lines, and what methods and classes it is part of.

So now let's apply the earlier query to RepoGPT.  In case you forgot, the query was

```text
show the embed_documents method in the AlephAlphaAsymmetricSemanticEmbedding class
```

The response from RepoGPT is the following

```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """Embed search docs."""

    try:
        from aleph_alpha_client import (
            Prompt,
            SemanticEmbeddingRequest,
            SemanticRepresentation,
        )
    except ImportError:
        raise ValueError(
            "Could not import aleph_alpha_client python package. "
            "Please install it with `pip install aleph_alpha_client`."
        )
    document_embeddings = []

    for text in texts:
        document_params = {
            "prompt": Prompt.from_text(text),
            "representation": SemanticRepresentation.Document,
            "compress_to_size": self.compress_to_size,
            "normalize": self.normalize,
            "contextual_control_threshold": self.contextual_control_threshold,
            "control_log_additive": self.control_log_additive,
        }
        document_request = SemanticEmbeddingRequest(**document_params)
        document_response = self.client.semantic_embed(
            request=document_request, model=self.model
        )

        document_embeddings.append(document_response.embedding)

    return document_embeddings
```
As you can see, this response is very close to being correct, it's just the docstring that is slightly off.  This response
could only be achieved because of the additional context which allowed the vector store to retrieve the correct chunks and
the LLM to combine the chunks based on the line numbers and the class/method contextual information.

Here are some further examples of [RepoGPT's](https://github.com/alexminnaar/RepoGPT) ability to answer questions that the original demo could not (all using 
the gpt-4 LLM, chunk size of 1000 and 0 chunk overlap).

## Example 1

__Query:__

```text
show the from_llm_and_tools method in the conversationalagent class
```

__Demo Response:__

```text
 I'm sorry, but the provided context does not contain a `from_llm_and_tools` method in the `ConversationalChatAgent` class.
```

__RepoGPT Response:__

```text
Here is the `from_llm_and_tools` method in the `ConversationalAgent` class: 

'''python
@classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            ai_prefix=ai_prefix,
            human_prefix=human_prefix,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser(
            ai_prefix=ai_prefix
        )
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            ai_prefix=ai_prefix,
            output_parser=_output_parser,
            **kwargs,
        )
'''
```

## Example 2:

__Query:__

```text
Show the methods in the KNNRetriever class
```

__Demo Response:__

```text
The KNNRetriever class has the following methods:

1. create_index(contexts: List[str], embeddings: Embeddings) -> np.ndarray - This is a static method that creates an index of embeddings for a list of contexts.

Please note that the code snippets provided do not show the complete class definition for KNNRetriever, and hence, all methods may not be listed here. 
```
* note that this is actually wrong, the `create_index` method is not actually part of the `KNNRetriever` class (however it is defined in the same file).

__RepoGPT Response:__

```text
The methods in the `KNNRetriever` class are:

1. `from_texts`: 
    ```python
    @classmethod
    def from_texts(
        cls, texts: List[str], embeddings: Embeddings, **kwargs: Any
    ) -> KNNRetriever:
        index = create_index(texts, embeddings)
        return cls(embeddings=embeddings, index=index, texts=texts, **kwargs)
    ```

2. `_get_relevant_documents`:
    ```python
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_embeds = np.array(self.embeddings.embed_query(query))
        # calc L2 norm
        index_embeds = self.index / np.sqrt((self.index**2).sum(1, keepdims=True))
        query_embeds = query_embeds / np.sqrt((query_embeds**2).sum())

        similarities = index_embeds.dot(query_embeds)
        sorted_ix = np.argsort(-similarities)

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator

        top_k_results = [
            Document(page_content=self.texts[row])
            for row in sorted_ix[0 : self.k]
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            )
        ]
        return top_k_results
    ```

3. `_aget_relevant_documents`: 
    ```python
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError
    ```
Please note that `aget_relevant_documents` returns a NotImplementedError, indicating it's a method to be implemented in a child class.
```

## Example #3:

__Query:__

```text
show the validate_environment method in the BaseOpenAI class
```

__Demo Response:__
```text
I'm sorry, but the provided context doesn't include a `BaseOpenAI` class or its `validate_environment` method.
```

__RepoGPT Response:__
```text
The `validate_environment` method in the `BaseOpenAI` class is shown in the following code snippet:

```python
@root_validator()
def validate_environment(cls, values: Dict) -> Dict:
    """Validate that api key and python package exists in environment."""
    values["openai_api_key"] = get_from_dict_or_env(
        values, "openai_api_key", "OPENAI_API_KEY"
    )
    values["openai_api_base"] = get_from_dict_or_env(
        values,
        "openai_api_base",
        "OPENAI_API_BASE",
        default="",
    )
    values["openai_proxy"] = get_from_dict_or_env(
        values,
        "openai_proxy",
        "OPENAI_PROXY",
        default="",
    )
    values["openai_organization"] = get_from_dict_or_env(
        values,
        "openai_organization",
        "OPENAI_ORGANIZATION",
        default="",
    )
    try:
        import openai
        openai.api_key = values["openai_api_key"]
        if values["openai_api_base"]:
            openai.api_base = values["openai_api_base"]
        if values["openai_organization"]:
            openai.organization = values["openai_organization"]
        if values["openai_proxy"]:
            openai.proxy = {"http": values["openai_proxy"], "https": values["openai_proxy"]} 
    except ImportError:
        raise ImportError(
            "Could not import openai python package. "
            "Please install it with `pip install openai`."
        )
    try:
        values["client"] = openai.Completion
    except AttributeError:
        raise ValueError(
            "`openai` has no `Completion` attribute, this is likely "
            "due to an old version of the openai package. Try upgrading it "
            "with `pip install --upgrade openai`."
        )
    if values["streaming"] and values["n"] > 1:
        raise ValueError("Cannot stream results when n > 1.")
    if values["streaming"] and values["best_of"] > 1:
        raise ValueError("Cannot stream results when best_of > 1.")
    return values
```

## Conclusion

[RepoGPT's](https://github.com/alexminnaar/RepoGPT) contextual chunking approach offers a much more effective methodology for question answering over code repositories compared to
plain chunking as demonstrated in this blog post.  However RepoGPT still isn't perfect - there are still many cases where
RepoGPT is not able to answer a question correctly for multiple reasons.  But as LLMs get more advanced and context lengths get longer
RepoGPT's performance should continue to improve.  If you think RepoGPT sounds interesting, please give it a try at 
https://github.com/alexminnaar/RepoGPT.