---
layout: post
title: "How RepoGPT Parses Code"
date: 2023-09-28
comments: false
categories: 
---

If you read the [previous post about RepoGPT](https://alexminnaar.com/2023/08/07/RepoGPT-Improved-Question-Answering-Over-Repositories.html)
it explains how parsing code provides context which greatly improves RepoGPT's ability to answer questions.  Specifically,
extracting the methods and classes associated with the code chunk.  Below is an example code chunk with its associated
context (shown on top of the code chunk).

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

The benefits of this contextual information are two-fold.  First, they help with vector DB retrieval in that this new 
context, when translated into an embedding, allows for more relevant code chunks to be returned.  And second, the context allows
the LLM answer more complex questions.  This post will explain how RepoGPT is able to parse the class and method information
that you see in the context.

## So how do you actually parse code into classes and methods?

Code from any language can be parsed into an AST (Abstract Syntax Tree) data structure which represents the hierarchical 
structure of the code including methods and classes (and much more) and how they relate to one and other.  Methods and 
classes are represented as nodes in the code's AST.  If we can traverse the AST and identify these nodes then we
can extract the information we need (i.e. the class/method's name and their line number spans).  Fortunately we don't 
have to implement the parsers ourselves, we can simply use the [py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter) 
python library.

Below is example python code for traversing an AST using py-tree-sitter

```python
from tree_sitter import Language, Parser

code = "some python code"
parser = Parser()
parser.set_language(Language('build/my-languages.so', 'python'))
tree = parser.parse(bytes(code, "utf-8"))

def traverse(node):

    print(f"node type: {node.type}")
    print(f"node content: {code[node.start_byte:node.end_byte]}")
    print(f"node start line: {node.start_point[0]}, node end line: {node.end_point[0]}")

    for child in node.children:
        traverse(child)

root_node = tree.root_node

traverse(root_node)
```

In the above code snippet the code is parsed according to its language and the resulting tree is recursively traversed node by node.
As you can see each node has children, a type string, a start line and end line and you can also extract the actual code associated
with the node.  Therefore, if we can know the node type associated with classes and methods then it is pretty straightforward
to get what we need.

But here's the tricky part.  Each language has its own grammar which means that each language may have it's own particular
node type associated with classes and methods.  Therefore, there is no generic way of parsing classes and methods that will
work for every language.  Fortunately it's not that difficult to implement a traversal function for each language we're 
interested in .  Below are some example traversal functions for a few languages that will parse out the information we
need.

### Java

For the Java programming language with the py-tree-sitter provided grammar the following `traverse` function does what we need

```python
def traverse(node):
    if node.type == 'constructor_declaration':
        for child in node.children:
            if child.type == 'identifier':
                print(f"method name: {code[child.start_byte: child.end_byte]}, \
                        start line: {node.start_point[0]}, \
                        end line: {node.end_point[0]}")

    if node.type == 'method_declaration':
        for child in node.children:
            if child.type == 'identifier':
                print(f"method name: {code[child.start_byte: child.end_byte]}, \
                        start line: {node.start_point[0]}, \
                        end line: {node.end_point[0]}")

    if node.type == 'class_declaration':
        for child in node.children:
            if child.type == 'identifier':
                print(f"class name: {code[child.start_byte: child.end_byte]}, \
                        start line: {node.start_point[0]}, \
                        end line: {node.end_point[0]}")

    for child in node.children:
        traverse(child)
```

As the above function shows, methods are associated with the node types 'constructor_declaration' and 'method_declaration'
so we identify these during the tree traversal.  Once they are found we traverse one level deeper into the child nodes and
search for the child with node type 'identifier' which contains the name of the function.  In order to get the start and ending
lines of the method we use `start_point[0]` and `end_point[0]` from the parent node.  Similarly, for classes we search
for the node type 'class_declaration' and the child node with type 'identifier'.

### C++

Now let's try parsing the C++ language.

```python
def traverse(node):
    if node.type == 'function_declarator':
        for child in node.children:
            if child.type == 'identifier' or child.type == 'field_identifier':
                print(f"function name: {code[child.start_byte: child.end_byte]}, \
                        start line: {node.start_point[0]}, \
                        end line: {node.end_point[0]}")
                        
    if node.type == 'class_specifier':
        for child in node.children:
            if child.type == 'type_identifier':
                print(f"class name: {code[child.start_byte: child.end_byte]}, \
                        start line: {node.start_point[0]}, \
                        end line: {node.end_point[0]}")

    for child in node.children:
        traverse(child)
```

Note how this looks similar to the Java traversal function however there are some slight differences.  This time method
nodes are of the type 'function_declarator' and the child node type can sometimes be 'field_identifier'.  Also, the class
node type is now called 'class_specifier' and the child node type is called 'type_identifier'.  These differences are due
to the differences in grammars between Java and C++.

### Go

Now let's try the Go language.  Go is different in that it doesn't have classes, instead it has structs and interfaces.
For the purposes of RepoGPT we'll treat structs and interfaces as classes.

```python
def traverse(node, current_line):
    if node.type == 'type_spec':
        for child in node.children:
            if child.type == 'type_identifier':
                print(f"class name: {code[child.start_byte: child.end_byte]}, \
                        start line: {node.start_point[0]}, \
                        end line: {node.end_point[0]}")

    if node.type == 'method_declaration':
        for child in node.children:
            if child.type == 'field_identifier':
                print(f"method name: {code[child.start_byte: child.end_byte]}, \
                        start line: {node.start_point[0]}, \
                        end line: {node.end_point[0]}")

    if node.type == 'function_declaration':
        for child in node.children:
            if child.type == 'identifier':
                print(f"method name: {code[child.start_byte: child.end_byte]}, \
                        start line: {node.start_point[0]}, \
                        end line: {node.end_point[0]}")

    for child in node.children:
        traverse(child, current_line)
```

As you can see the classes are now found in the 'type_spec' node type with the child node called 'type_identifier'.

## Conclusion

Once we have extracted the information about the classes and methods in a file it is quite straight-forward to create the 
code chunk context that was shown at the beginning of this post.  Again, this added chunk context greatly improves RAG 
performance both in the vector store retrieval step and in the LLM response step.  To see how this works in more detail 
checkout the [RepoGPT project](https://github.com/alexminnaar/RepoGPT).