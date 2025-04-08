---
layout: post
title: "The Third UI for LLM Applications"
date: 2025-04-07
comments: false
categories: 
---

Ever since GPT-3.5 entered the scene in late 2022, LLM-based applications have exploded in popularity.  Even in this short amount of
time it is interesting to observe how the user interfaces for these applications have evolved.  In my opinion there have
been two main types of UI's that have caught on but there is a third that is starting to emerge.  In this blog post
I'll delve into all three.

## The First UI: ChatGPT

The first UI was the simplest and most recognizable.  It came from [ChatGPT](https://chatgpt.com/) itself and you can see a screenshot of it below.

<div style="text-align:center">
<img src="{{site.baseurl}}/assets/chat_ui.png" width="100%" height="100%">
</div>

This is a simple chat interface where the user enters their message and the LLM responds.  Despite its simplicity, this UI is very
extensible.  Developers could integrate this UI into their own applications with custom LLM behavior.

For example, a developer could use this interface for:
<ul style="margin-left: 20px">
  <li style="font-size:19px">their own proprietary foundational model</li>
  <li style="font-size:19px">an open-source model</li>
  <li style="font-size:19px">a fine-tuned version of an existing model</li>
</ul>

Better yet, a developer could connect the LLM to external knowledge-bases through RAG to build applications like
"ChatGPT for law", "ChatGPT for code repository x", "ChatGPT for your toaster's user's manual", etc.  Developers have
gone even further by connecting this UI to complex agentic systems such that their application can perform any number of
tasks through chat.  I've even seen cases where online forms have been completely replaced with a chat interface that
extracts the required information from the user via chat.

## The Second UI: Copilot

I can't say with any certainty when the second UI was first created or by who but [Microsoft Copilot](https://copilot.microsoft.com/) was one of the earliest
adopters.  You can see a screenshot below.

<div style="text-align:center">
<img src="{{site.baseurl}}/assets/copilot_ui.png" width="100%" height="100%">
</div>

As you can see, with this UI there is both a chat component and something the user would like to chat _about_ (usually
some kind of document).  Generally speaking, this chat component could allow users to both ask questions about the document
and ask the LLM to make specified changes on the user's behalf.

This was an upgrade on _the first UI_ because now users could interact directly with a document rather than simply asking
questions against some corpus of documents.  This new UI essentially ushered in the _copilot era_ where developers built
applications that allows users to do things like write/understand code, emails, pdfs, etc. with the help of an LLM.  Similar
to _the first UI_, developers could also augment the LLM with fine-tuning, RAG, agentic workflows etc. depending on the 
use case.

## The Third UI: HinterviewGPT

[HinterviewGPT](https://hinterviewgpt.com/) offers a unique twist on _the second UI_.  Instead of only offering a chat UI
and a reference document, this _third UI_ offers an additional component where the user can enter their own information.

For example, in the case of HinterviewGPT, the reference document is an interview question, and the additional component
is a virtual whiteboard where the user can write their solution - see the below screenshot for clarity.

<div style="text-align:center">
<img src="{{site.baseurl}}/assets/hinterviewgpt.png" width="100%" height="100%">
</div>

The chat UI is context-aware of both the interview question and the contents of the whiteboard which allows the LLM to 
act as a tutor, helping the user understand the question and how their current solution on the whiteboard might be 
misguided.

In effect, this _third UI_ allows the user to actively __co-create__ alongside the LLM.  Specifically, it introduces a live
user-generated context that the LLM can continuously interpret and respond to.

As LLM applications continue to evolve, I believe this _third UI_ — co-creative, context-rich, and user-driven — will become increasingly important for serious learning, creativity, and productivity tools.



