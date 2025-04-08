---
layout: post
title: "How to Nicely Format Streaming OpenAI API Output for the Web"
date: 2025-04-07
comments: false
categories:
---

When I was building [HinterviewGPT](https://hinterviewgpt.com/), I was pretty surprised to discover that one of the most difficult parts was making the
raw OpenAI API output look nice on the screen.  Everyone has used ChatGPT at this point and users are accustomed to
seeing nicely formatted text, code and even math.  In order for a product to look professional, the output formatting
must at least be on par with ChatGPT.  HinterviewGPT is a React app and I soon learned that this is not as easy as I
had first assumed.

First some context, HinterviewGPT is a web app that serves as a platform to help users prepare for job interviews.  It
allows users to generate interview questions and also study them with a personal AI tutor.  Both of these features require
nicely formatted ChatGPT API formatted output.

Out-of-the-box, the OpenAI API's output is in markdown format.  In order for this to be shown on an actual web page this
markdown needs to be converted to HTML.  Also, since I was using the streaming API, this formatting needed to be done
in a streaming fashion.  After a lot of struggling i.e. trying a bunch of React libraries where some
  worked, some didn't, some seemed to work only to realize days later that they caused a memory leak... I finally made it work.
In this blog post I'll refer to this [instructive example repo](https://github.com/alexminnaar/llm-output-formatter/tree/main) I created which is a simple React project that renders markdown
from streaming OpenAI API output.

So let's begin with text formatting.

## Text Formatting

The backbone for markdown formatting in my project was [react-markdown](https://github.com/remarkjs/react-markdown).  Text
formatting works well out-of-the-box using this component.  Here is what text formatting looks like from my example repo

<div style="text-align:center">
<img src="{{site.baseurl}}/assets/formatted_text.png" width="100%" height="100%"     
style="border: 2px solid #000; border-radius: 4px;"
>
</div>

To be clear, the left pane has a text input where the prompt is entered and when the submit button is clicked the OpenAI
API is called in streaming mode using the entered prompt.  The resulting markdown stream is passed through the formatter
and displayed on the right pane.  As you can see, the text, headings, and bullet lists are formatted correctly.  And of
course the styling can be changed to your liking.

The right pane has the following structure.

```javascript
<div>
   <h3>OpenAI Response (Formatted)</h3>
   <div>
      <ReactMarkdown
         children={markdownOutput}
         remarkPlugins={[]}
         rehypePlugins={[]}
      />
   </div>
</div>
```

Here `markdownOutput` is the output from the OpenAI API - so it is simply passed into the `children` parameter.

## Code Formatting

Next up is code formatting.  This basically means making the generated code look nice and have syntax highlighting.  This
requires the `rehype-highlight` plugin for `react-markdown`.  Once installed, add it to the `rehypePlugins` parameter.

```
<div>
   <h3>OpenAI Response (Formatted)</h3>
   <div>
      <ReactMarkdown
         children={markdownOutput}
         remarkPlugins={[]}
         rehypePlugins={[rehypeHighlight]}
      />
   </div>
</div>
```

The result looks like the following.

<div style="text-align:center">
<img src="{{site.baseurl}}/assets/code_formatting.png" width="100%" height="100%"
style="border: 2px solid #000; border-radius: 4px;"
>
</div>

As you can see the code is displayed with syntax highlighting as desired.  You can also experiment with different highlighting
plugins if you don't like the look (e.g. `rehype-prism`).


## Math Formatting

Finally we also need to be able to display generated math content.  Unfortunately this is the trickiest one.  For this
we need two plugins - `rehype-math` and `rehype-katex`.  `rehype-math` detects the math parts of the generated markdown
and converts it into an AST (abstract syntax tree), then `rehype-katex` converts the AST into something that looks like
LaTeX formatting in the browser.  The code looks like the following.

```
<div>
   <h3>OpenAI Response (Formatted)</h3>
   <div>
      <ReactMarkdown
         children={markdownOutput}
         remarkPlugins={[remarkMath]}
         rehypePlugins={[rehypeKatex, rehypeHighlight]}
      />
   </div>
</div>
```

And the result looks like the following.

<div style="text-align:center">
<img src="{{site.baseurl}}/assets/math_formatting_bad.png" width="100%" height="100%"
style="border: 2px solid #000; border-radius: 4px;"
>
</div>

Hey, that doesn't look right!  The reason for this is the OpenAI API generates math with inline statements like `\(...\)`
or block statements like `\[...\]` but `remark-math` only recognizes `$...$` and `$$...$$` statements.  To fix this we
need to add a pre-processing step that converts from the former to the latter.

```javascript
const preprocessLaTeX = (content) => {
    return content
      .replace(/\\\[(.*?)\\\]/gs, (_, eq) => `$$${eq}$$`)   // block math
      .replace(/\\\((.*?)\\\)/gs, (_, eq) => `$${eq}$`);    // inline math
};
```

So let's change the code to incorporate this pre-processing step.

```
<div>
   <h3>OpenAI Response (Formatted)</h3>
   <div>
      <ReactMarkdown
         children={preprocessLaTeX(markdownOutput)}
         remarkPlugins={[remarkMath]}
         rehypePlugins={[rehypeKatex, rehypeHighlight]}
      />
   </div>
</div>
```

and the result is

<div style="text-align:center">
<img src="{{site.baseurl}}/assets/math_formatting_good.png" width="100%" height="100%"
style="border: 2px solid #000; border-radius: 4px;"
>
</div>

That looks much better.  Using regex is always a bit risky here but unfortunately it is required since the way the 
OpenAI API generates math seems to be incompatible with the way `remark-math` parses it.

To summarize, in this blog post we discussed how to render streaming OpenAI API markdown output into nicely formatted
HTML for text, code and math content.  Again, in this blog post I have referred to this [example repo](https://github.com/alexminnaar/llm-output-formatter/tree/main),
specifically the [App.js](https://github.com/alexminnaar/llm-output-formatter/blob/main/src/App.js) file.