from __future__ import annotations 

from textual .widgets import Markdown as TextualMarkdown 


def render_markdown (text :str )->TextualMarkdown :
    return TextualMarkdown (text )
