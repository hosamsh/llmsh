from __future__ import annotations 

from textual .app import ComposeResult 
from textual .containers import VerticalScroll 
from textual .message import Message 
from textual .widget import Widget 
from textual .widgets import Input ,Static 
from textual .widgets import Markdown as TextualMarkdown 

from llmsh .markdown import render_markdown 
from llmsh .models import ChatMessage 


class MessageWidget (Widget ):
    def __init__ (self ,role :str ,content :str )->None :
        if role =="user":
            css_class ="user-message"
        elif role =="system":
            css_class ="system-message"
        else :
            css_class ="assistant-message"
        super ().__init__ (classes =css_class )
        self ._role =role 
        self ._content =content 

    def compose (self )->ComposeResult :
        if self ._role =="system":
            yield Static (self ._content )
        elif self ._role =="assistant":
            yield Static ("[bold]Assistant:[/bold] ",classes ="role-label")
            yield render_markdown (self ._content )
        else :
            yield Static (f"[bold]You:[/bold] {self ._content }")

    @property 
    def content (self )->str :
        return f"[{self ._role }]\n{self ._content }"

    def append_text (self ,delta :str )->None :
        self ._content +=delta 
        md_widgets =self .query (TextualMarkdown )
        if md_widgets :
            md_widgets .first ().update (self ._content )


class TranscriptPane (VerticalScroll ):
    def __init__ (
    self ,
    name :str |None =None ,
    id :str |None =None ,
    classes :str |None =None ,
    disabled :bool =False ,
    )->None :
        super ().__init__ (name =name ,id =id ,classes =classes ,disabled =disabled )
        self ._active_message :MessageWidget |None =None 
        self ._reasoning_widget :Static |None =None 
        self ._reasoning_text :str =""

    async def add_message (self ,msg :ChatMessage )->None :
        widget =MessageWidget (role =msg .role ,content =msg .content )
        await self .mount (widget )
        self .scroll_end (animate =False )

    def begin_streaming (self )->None :
        if self ._active_message is not None :
            self ._active_message .remove ()
        widget =MessageWidget (role ="assistant",content ="")
        widget .add_class ("streaming")
        self ._active_message =widget 
        self .mount (widget )
        self .scroll_end (animate =False )

    def append_text (self ,delta :str )->None :
        if self ._active_message is not None :
            self ._active_message .append_text (delta )
            self .scroll_end (animate =False )

    def end_streaming (self )->None :
        if self ._active_message is not None :
            self ._active_message .remove_class ("streaming")
            self ._active_message =None 

    def begin_reasoning (self )->None :
        """Start a reasoning/thinking block above the response."""
        self ._reasoning_text ="\U0001f4ad *Thinking...*\n"
        widget =Static (self ._reasoning_text ,classes ="reasoning-block")
        self ._reasoning_widget =widget 
        self .mount (widget )
        self .scroll_end (animate =False )

    def append_reasoning (self ,delta :str )->None :
        """Append text to the active reasoning block."""
        if self ._reasoning_widget is not None :
            self ._reasoning_text +=delta 
            self ._reasoning_widget .update (self ._reasoning_text )
            self .scroll_end (animate =False )

    def end_reasoning (self )->None :
        """Finalize the reasoning block."""
        self ._reasoning_widget =None 
        self ._reasoning_text =""


class SubmitMessage (Message ):
    def __init__ (self ,text :str )->None :
        super ().__init__ ()
        self .text =text 


class ChatInput (Input ):
    def on_input_submitted (self ,event :Input .Submitted )->None :
        text =event .value .strip ()
        event .stop ()
        if text :
            self .post_message (SubmitMessage (text ))
        self .value =""
