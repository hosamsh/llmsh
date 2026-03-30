from __future__ import annotations 

from pathlib import Path 

from textual .app import App 

from llmsh .app import AppCore 
from llmsh .ui .screens import MainScreen 


class LlmshApp (App ):
    CSS_PATH =Path (__file__ ).parent /"theme.tcss"

    def __init__ (self ,core :AppCore |None =None )->None :
        super ().__init__ ()
        self .core =core 

    def on_mount (self )->None :
        self .push_screen (MainScreen ())
