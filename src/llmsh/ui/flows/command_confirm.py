from __future__ import annotations 

from typing import TYPE_CHECKING 

from llmsh .ui .flow import InteractiveFlow 
from llmsh .ui .slash import show_system_message 

if TYPE_CHECKING :
    from llmsh .ui .screens import MainScreen 


class CommandConfirmFlow (InteractiveFlow ):
    """Confirms and executes a suggested slash command."""

    def __init__ (self ,command :str ,reasoning :str |None =None )->None :
        self ._command =command 
        self ._reasoning =reasoning 

    async def start (self ,screen :MainScreen )->None :
        msg =(
        f"It looks like you meant to type `{self ._command }`."
        " Run this command? (y/n)"
        )
        await show_system_message (screen ,msg )

    @property 
    def placeholder (self )->str :
        return "y to run, n to cancel..."

    async def handle_input (self ,text :str ,screen :MainScreen )->bool :
        if text .strip ().lower ()in ("y","yes"):
            from llmsh .ui .slash import handle_slash_command ,parse_slash_command 

            parsed =parse_slash_command (self ._command )
            if parsed is not None :
                command ,args =parsed 
                await handle_slash_command (command ,args ,screen )
            else :
                await show_system_message (screen ,"Invalid command.")
        else :
            await show_system_message (screen ,"Cancelled.")
        return True 
