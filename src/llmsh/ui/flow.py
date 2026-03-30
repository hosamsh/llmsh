from __future__ import annotations 

from typing import TYPE_CHECKING 

if TYPE_CHECKING :
    from llmsh .ui .screens import MainScreen 


class InteractiveFlow :
    async def start (self ,screen :MainScreen )->None :
        """Post initial system message(s) and set up state."""
        raise NotImplementedError 

    async def handle_input (self ,text :str ,screen :MainScreen )->bool :
        """Process one user input. Return True if flow is complete."""
        raise NotImplementedError 

    @property 
    def placeholder (self )->str :
        """Placeholder text for the chat input during this flow."""
        return "Type your response..."

    async def cancel (self ,screen :MainScreen )->None :
        """Called on Escape or /cancel during a flow. Clean up."""
        pass 
