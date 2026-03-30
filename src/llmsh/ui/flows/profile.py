from __future__ import annotations 

import re 
from typing import TYPE_CHECKING 

from llmsh .config import AppConfig ,save_config 
from llmsh .models import ModelCapabilities ,ProfileConfig 
from llmsh .ui .flow import InteractiveFlow 
from llmsh .ui .slash import show_system_message 

if TYPE_CHECKING :
    from llmsh .ui .screens import MainScreen 


class ProfileAddFlow (InteractiveFlow ):
    def __init__ (self ,config :AppConfig )->None :
        self ._config =config 
        self ._name =""
        self ._endpoint =""
        self ._model =""
        self ._step =0 

    @property 
    def placeholder (self )->str :
        placeholders ={
        1 :"e.g., my-profile",
        2 :"endpoint name (e.g., local)",
        3 :"e.g., gpt-4",
        }
        return placeholders .get (self ._step ,"Type your response...")

    async def start (self ,screen :MainScreen )->None :
        await show_system_message (screen ,"Name for this profile?")
        self ._step =1 

    async def handle_input (self ,text :str ,screen :MainScreen )->bool :
        text =text .strip ()
        if self ._step ==1 :
            return await self ._handle_name (text ,screen )
        elif self ._step ==2 :
            return await self ._handle_endpoint (text ,screen )
        elif self ._step ==3 :
            return await self ._handle_model (text ,screen )
        return True 

    async def _handle_name (self ,text :str ,screen :MainScreen )->bool :
        if not re .match (r"^[a-zA-Z0-9-]+$",text ):
            await show_system_message (
            screen ,"Name must be alphanumeric (hyphens allowed)."
            )
            return False 
        if text in self ._config .profiles :
            await show_system_message (screen ,f"Profile '{text }' already exists.")
            return False 
        self ._name =text 
        endpoints =list (self ._config .endpoints .keys ())
        ep_list =", ".join (endpoints )if endpoints else "(none)"
        await show_system_message (
        screen ,f"Which endpoint? Available: {ep_list }"
        )
        self ._step =2 
        return False 

    async def _handle_endpoint (self ,text :str ,screen :MainScreen )->bool :
        if text not in self ._config .endpoints :
            await show_system_message (screen ,f"Endpoint not found: {text }")
            return False 
        self ._endpoint =text 
        await show_system_message (screen ,"What model?")
        self ._step =3 
        return False 

    async def _handle_model (self ,text :str ,screen :MainScreen )->bool :
        if not text :
            await show_system_message (screen ,"Model name cannot be empty.")
            return False 
        self ._model =text 
        profile =ProfileConfig (
        name =self ._name ,
        endpoint =self ._endpoint ,
        model =self ._model ,
        capabilities =ModelCapabilities (),
        )
        self ._config .profiles [self ._name ]=profile 
        save_config (self ._config )
        await show_system_message (
        screen ,
        f"Profile '{self ._name }' created."
        f" Use `/profile use {self ._name }` to activate.",
        )
        return True 

    async def cancel (self ,screen :MainScreen )->None :
        await show_system_message (screen ,"Profile add cancelled.")
