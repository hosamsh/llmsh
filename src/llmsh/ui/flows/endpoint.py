from __future__ import annotations 

import re 
from typing import TYPE_CHECKING ,Literal 

from llmsh .config import AppConfig ,save_config 
from llmsh .models import EndpointConfig 
from llmsh .ui .flow import InteractiveFlow 
from llmsh .ui .slash import show_system_message 

if TYPE_CHECKING :
    from llmsh .ui .screens import MainScreen 


class EndpointAddFlow (InteractiveFlow ):
    def __init__ (self ,config :AppConfig )->None :
        self ._config =config 
        self ._step =0 
        self ._url =""
        self ._auth_mode :Literal ["api_key","none"]="none"
        self ._api_key_env :str |None =None 

    async def start (self ,screen :MainScreen )->None :
        await show_system_message (screen ,"What's the base URL?")
        self ._step =1 

    @property 
    def placeholder (self )->str :
        placeholders ={
        1 :"e.g., http://localhost:8006/v1",
        2 :"api_key or none",
        3 :"e.g., OPENAI_API_KEY",
        4 :"e.g., my-endpoint",
        }
        return placeholders .get (self ._step ,"Type your response...")

    async def handle_input (self ,text :str ,screen :MainScreen )->bool :
        text =text .strip ()
        if self ._step ==1 :
            return await self ._handle_url (text ,screen )
        elif self ._step ==2 :
            return await self ._handle_auth (text ,screen )
        elif self ._step ==3 :
            return await self ._handle_api_key_env (text ,screen )
        elif self ._step ==4 :
            return await self ._handle_name (text ,screen )
        return True 

    async def _handle_url (self ,text :str ,screen :MainScreen )->bool :
        if not text .startswith (("http://","https://")):
            await show_system_message (screen ,"URL must start with http:// or https://")
            return False 
        self ._url =text .rstrip ("/")
        await show_system_message (screen ,"Auth mode? (api_key / none)")
        self ._step =2 
        return False 

    async def _handle_auth (self ,text :str ,screen :MainScreen )->bool :
        if text not in ("api_key","none"):
            await show_system_message (screen ,"Please enter api_key or none.")
            return False 
        if text =="api_key":
            self ._auth_mode ="api_key"
            await show_system_message (screen ,"Environment variable for the API key?")
            self ._step =3 
        else :
            self ._auth_mode ="none"
            await show_system_message (screen ,"Name for this endpoint?")
            self ._step =4 
        return False 

    async def _handle_api_key_env (self ,text :str ,screen :MainScreen )->bool :
        if not text or " "in text :
            await show_system_message (
            screen ,"Environment variable must be non-empty with no spaces."
            )
            return False 
        self ._api_key_env =text 
        await show_system_message (screen ,"Name for this endpoint?")
        self ._step =4 
        return False 

    async def _handle_name (self ,text :str ,screen :MainScreen )->bool :
        if not re .match (r"^[a-zA-Z0-9-]+$",text ):
            await show_system_message (
            screen ,"Name must be alphanumeric (hyphens allowed)."
            )
            return False 
        if text in self ._config .endpoints :
            await show_system_message (screen ,f"Endpoint '{text }' already exists.")
            return False 
        endpoint =EndpointConfig (
        name =text ,
        base_url =self ._url ,
        auth_mode =self ._auth_mode ,
        api_key_env =self ._api_key_env ,
        )
        self ._config .endpoints [text ]=endpoint 
        save_config (self ._config )
        await show_system_message (screen ,f"Endpoint '{text }' added.")
        return True 

    async def cancel (self ,screen :MainScreen )->None :
        await show_system_message (screen ,"Endpoint add cancelled.")
