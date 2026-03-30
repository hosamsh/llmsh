from __future__ import annotations 

import re 
from typing import TYPE_CHECKING 
from urllib .parse import urlparse 

from llmsh .config import AppConfig ,save_config 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
from llmsh .ui .flow import InteractiveFlow 
from llmsh .ui .slash import show_system_message 

if TYPE_CHECKING :
    from llmsh .ui .screens import MainScreen 


class SetupFlow (InteractiveFlow ):
    def __init__ (self )->None :
        self ._step =0 
        self ._url =""
        self ._auth_mode ="none"
        self ._api_key_env :str |None =None 
        self ._model =""
        self ._name =""

    async def start (self ,screen :MainScreen )->None :
        await show_system_message (
        screen ,
        "Welcome to llmsh! No configuration found. Let's set one up.",
        )
        await show_system_message (
        screen ,
        "What's the base URL of your API endpoint?",
        )
        self ._step =1 

    @property 
    def placeholder (self )->str :
        placeholders ={
        1 :"e.g., http://localhost:8006/v1",
        2 :"y or n",
        3 :"e.g., OPENAI_API_KEY",
        4 :"e.g., gpt-4, llama-3",
        5 :"e.g., local, openai",
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
            return await self ._handle_model (text ,screen )
        elif self ._step ==5 :
            return await self ._handle_name (text ,screen )
        return True 

    async def _handle_url (self ,text :str ,screen :MainScreen )->bool :
        if not text .startswith (("http://","https://")):
            msg ="URL must start with http:// or https://"
            await show_system_message (screen ,msg )
            return False 
        self ._url =text .rstrip ("/")
        msg ="Does this endpoint require an API key? (y/n)"
        await show_system_message (screen ,msg )
        self ._step =2 
        return False 

    async def _handle_auth (self ,text :str ,screen :MainScreen )->bool :
        if text .lower ()in ("y","yes"):
            self ._auth_mode ="api_key"
            msg ="What environment variable holds the API key?"
            await show_system_message (screen ,msg )
            self ._step =3 
        elif text .lower ()in ("n","no"):
            self ._auth_mode ="none"
            await show_system_message (screen ,"What model do you want to use?")
            self ._step =4 
        else :
            await show_system_message (screen ,"Please enter y or n.")
        return False 

    async def _handle_api_key_env (self ,text :str ,screen :MainScreen )->bool :
        if not text or " "in text :
            msg ="Environment variable must be non-empty with no spaces."
            await show_system_message (screen ,msg )
            return False 
        self ._api_key_env =text 
        await show_system_message (screen ,"What model do you want to use?")
        self ._step =4 
        return False 

    async def _handle_model (self ,text :str ,screen :MainScreen )->bool :
        if not text :
            await show_system_message (screen ,"Model name cannot be empty.")
            return False 
        self ._model =text 
        default =self ._derive_name ()
        msg =f"What name for this endpoint? (default: {default })"
        await show_system_message (screen ,msg )
        self ._step =5 
        return False 

    async def _handle_name (self ,text :str ,screen :MainScreen )->bool :
        name =text or self ._derive_name ()
        if not re .match (r"^[a-zA-Z0-9-]+$",name ):
            msg ="Name must be alphanumeric (hyphens allowed)."
            await show_system_message (screen ,msg )
            return False 
        self ._name =name 
        await self ._save_and_init (screen )
        return True 

    def _derive_name (self )->str :
        hostname =urlparse (self ._url ).hostname or "default"
        return re .sub (r"[^a-zA-Z0-9-]","-",hostname ).strip ("-")or "default"

    async def _save_and_init (self ,screen :MainScreen )->None :
        endpoint =EndpointConfig (
        name =self ._name ,
        base_url =self ._url ,
        auth_mode =self ._auth_mode ,
        api_key_env =self ._api_key_env ,
        )
        profile =ProfileConfig (
        name =self ._name ,
        endpoint =self ._name ,
        model =self ._model ,
        capabilities =ModelCapabilities (),
        )
        config =AppConfig (
        current_profile =self ._name ,
        endpoints ={self ._name :endpoint },
        profiles ={self ._name :profile },
        )
        save_config (config )

        from llmsh .app import AppCore 
        from llmsh .ui .screens import HeaderBar 

        screen .app .core =AppCore (config =config )
        screen .query_one (HeaderBar ).refresh_display (
        profile =self ._name ,model =self ._model 
        )

        msg =(
        f"Configuration saved! Profile '{self ._name }' is active."
        " You can start chatting now."
        )
        await show_system_message (screen ,msg )

    async def cancel (self ,screen :MainScreen )->None :
        await show_system_message (screen ,"Setup cancelled. Run llmsh again to retry.")
