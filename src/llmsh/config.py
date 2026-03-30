import os 
import tomllib 
from pathlib import Path 
from typing import Any 

import tomli_w 
from pydantic import BaseModel ,ValidationError 

from llmsh .errors import ConfigError ,ProfileNotFoundError 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
from llmsh .paths import config_path 


class AppConfig (BaseModel ):
    current_profile :str 
    endpoints :dict [str ,EndpointConfig ]
    profiles :dict [str ,ProfileConfig ]


def load_config (path :Path |None =None )->AppConfig :
    target =path or config_path ()
    try :
        with open (target ,"rb")as f :
            data =tomllib .load (f )
    except FileNotFoundError :
        raise ConfigError (
        f"Config file not found: {target }."
        " Run 'llmsh endpoint add' to set up your config."
        )

    try :
        endpoints ={
        name :EndpointConfig (name =name ,**fields )
        for name ,fields in data .get ("endpoints",{}).items ()
        }
        profiles :dict [str ,ProfileConfig ]={}
        for name ,fields in data .get ("profiles",{}).items ():
            caps_data =fields .pop ("capabilities",{})
            profiles [name ]=ProfileConfig (
            name =name ,capabilities =ModelCapabilities (**caps_data ),**fields 
            )
        return AppConfig (
        current_profile =data ["current_profile"],
        endpoints =endpoints ,
        profiles =profiles ,
        )
    except (ValidationError ,KeyError )as exc :
        raise ConfigError (
        f"Invalid config at {target }: {exc }."
        " Run 'llmsh endpoint add' to set up your config."
        )from exc 


def save_config (config :AppConfig ,path :Path |None =None )->None :
    target =path or config_path ()
    target .parent .mkdir (parents =True ,exist_ok =True )
    data :dict [str ,Any ]={"current_profile":config .current_profile }
    if config .endpoints :
        data ["endpoints"]={
        name :{k :v for k ,v in ep .model_dump (exclude ={"name"}).items ()
        if v is not None }
        for name ,ep in config .endpoints .items ()
        }
    if config .profiles :
        profiles_data :dict [str ,Any ]={}
        for name ,prof in config .profiles .items ():
            entry :dict [str ,Any ]={
            "endpoint":prof .endpoint ,
            "model":prof .model ,
            "capabilities":prof .capabilities .model_dump (),
            }
            if prof .max_tokens is not None :
                entry ["max_tokens"]=prof .max_tokens 
            profiles_data [name ]=entry 
        data ["profiles"]=profiles_data 
    target .write_bytes (tomli_w .dumps (data ).encode ())


def get_active_profile (config :AppConfig )->ProfileConfig :
    profile =config .profiles .get (config .current_profile )
    if profile is None :
        raise ProfileNotFoundError (f"Profile not found: {config .current_profile }")
    return profile 


def get_endpoint (config :AppConfig ,profile :ProfileConfig )->EndpointConfig :
    endpoint =config .endpoints .get (profile .endpoint )
    if endpoint is None :
        raise ConfigError (f"Endpoint not found: {profile .endpoint }")
    return endpoint 


def resolve_api_key (endpoint :EndpointConfig )->str |None :
    if endpoint .api_key_env is None :
        return None 
    return os .environ .get (endpoint .api_key_env )
