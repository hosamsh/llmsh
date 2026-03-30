"""Shared endpoint operations — used by both CLI and slash commands."""

from __future__ import annotations 

from pathlib import Path 
from typing import Any ,Literal 

from llmsh .config import load_config ,save_config 
from llmsh .errors import EndpointNotFoundError 
from llmsh .models import EndpointConfig 


def list_endpoints (config_path :Path )->list [dict [str ,Any ]]:
    """Return endpoint info dicts with name, base_url, auth_mode."""
    config =load_config (config_path )
    return [
    {
    "name":name ,
    "base_url":ep .base_url ,
    "auth_mode":ep .auth_mode ,
    }
    for name ,ep in config .endpoints .items ()
    ]


def add_endpoint (
config_path :Path ,
name :str ,
url :str ,
auth_mode :Literal ["api_key","none"]="none",
api_key_env :str |None =None ,
)->None :
    """Add a new endpoint to the config file."""
    config =load_config (config_path )
    config .endpoints [name ]=EndpointConfig (
    name =name ,
    base_url =url ,
    auth_mode =auth_mode ,
    api_key_env =api_key_env ,
    )
    save_config (config ,config_path )


def remove_endpoint (config_path :Path ,name :str )->list [str ]:
    """Remove an endpoint. Returns list of profile names still referencing it."""
    config =load_config (config_path )
    if name not in config .endpoints :
        raise EndpointNotFoundError (f"Endpoint not found: {name }")
    referencing =[p .name for p in config .profiles .values ()if p .endpoint ==name ]
    del config .endpoints [name ]
    save_config (config ,config_path )
    return referencing 
