"""Shared profile operations — used by both CLI and slash commands."""

from __future__ import annotations 

from pathlib import Path 
from typing import Any 

from llmsh .config import load_config ,save_config 
from llmsh .errors import ConfigError ,ProfileNotFoundError 
from llmsh .models import ModelCapabilities ,ProfileConfig 


def list_profiles (config_path :Path )->list [dict [str ,Any ]]:
    """Return profile info dicts with name, endpoint, model, active flag."""
    config =load_config (config_path )
    return [
    {
    "name":name ,
    "endpoint":prof .endpoint ,
    "model":prof .model ,
    "active":name ==config .current_profile ,
    }
    for name ,prof in config .profiles .items ()
    ]


def add_profile (
config_path :Path ,name :str ,endpoint :str ,model :str ,
)->None :
    """Add a new profile to the config file."""
    config =load_config (config_path )
    config .profiles [name ]=ProfileConfig (
    name =name ,
    endpoint =endpoint ,
    model =model ,
    capabilities =ModelCapabilities (),
    )
    save_config (config ,config_path )


def use_profile (config_path :Path ,name :str )->None :
    """Switch the current profile."""
    config =load_config (config_path )
    if name not in config .profiles :
        raise ProfileNotFoundError (f"Profile not found: {name }")
    config .current_profile =name 
    save_config (config ,config_path )


def set_model (config_path :Path ,name :str ,model :str )->None :
    """Change the model for a named profile."""
    config =load_config (config_path )
    if name not in config .profiles :
        raise ProfileNotFoundError (f"Profile not found: {name }")
    config .profiles [name ].model =model 
    save_config (config ,config_path )


def remove_profile (config_path :Path ,name :str )->None :
    """Remove a profile from the config file."""
    config =load_config (config_path )
    if name not in config .profiles :
        raise ProfileNotFoundError (f"Profile not found: {name }")
    if name ==config .current_profile :
        raise ConfigError (
        f"Cannot remove active profile '{name }'."
        " Switch to another profile first."
        )
    del config .profiles [name ]
    save_config (config ,config_path )
