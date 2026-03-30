"""CLI command group for profile management."""

from __future__ import annotations 

import typer 

from llmsh .errors import ConfigError ,ProfileNotFoundError 
from llmsh .paths import config_path 
from llmsh .profiles import (
add_profile as _add_profile ,
)
from llmsh .profiles import (
list_profiles as _list_profiles ,
)
from llmsh .profiles import (
remove_profile as _remove_profile ,
)
from llmsh .profiles import (
set_model as _set_model ,
)
from llmsh .profiles import (
use_profile as _use_profile ,
)

profile_app =typer .Typer (help ="Manage profiles")


@profile_app .command ("list")
def list_cmd ()->None :
    """List all profiles."""
    entries =_list_profiles (config_path ())
    for entry in entries :
        active =" (active)"if entry ["active"]else ""
        typer .echo (
        f"  {entry ['name']} -- endpoint: {entry ['endpoint']},"
        f" model: {entry ['model']}{active }"
        )


@profile_app .command ("add")
def add_cmd (
name :str =typer .Argument (help ="Profile name"),
endpoint :str =typer .Option (help ="Endpoint name"),
model :str =typer .Option (help ="Model identifier"),
)->None :
    """Add a new profile."""
    _add_profile (config_path (),name ,endpoint ,model )
    typer .echo (f"Added profile: {name }")


@profile_app .command ("use")
def use_cmd (
name :str =typer .Argument (help ="Profile name to switch to"),
)->None :
    """Switch the active profile."""
    try :
        _use_profile (config_path (),name )
    except ProfileNotFoundError :
        typer .echo (f"Profile not found: {name }",err =True )
        raise typer .Exit (code =1 )
    typer .echo (f"Switched to profile: {name }")


@profile_app .command ("set-model")
def set_model_cmd (
name :str =typer .Argument (help ="Profile name"),
model :str =typer .Argument (help ="New model identifier"),
)->None :
    """Change the model for a profile."""
    try :
        _set_model (config_path (),name ,model )
    except ProfileNotFoundError :
        typer .echo (f"Profile not found: {name }",err =True )
        raise typer .Exit (code =1 )
    typer .echo (f"Model for '{name }' set to: {model }")


@profile_app .command ("remove")
def remove_cmd (
name :str =typer .Argument (help ="Profile name to remove"),
)->None :
    """Remove a profile."""
    try :
        _remove_profile (config_path (),name )
    except ProfileNotFoundError :
        typer .echo (f"Profile not found: {name }",err =True )
        raise typer .Exit (code =1 )
    except ConfigError as exc :
        typer .echo (str (exc ),err =True )
        raise typer .Exit (code =1 )
    typer .echo (f"Removed profile: {name }")
