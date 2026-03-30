"""CLI command group for endpoint management."""

from __future__ import annotations 

from typing import Literal 

import typer 

from llmsh .endpoints import (
add_endpoint as _add_endpoint ,
)
from llmsh .endpoints import (
list_endpoints as _list_endpoints ,
)
from llmsh .endpoints import (
remove_endpoint as _remove_endpoint ,
)
from llmsh .errors import EndpointNotFoundError 
from llmsh .paths import config_path 

endpoint_app =typer .Typer (help ="Manage endpoints")


@endpoint_app .command ("list")
def list_cmd ()->None :
    """List all endpoints."""
    entries =_list_endpoints (config_path ())
    for entry in entries :
        typer .echo (
        f"  {entry ['name']} -- url: {entry ['base_url']},"
        f" auth: {entry ['auth_mode']}"
        )


@endpoint_app .command ("add")
def add_cmd (
name :str =typer .Argument (help ="Endpoint name"),
url :str =typer .Option (help ="Base URL"),
auth_mode :Literal ["api_key","none"]=typer .Option (
"none",help ="Authentication mode",
),
api_key_env :str |None =typer .Option (
None ,help ="Environment variable for API key",
),
)->None :
    """Add a new endpoint."""
    _add_endpoint (
    config_path (),name ,url ,
    auth_mode =auth_mode ,api_key_env =api_key_env ,
    )
    typer .echo (f"Added endpoint: {name }")


@endpoint_app .command ("remove")
def remove_cmd (
name :str =typer .Argument (help ="Endpoint name to remove"),
)->None :
    """Remove an endpoint."""
    try :
        referencing =_remove_endpoint (config_path (),name )
    except EndpointNotFoundError :
        typer .echo (f"Endpoint not found: {name }",err =True )
        raise typer .Exit (code =1 )
    msg =f"Removed endpoint: {name }"
    if referencing :
        msg +=f"\nWarning: profiles still referencing it: {', '.join (referencing )}"
    typer .echo (msg )
