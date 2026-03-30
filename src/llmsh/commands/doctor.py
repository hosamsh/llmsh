from __future__ import annotations 

import asyncio 
from typing import Optional 

import typer 

from llmsh .app import AppCore 
from llmsh .config import get_endpoint 
from llmsh .doctor import run_doctor 
from llmsh .errors import LlmshError 


def _make_core (profile :str |None =None )->AppCore :
    core =AppCore ()
    if profile is not None :
        core .switch_profile (profile )
    return core 


def doctor (
profile :Optional [str ]=typer .Option (None ,"--profile",help ="Profile to use"),
)->None :
    try :
        core =_make_core (profile =profile )
    except LlmshError as exc :
        typer .echo (str (exc ),err =True )
        raise typer .Exit (code =1 )

    endpoint =get_endpoint (core ._config ,core .profile )
    report =asyncio .run (run_doctor (core .profile ,endpoint ,core ._provider ))

    has_failure =False 
    for check in report .checks :
        status ="PASS"if check .passed else "FAIL"
        if not check .passed :
            has_failure =True 
        typer .echo (f"  [{status }] {check .name }: {check .message }")

    if has_failure :
        raise typer .Exit (code =1 )
