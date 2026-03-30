from __future__ import annotations 

import asyncio 
import json 
import sys 
from typing import Optional 

import typer 

from llmsh .app import AppCore 
from llmsh .errors import LlmshError 
from llmsh .models import UsageInfo 
from llmsh .providers .base import ErrorEvent ,TextDelta ,TokenUsageEvent 


def _make_app_core (
profile :str |None =None ,
system_prompt :str |None =None ,
)->AppCore :
    core =AppCore (system_prompt =system_prompt )
    if profile is not None :
        core .switch_profile (profile )
    return core 


def ask (
prompt :Optional [str ]=typer .Argument (None ,help ="Prompt to send"),
stdin :bool =typer .Option (False ,"--stdin",help ="Read prompt from stdin"),
json_output :bool =typer .Option (False ,"--json",help ="Output as JSON"),
profile :Optional [str ]=typer .Option (None ,"--profile",help ="Profile to use"),
system_prompt :Optional [str ]=typer .Option (None ,"--system",help ="System prompt"),
)->None :
    if stdin :
        text =sys .stdin .read ().strip ()
    elif prompt is not None :
        text =prompt 
    else :
        typer .echo ("Error: provide a prompt or use --stdin",err =True )
        raise typer .Exit (code =1 )

    try :
        core =_make_app_core (profile =profile ,system_prompt =system_prompt )
    except LlmshError as exc :
        typer .echo (str (exc ),err =True )
        raise typer .Exit (code =1 )

    response_text ,usage =asyncio .run (_stream (core ,text ))

    if json_output :
        output ={
        "response":response_text ,
        "model":core .model ,
        "profile":core .profile .name ,
        "usage":usage .model_dump ()if usage else None ,
        }
        typer .echo (json .dumps (output ))
    else :
        typer .echo (response_text )


async def _stream (core :AppCore ,prompt :str )->tuple [str ,UsageInfo |None ]:
    chunks :list [str ]=[]
    usage :UsageInfo |None =None 
    async for event in core .send_message (prompt ):
        if isinstance (event ,ErrorEvent ):
            typer .echo (event .message ,err =True )
            raise typer .Exit (code =1 )
        elif isinstance (event ,TextDelta ):
            chunks .append (event .text )
        elif isinstance (event ,TokenUsageEvent ):
            usage =event .usage 
    return "".join (chunks ),usage 
