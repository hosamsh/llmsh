from __future__ import annotations 

import asyncio 
import json 
from collections .abc import Callable 
from pathlib import Path 
from typing import Optional 

import typer 

from llmsh .config import (
get_active_profile ,
get_endpoint ,
load_config ,
resolve_api_key ,
)
from llmsh .errors import LlmshError ,ProfileNotFoundError 
from llmsh .providers .base import BaseProvider 
from llmsh .providers .openai_compatible import OpenAICompatibleProvider 
from llmsh .summarize import SummarizePlan ,summarize_file 


def _make_provider (endpoint )->OpenAICompatibleProvider :
    api_key =resolve_api_key (endpoint )
    return OpenAICompatibleProvider (endpoint ,api_key =api_key )


def _read_directory (dir_path :Path )->str :
    """Read all text files in a directory recursively."""
    parts :list [str ]=[]
    for file in sorted (dir_path .rglob ("*")):
        if not file .is_file ():
            continue 
        rel =file .relative_to (dir_path )
        if any (p .startswith (".")for p in rel .parts ):
            continue 
        try :
            content =file .read_text ()
        except (UnicodeDecodeError ,OSError ):
            continue 
        parts .append (f"--- FILE: {rel } ---\n{content }")
    if not parts :
        raise LlmshError (f"No readable text files found in: {dir_path }")
    return "\n\n".join (parts )


def _resolve_setup (
file :str ,profile_name :str |None 
)->tuple [BaseProvider ,str ,str ,str ,int ,int ]:
    """Validate file, load config, create provider, determine budget."""
    path =Path (file )
    if path .is_dir ():
        file_text =_read_directory (path )
    elif path .is_file ():
        try :
            file_text =path .read_text ()
        except OSError as exc :
            raise LlmshError (f"Cannot read file: {exc }")
    else :
        raise LlmshError (f"Not found: {file }")

    config =load_config ()
    if profile_name is not None :
        prof =config .profiles .get (profile_name )
        if prof is None :
            raise ProfileNotFoundError (
            f"Profile not found: {profile_name }"
            )
    else :
        prof =get_active_profile (config )

    endpoint =get_endpoint (config ,prof )
    provider =_make_provider (endpoint )
    model =prof .model 


    context_length :int 
    try :
        model_info =asyncio .run (provider .get_model_info (model ))
        context_length =model_info .context_length or 4096 
    except Exception :
        context_length =4096 
        typer .echo (
        "Warning: could not fetch model info, "
        f"using default context length {context_length }",
        err =True ,
        )

    max_output_tokens =prof .max_tokens or 1024 

    return provider ,model ,prof .name ,file_text ,context_length ,max_output_tokens 


async def _run_summarize (
provider :BaseProvider ,
model :str ,
instruction :str ,
file_text :str ,
context_length :int ,
max_output_tokens :int ,
on_plan :Callable [[SummarizePlan ],None ]|None =None ,
on_progress :Callable [[str ,int ,int ],None ]|None =None ,
)->tuple [str ,int ,int ]:
    return await summarize_file (
    provider =provider ,
    model =model ,
    instruction =instruction ,
    file_text =file_text ,
    context_length =context_length ,
    max_output_tokens =max_output_tokens ,
    on_plan =on_plan ,
    on_progress =on_progress ,
    )


def summarize (
file :str =typer .Argument (...,help ="Path to the file to analyze"),
instruction :str =typer .Argument (...,help ="What to extract or analyze"),
profile :Optional [str ]=typer .Option (None ,"--profile",help ="Profile to use"),
json_output :bool =typer .Option (False ,"--json",help ="Output as JSON"),
)->None :
    try :
        provider ,model ,profile_name ,file_text ,context_length ,max_output_tokens =(
        _resolve_setup (file ,profile )
        )
    except LlmshError as exc :
        typer .echo (str (exc ),err =True )
        raise typer .Exit (code =1 )

    def on_plan (plan :SummarizePlan )->None :
        typer .echo (
        f"Plan: {plan .total_chunks } chunks, "
        f"~{plan .estimated_reduce_rounds } reduce rounds, "
        f"~{plan .estimated_total_calls } total LLM calls",
        err =True ,
        )

    def on_progress (phase :str ,current :int ,total :int )->None :
        if phase =="map":
            typer .echo (f"Map: chunk {current }/{total }...",err =True )
        elif phase .startswith ("reduce-"):
            round_num =phase .split ("-",1 )[1 ]
            typer .echo (
            f"Reduce round {round_num }: batch {current }/{total }...",
            err =True ,
            )

    try :
        answer ,chunks_processed ,truncated_calls =asyncio .run (
        _run_summarize (
        provider =provider ,
        model =model ,
        instruction =instruction ,
        file_text =file_text ,
        context_length =context_length ,
        max_output_tokens =max_output_tokens ,
        on_plan =on_plan ,
        on_progress =on_progress ,
        )
        )
    except Exception as exc :
        typer .echo (str (exc ),err =True )
        raise typer .Exit (code =1 )

    if truncated_calls >0 :
        typer .echo (
        f"Warning: {truncated_calls } response(s) were truncated"
        " — results may be incomplete",
        err =True ,
        )

    if json_output :
        output ={
        "answer":answer ,
        "model":model ,
        "profile":profile_name ,
        "file":file ,
        "chunks_processed":chunks_processed ,
        "truncated_calls":truncated_calls ,
        }
        typer .echo (json .dumps (output ))
    else :
        typer .echo (answer )
