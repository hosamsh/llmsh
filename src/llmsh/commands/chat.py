from __future__ import annotations 

from typing import Optional 

import typer 

from llmsh .app import AppCore 
from llmsh .errors import ConfigError ,LlmshError 
from llmsh .ui .main import LlmshApp 


def chat (
profile :Optional [str ]=typer .Option (None ,"--profile",help ="Profile to use"),
)->None :
    core =None 
    try :
        core =AppCore ()
        if profile is not None :
            core .switch_profile (profile )
    except ConfigError :
        pass 
    except LlmshError as exc :
        typer .echo (str (exc ),err =True )
        raise typer .Exit (code =1 )

    app =LlmshApp (core )
    app .run ()
