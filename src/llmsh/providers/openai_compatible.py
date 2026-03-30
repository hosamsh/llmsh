from __future__ import annotations 

import asyncio 
import json as _json 
from typing import AsyncIterator 

import httpx 

from llmsh .errors import ModelNotFoundError ,ProviderError 
from llmsh .models import EndpointConfig ,ModelInfo ,ToolCall ,ToolDefinition ,UsageInfo 
from llmsh .providers .base import (
BaseProvider ,
CancelledEvent ,
DoctorCheck ,
DoctorReport ,
ErrorEvent ,
ProviderEvent ,
ProviderRequest ,
ProviderResult ,
ReasoningDelta ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
TokenUsageEvent ,
ToolCallEvent ,
)


def _parse_error_body (body :str ,status_code :int )->str :
    """Extract human-readable message from JSON error response."""
    try :
        data =_json .loads (body )
        if isinstance (data ,dict )and "error"in data :
            err =data ["error"]
            if isinstance (err ,dict )and "message"in err :
                return err ["message"]
            if isinstance (err ,str ):
                return err 
    except (ValueError ,KeyError ,TypeError ):
        pass 
    if len (body )>200 :
        return f"HTTP {status_code } error"
    return f"HTTP {status_code }: {body }"if body else f"HTTP {status_code } error"


def _classify_http_error (status_code :int ,body :str )->str :
    """Classify HTTP error into a type string."""
    if status_code in (401 ,403 ):
        return "auth"
    body_lower =body .lower ()
    if status_code ==400 and any (
    kw in body_lower for kw in ("context","exceeds","too long","token")
    ):
        return "context_overflow"
    if status_code ==404 and (
    "model"in body_lower or "not_found"in body_lower or "not found"in body_lower 
    ):
        return "model_not_found"
    if status_code ==404 :
        return "not_found"
    if status_code >=500 :
        return "server_error"
    return "unknown"


class OpenAICompatibleProvider (BaseProvider ):
    def __init__ (self ,endpoint :EndpointConfig ,api_key :str |None =None )->None :
        self ._endpoint =endpoint 
        self ._api_key =api_key 

    def _build_headers (self )->dict [str ,str ]:
        headers :dict [str ,str ]={}
        if self ._endpoint .auth_mode =="api_key"and self ._api_key :
            headers ["Authorization"]=f"Bearer {self ._api_key }"
        return headers 

    def _build_client (self )->httpx .AsyncClient :
        return httpx .AsyncClient (
        verify =self ._endpoint .verify_ssl ,
        timeout =self ._endpoint .timeout_seconds ,
        headers =self ._build_headers (),
        )

    def _build_body (self ,request :ProviderRequest ,*,stream :bool )->dict :
        body :dict ={
        "model":request .model ,
        "messages":[
        {"role":m .role ,"content":m .content }for m in request .messages 
        ],
        "stream":stream ,
        }
        if request .temperature is not None :
            body ["temperature"]=request .temperature 
        if request .max_tokens is not None :
            body ["max_tokens"]=request .max_tokens 
        if stream :
            body ["stream_options"]={"include_usage":True }
        if request .tools :
            body ["tools"]=[self ._tool_to_openai (t )for t in request .tools ]
        return body 

    @staticmethod 
    def _tool_to_openai (tool :ToolDefinition )->dict :
        properties ={}
        required =[]
        for p in tool .parameters :
            properties [p .name ]={"type":p .type ,"description":p .description }
            if p .required :
                required .append (p .name )
        return {
        "type":"function",
        "function":{
        "name":tool .name ,
        "description":tool .description ,
        "parameters":{
        "type":"object",
        "properties":properties ,
        "required":required ,
        },
        },
        }

    def _parse_usage (self ,raw :dict )->UsageInfo :
        return UsageInfo (
        input_tokens =raw .get ("prompt_tokens"),
        output_tokens =raw .get ("completion_tokens"),
        total_tokens =raw .get ("total_tokens"),
        )

    def _parse_sse_line (
    self ,line :str 
    )->tuple [str ,str ,UsageInfo |None ,dict |None ]|None :
        """Parse one SSE line.

        Returns (text, reasoning, usage, tool_call_chunk) or None.
        """
        if not line .startswith ("data: "):
            return None 
        payload =line [len ("data: "):]
        if payload =="[DONE]":
            return None 
        chunk =_json .loads (payload )
        choices =chunk .get ("choices",[])
        if choices :
            delta =choices [0 ].get ("delta",{})
            text =delta .get ("content")or ""
            reasoning =delta .get ("reasoning_content")or ""
            tool_calls =delta .get ("tool_calls")
            tc_chunk =tool_calls [0 ]if tool_calls else None 
        else :
            text =""
            reasoning =""
            tc_chunk =None 
        usage =self ._parse_usage (chunk ["usage"])if "usage"in chunk else None 
        return text ,reasoning ,usage ,tc_chunk 

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        body =self ._build_body (request ,stream =False )
        url =f"{self ._endpoint .base_url }/chat/completions"
        async with self ._build_client ()as client :
            response =await client .post (url ,json =body )
            if response .status_code >=400 :
                msg =_parse_error_body (response .text ,response .status_code )
                error_type =_classify_http_error (response .status_code ,response .text )
                raise ProviderError (msg ,error_type =error_type )
            data =response .json ()
        choice =data ["choices"][0 ]
        content =choice ["message"]["content"]
        finish_reason =choice .get ("finish_reason","")
        truncated =finish_reason =="length"
        usage =self ._parse_usage (data ["usage"])if "usage"in data else None 
        return ProviderResult (content =content ,usage =usage ,truncated =truncated )

    async def _check_stream_error (
    self ,response :httpx .Response ,
    )->ErrorEvent |None :
        if response .status_code in (401 ,403 ):
            return ErrorEvent (
            message =f"Authentication failed (HTTP {response .status_code })",
            error_type ="auth",
            )
        if response .status_code >=400 :
            await response .aread ()
            msg =_parse_error_body (response .text ,response .status_code )
            error_type =_classify_http_error (response .status_code ,response .text )
            return ErrorEvent (message =msg ,error_type =error_type )
        return None 

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        body =self ._build_body (request ,stream =True )
        url =f"{self ._endpoint .base_url }/chat/completions"

        async with self ._build_client ()as client :
            async with client .stream ("POST",url ,json =body )as response :
                error =await self ._check_stream_error (response )
                if error is not None :
                    yield error 
                    return 

                yield ResponseStarted ()
                accumulated =""
                usage :UsageInfo |None =None 
                tc_id :str |None =None 
                tc_name :str |None =None 
                tc_args =""

                try :
                    async for line in response .aiter_lines ():
                        parsed =self ._parse_sse_line (line )
                        if parsed is None :
                            continue 
                        text ,reasoning ,line_usage ,tc_chunk =parsed 
                        if reasoning :
                            yield ReasoningDelta (text =reasoning )
                        if text :
                            accumulated +=text 
                            yield TextDelta (text =text )
                        if line_usage is not None :
                            usage =line_usage 
                        if tc_chunk :
                            if "id"in tc_chunk :
                                tc_id =tc_chunk ["id"]
                            func =tc_chunk .get ("function",{})
                            if "name"in func :
                                tc_name =func ["name"]
                            tc_args +=func .get ("arguments","")
                except asyncio .CancelledError :
                    yield CancelledEvent ()
                    return 

                if tc_id and tc_name and tc_args :
                    yield ToolCallEvent (
                    tool_call =ToolCall (
                    id =tc_id ,
                    name =tc_name ,
                    arguments =_json .loads (tc_args ),
                    )
                    )
                if usage :
                    yield TokenUsageEvent (usage =usage )
                yield ResponseCompleted (content =accumulated )

    async def list_models (self )->list [str ]:
        url =f"{self ._endpoint .base_url }/models"
        async with self ._build_client ()as client :
            response =await client .get (url )
            response .raise_for_status ()
            data =response .json ()
        return [m ["id"]for m in data ["data"]]

    async def get_model_info (self ,model_id :str )->ModelInfo :
        url =f"{self ._endpoint .base_url }/models"
        async with self ._build_client ()as client :
            response =await client .get (url )
            response .raise_for_status ()
            data =response .json ()
        for m in data ["data"]:
            if m ["id"]==model_id :
                return ModelInfo (id =m ["id"],context_length =m .get ("context_length"))
        raise ModelNotFoundError (f"Model '{model_id }' not found")

    async def doctor (self )->DoctorReport :
        base =self ._endpoint .base_url 
        if base .endswith ("/v1"):
            health_url =base [:-len ("/v1")]+"/health"
        else :
            health_url =base .rstrip ("/")+"/health"

        try :
            async with self ._build_client ()as client :
                response =await client .get (health_url )
            passed =response .status_code ==200 
            msg =(
            f"Endpoint reachable at {health_url }"
            if passed 
            else f"HTTP {response .status_code } from {health_url }"
            )
        except httpx .HTTPError as exc :
            passed ,msg =False ,f"Connection failed: {exc }"

        check =DoctorCheck (name ="reachability",passed =passed ,message =msg )
        return DoctorReport (checks =[check ])
