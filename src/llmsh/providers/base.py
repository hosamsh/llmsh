from __future__ import annotations 

from dataclasses import dataclass 
from typing import AsyncIterator ,Union 

from pydantic import BaseModel 

from llmsh .models import ChatMessage ,ModelInfo ,ToolCall ,ToolDefinition ,UsageInfo 


@dataclass 
class ResponseStarted :
    pass 


@dataclass 
class TextDelta :
    text :str 


@dataclass 
class ReasoningDelta :
    text :str 


@dataclass 
class ResponseCompleted :
    content :str 


@dataclass 
class TokenUsageEvent :
    usage :UsageInfo 


@dataclass 
class ErrorEvent :
    message :str 
    error_type :str ="unknown"


@dataclass 
class ToolCallEvent :
    tool_call :ToolCall 


@dataclass 
class CancelledEvent :
    pass 


@dataclass 
class BudgetWarningEvent :
    utilization :float 
    remaining_tokens :int 
    message :str 


@dataclass 
class CompactionEvent :
    messages_dropped :int 
    message :str 


ProviderEvent =Union [
ResponseStarted ,
TextDelta ,
ReasoningDelta ,
ResponseCompleted ,
TokenUsageEvent ,
ToolCallEvent ,
ErrorEvent ,
CancelledEvent ,
BudgetWarningEvent ,
CompactionEvent ,
]


class ProviderRequest (BaseModel ):
    messages :list [ChatMessage ]
    model :str 
    temperature :float |None =None 
    max_tokens :int |None =None 
    tools :list [ToolDefinition ]|None =None 


class ProviderResult (BaseModel ):
    content :str 
    usage :UsageInfo |None =None 
    truncated :bool =False 


class DoctorCheck (BaseModel ):
    name :str 
    passed :bool 
    message :str 


class DoctorReport (BaseModel ):
    checks :list [DoctorCheck ]


class BaseProvider :
    async def chat (self ,request :ProviderRequest )->ProviderResult :
        raise NotImplementedError 

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        raise NotImplementedError 
        yield 

    async def list_models (self )->list [str ]:
        raise NotImplementedError 

    async def get_model_info (self ,model_id :str )->ModelInfo :
        raise NotImplementedError 

    async def doctor (self )->DoctorReport :
        raise NotImplementedError 
