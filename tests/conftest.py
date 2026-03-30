"""Shared test fixtures for llmsh."""

from typing import AsyncIterator 

from llmsh .models import ModelInfo 
from llmsh .providers .base import (
BaseProvider ,
ProviderEvent ,
ProviderRequest ,
ProviderResult ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
)


class StubProvider (BaseProvider ):
    """A controllable provider for tests.

    Pass a list of events to ``__init__``; ``stream_chat`` yields them in order.
    ``chat`` returns the last ``ResponseCompleted`` content (or empty string).
    """

    def __init__ (self ,events :list [ProviderEvent ]|None =None )->None :
        self .events :list [ProviderEvent ]=events or [
        ResponseStarted (),
        TextDelta (text ="hello "),
        TextDelta (text ="world"),
        ResponseCompleted (content ="hello world"),
        ]
        self .requests :list [ProviderRequest ]=[]

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        self .requests .append (request )
        for event in self .events :
            yield event 

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        self .requests .append (request )
        content =""
        for event in self .events :
            if isinstance (event ,ResponseCompleted ):
                content =event .content 
        return ProviderResult (content =content )


class BudgetStubProvider (StubProvider ):
    """StubProvider that also supports get_model_info and error injection."""

    def __init__ (
    self ,
    events :list [ProviderEvent ]|None =None ,
    model_info :ModelInfo |None =None ,
    model_info_error :Exception |None =None ,
    error_on_stream :Exception |None =None ,
    )->None :
        super ().__init__ (events =events )
        self ._model_info =model_info 
        self ._model_info_error =model_info_error 
        self ._error_on_stream =error_on_stream 
        self .get_model_info_calls :list [str ]=[]

    async def get_model_info (self ,model_id :str )->ModelInfo :
        self .get_model_info_calls .append (model_id )
        if self ._model_info_error is not None :
            raise self ._model_info_error 
        if self ._model_info is not None :
            return self ._model_info 
        raise NotImplementedError 

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        if self ._error_on_stream is not None :
            raise self ._error_on_stream 
        async for event in super ().stream_chat (request ):
            yield event 
