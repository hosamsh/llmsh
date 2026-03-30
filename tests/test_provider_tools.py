"""Tests for provider tool-calling support."""

import json 

import httpx 
import pytest 
import respx 

from llmsh .models import ChatMessage ,EndpointConfig ,ToolDefinition ,ToolParameter 
from llmsh .providers .base import (
ErrorEvent ,
ProviderRequest ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
ToolCallEvent ,
)
from llmsh .providers .openai_compatible import OpenAICompatibleProvider 





API_KEY ="test-key-123"
BASE_URL ="http://localhost:8006/v1"


def _endpoint ()->EndpointConfig :
    return EndpointConfig (
    name ="test",
    base_url =BASE_URL ,
    auth_mode ="api_key",
    provider_type ="openai_compatible",
    )


def _tool_defs ()->list [ToolDefinition ]:
    return [
    ToolDefinition (
    name ="get_weather",
    description ="Get the weather for a city",
    parameters =[
    ToolParameter (
    name ="city",
    type ="string",
    description ="The city name",
    required =True ,
    ),
    ToolParameter (
    name ="units",
    type ="string",
    description ="Temperature units",
    required =False ,
    ),
    ],
    )
    ]


def _request (tools :list [ToolDefinition ]|None =None )->ProviderRequest :
    return ProviderRequest (
    messages =[ChatMessage (role ="user",content ="hello")],
    model ="test-model",
    tools =tools ,
    )


def _sse_lines (*chunks :str ,usage :dict |None =None )->str :
    """Build raw SSE text from delta content chunks."""
    lines =[]
    for chunk in chunks :
        data ={"choices":[{"delta":{"content":chunk }}]}
        lines .append (f"data: {json .dumps (data )}")
        lines .append ("")
    if usage :
        data ={"choices":[{"delta":{}}],"usage":usage }
        lines .append (f"data: {json .dumps (data )}")
        lines .append ("")
    lines .append ("data: [DONE]")
    lines .append ("")
    return "\n".join (lines )


def _sse_tool_call_lines (
tool_id :str ,name :str ,arguments :dict 
)->str :
    """Build raw SSE text simulating a streamed tool call."""
    arg_str =json .dumps (arguments )
    lines =[]


    data ={
    "choices":[
    {
    "delta":{
    "tool_calls":[
    {
    "index":0 ,
    "id":tool_id ,
    "type":"function",
    "function":{"name":name ,"arguments":""},
    }
    ]
    }
    }
    ]
    }
    lines .append (f"data: {json .dumps (data )}")
    lines .append ("")


    mid =len (arg_str )//2 
    for part in [arg_str [:mid ],arg_str [mid :]]:
        data ={
        "choices":[
        {
        "delta":{
        "tool_calls":[
        {
        "index":0 ,
        "function":{"arguments":part },
        }
        ]
        }
        }
        ]
        }
        lines .append (f"data: {json .dumps (data )}")
        lines .append ("")

    lines .append ("data: [DONE]")
    lines .append ("")
    return "\n".join (lines )


def _sse_text_and_tool_call (
text_chunks :list [str ],tool_id :str ,tool_name :str ,tool_args :dict 
)->str :
    """Build SSE with text deltas followed by a tool call."""
    arg_str =json .dumps (tool_args )
    lines =[]


    for chunk in text_chunks :
        data ={"choices":[{"delta":{"content":chunk }}]}
        lines .append (f"data: {json .dumps (data )}")
        lines .append ("")


    data ={
    "choices":[
    {
    "delta":{
    "tool_calls":[
    {
    "index":0 ,
    "id":tool_id ,
    "type":"function",
    "function":{"name":tool_name ,"arguments":""},
    }
    ]
    }
    }
    ]
    }
    lines .append (f"data: {json .dumps (data )}")
    lines .append ("")


    data ={
    "choices":[
    {
    "delta":{
    "tool_calls":[
    {
    "index":0 ,
    "function":{"arguments":arg_str },
    }
    ]
    }
    }
    ]
    }
    lines .append (f"data: {json .dumps (data )}")
    lines .append ("")

    lines .append ("data: [DONE]")
    lines .append ("")
    return "\n".join (lines )







class TestToolCallStreaming :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_tool_call_yields_tool_call_event (self ):
        """SSE stream with tool_call chunks yields ToolCallEvent with correct data."""
        sse_text =_sse_tool_call_lines (
        "call_123","get_weather",{"city":"London","units":"celsius"}
        )
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (
        200 ,
        content =sse_text .encode (),
        headers ={"content-type":"text/event-stream"},
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request (tools =_tool_defs ())):
            events .append (event )

        tool_events =[e for e in events if isinstance (e ,ToolCallEvent )]
        assert len (tool_events )==1 
        tc =tool_events [0 ].tool_call 
        assert tc .id =="call_123"
        assert tc .name =="get_weather"
        assert tc .arguments =={"city":"London","units":"celsius"}

    @pytest .mark .anyio 
    @respx .mock 
    async def test_text_only_no_tool_call_event (self ):
        """SSE stream with text only (no tools in request) yields no ToolCallEvent."""
        sse_text =_sse_lines ("Hello"," world")
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (
        200 ,
        content =sse_text .encode (),
        headers ={"content-type":"text/event-stream"},
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request ()):
            events .append (event )

        tool_events =[e for e in events if isinstance (e ,ToolCallEvent )]
        assert len (tool_events )==0 
        text_events =[e for e in events if isinstance (e ,TextDelta )]
        assert len (text_events )==2 
        assert isinstance (events [0 ],ResponseStarted )
        assert isinstance (events [-1 ],ResponseCompleted )

    @pytest .mark .anyio 
    @respx .mock 
    async def test_text_and_tool_call_both_yielded (self ):
        """SSE with both text and tool_call yields TextDelta and ToolCallEvent."""
        sse_text =_sse_text_and_tool_call (
        ["Let me ","check."],
        "call_456",
        "get_weather",
        {"city":"Paris"},
        )
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (
        200 ,
        content =sse_text .encode (),
        headers ={"content-type":"text/event-stream"},
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request (tools =_tool_defs ())):
            events .append (event )

        text_events =[e for e in events if isinstance (e ,TextDelta )]
        tool_events =[e for e in events if isinstance (e ,ToolCallEvent )]
        assert len (text_events )==2 
        assert text_events [0 ].text =="Let me "
        assert text_events [1 ].text =="check."
        assert len (tool_events )==1 
        assert tool_events [0 ].tool_call .name =="get_weather"
        assert tool_events [0 ].tool_call .arguments =={"city":"Paris"}







class TestRequestBody :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_includes_tools_when_provided (self ):
        """Request body includes tools array when tools are provided."""
        route =respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (
        200 ,
        content =_sse_lines ("ok").encode (),
        headers ={"content-type":"text/event-stream"},
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        async for _ in provider .stream_chat (_request (tools =_tool_defs ())):
            pass 

        assert route .called 
        body =json .loads (route .calls [0 ].request .content )
        assert "tools"in body 
        assert len (body ["tools"])==1 
        tool =body ["tools"][0 ]
        assert tool ["type"]=="function"
        assert tool ["function"]["name"]=="get_weather"
        assert tool ["function"]["description"]=="Get the weather for a city"
        params =tool ["function"]["parameters"]
        assert params ["type"]=="object"
        assert "city"in params ["properties"]
        assert params ["properties"]["city"]["type"]=="string"
        assert "city"in params ["required"]
        assert "units"not in params ["required"]

    @pytest .mark .anyio 
    @respx .mock 
    async def test_omits_tools_when_not_provided (self ):
        """Request body omits tools field when no tools provided."""
        route =respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (
        200 ,
        content =_sse_lines ("ok").encode (),
        headers ={"content-type":"text/event-stream"},
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        async for _ in provider .stream_chat (_request ()):
            pass 

        assert route .called 
        body =json .loads (route .calls [0 ].request .content )
        assert "tools"not in body 







class TestToolErrors :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_backend_400_with_tools_yields_error_event (self ):
        """Backend 400 error when tools included yields graceful ErrorEvent."""
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (
        400 ,
        json ={"error":{"message":"tools not supported by this model"}},
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request (tools =_tool_defs ())):
            events .append (event )

        assert len (events )==1 
        assert isinstance (events [0 ],ErrorEvent )
        assert "tools not supported"in events [0 ].message .lower ()
