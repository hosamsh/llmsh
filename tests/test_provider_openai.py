"""Tests for OpenAI-compatible provider."""

import json 

import httpx 
import pytest 
import respx 

from llmsh .errors import ModelNotFoundError ,ProviderError 
from llmsh .models import ChatMessage ,EndpointConfig ,ModelInfo ,UsageInfo 
from llmsh .providers .base import (
ErrorEvent ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
TokenUsageEvent ,
)
from llmsh .providers .openai_compatible import OpenAICompatibleProvider 





API_KEY ="test-key-123"
BASE_URL ="http://localhost:8006/v1"


def _endpoint (auth_mode :str ="api_key",**overrides )->EndpointConfig :
    defaults ={
    "name":"test",
    "base_url":BASE_URL ,
    "auth_mode":auth_mode ,
    "provider_type":"openai_compatible",
    }
    defaults .update (overrides )
    return EndpointConfig (**defaults )


def _request ():
    from llmsh .providers .base import ProviderRequest 

    return ProviderRequest (
    messages =[ChatMessage (role ="user",content ="hello")],
    model ="test-model",
    )


def _chat_response (content :str ="Hello there!",usage :dict |None =None )->dict :
    resp :dict ={
    "id":"chatcmpl-abc",
    "choices":[{"message":{"role":"assistant","content":content }}],
    }
    if usage :
        resp ["usage"]=usage 
    return resp 


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







class TestChat :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_returns_content (self ):
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (200 ,json =_chat_response ("Hi!"))
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        result =await provider .chat (_request ())
        assert result .content =="Hi!"

    @pytest .mark .anyio 
    @respx .mock 
    async def test_returns_usage (self ):
        usage ={
        "prompt_tokens":10 ,
        "completion_tokens":20 ,
        "total_tokens":30 ,
        }
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (200 ,json =_chat_response ("Hi!",usage =usage ))
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        result =await provider .chat (_request ())
        assert result .usage is not None 
        assert result .usage .input_tokens ==10 
        assert result .usage .output_tokens ==20 
        assert result .usage .total_tokens ==30 







class TestStreamChat :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_event_sequence (self ):
        sse_text =_sse_lines ("Hello"," world","!")
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

        assert isinstance (events [0 ],ResponseStarted )
        assert isinstance (events [1 ],TextDelta )
        assert events [1 ].text =="Hello"
        assert isinstance (events [2 ],TextDelta )
        assert events [2 ].text ==" world"
        assert isinstance (events [3 ],TextDelta )
        assert events [3 ].text =="!"
        assert isinstance (events [4 ],ResponseCompleted )
        assert events [4 ].content =="Hello world!"

    @pytest .mark .anyio 
    @respx .mock 
    async def test_usage_emitted_before_completed (self ):
        usage ={
        "prompt_tokens":5 ,
        "completion_tokens":15 ,
        "total_tokens":20 ,
        }
        sse_text =_sse_lines ("Hi",usage =usage )
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


        usage_idx =next (
        i for i ,e in enumerate (events )if isinstance (e ,TokenUsageEvent )
        )
        completed_idx =next (
        i for i ,e in enumerate (events )if isinstance (e ,ResponseCompleted )
        )
        assert usage_idx <completed_idx 
        usage_event =events [usage_idx ]
        assert isinstance (usage_event ,TokenUsageEvent )
        assert usage_event .usage .input_tokens ==5 
        assert usage_event .usage .output_tokens ==15 







class TestStreamChatHttpErrors :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_401_yields_auth_error (self ):
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (401 ,json ={"error":"unauthorized"})
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request ()):
            events .append (event )

        assert len (events )==1 
        assert isinstance (events [0 ],ErrorEvent )
        assert "auth"in events [0 ].message .lower ()

    @pytest .mark .anyio 
    @respx .mock 
    async def test_500_yields_error (self ):
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (500 ,text ="Internal Server Error")
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request ()):
            events .append (event )

        assert len (events )==1 
        assert isinstance (events [0 ],ErrorEvent )


class TestChatHttpErrors :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_400_with_json_error_raises_provider_error (self ):
        error_body ={"error":{"message":"context length exceeded"}}
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (400 ,json =error_body )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        with pytest .raises (ProviderError ,match ="context length exceeded"):
            await provider .chat (_request ())

    @pytest .mark .anyio 
    @respx .mock 
    async def test_400_provider_error_has_error_type (self ):
        error_body ={"error":{"message":"context length exceeded"}}
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (400 ,json =error_body )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        with pytest .raises (ProviderError )as exc_info :
            await provider .chat (_request ())
        assert exc_info .value .error_type =="context_overflow"

    @pytest .mark .anyio 
    @respx .mock 
    async def test_401_raises_provider_error_with_auth_type (self ):
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (401 ,json ={"error":"unauthorized"})
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        with pytest .raises (ProviderError )as exc_info :
            await provider .chat (_request ())
        assert exc_info .value .error_type =="auth"

    @pytest .mark .anyio 
    @respx .mock 
    async def test_500_raises_provider_error_with_server_error_type (self ):
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (500 ,text ="Internal Server Error")
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        with pytest .raises (ProviderError )as exc_info :
            await provider .chat (_request ())
        assert exc_info .value .error_type =="server_error"

    @pytest .mark .anyio 
    @respx .mock 
    async def test_provider_error_inherits_from_llmsh_error (self ):
        from llmsh .errors import LlmshError 
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (500 ,text ="fail")
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        with pytest .raises (LlmshError ):
            await provider .chat (_request ())







class TestListModels :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_returns_model_ids (self ):
        respx .get (f"{BASE_URL }/models").mock (
        return_value =httpx .Response (
        200 ,
        json ={"data":[{"id":"model-a"},{"id":"model-b"}]},
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        models =await provider .list_models ()
        assert models ==["model-a","model-b"]







class TestAuthHeader :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_api_key_sends_bearer_header (self ):
        route =respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (200 ,json =_chat_response ("ok"))
        )
        endpoint =_endpoint (auth_mode ="api_key")
        provider =OpenAICompatibleProvider (endpoint ,api_key =API_KEY )
        await provider .chat (_request ())

        assert route .called 
        request =route .calls [0 ].request 
        assert request .headers ["authorization"]==f"Bearer {API_KEY }"

    @pytest .mark .anyio 
    @respx .mock 
    async def test_no_auth_sends_no_header (self ):
        route =respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (200 ,json =_chat_response ("ok"))
        )
        provider =OpenAICompatibleProvider (_endpoint (auth_mode ="none"))
        await provider .chat (_request ())

        assert route .called 
        request =route .calls [0 ].request 
        assert "authorization"not in request .headers 












class TestBuildBodyStreamOptions :
    def test_streaming_includes_stream_options (self ):
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        body =provider ._build_body (_request (),stream =True )
        assert body ["stream_options"]=={"include_usage":True }

    def test_non_streaming_excludes_stream_options (self ):
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        body =provider ._build_body (_request (),stream =False )
        assert "stream_options"not in body 







class TestGetModelInfo :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_returns_context_length_when_present (self ):
        respx .get (f"{BASE_URL }/models").mock (
        return_value =httpx .Response (
        200 ,
        json ={
        "data":[
        {"id":"model-a","context_length":4096 },
        {"id":"model-b","context_length":8192 },
        ]
        },
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        info =await provider .get_model_info ("model-b")
        assert isinstance (info ,ModelInfo )
        assert info .id =="model-b"
        assert info .context_length ==8192 

    @pytest .mark .anyio 
    @respx .mock 
    async def test_returns_none_context_length_when_absent (self ):
        respx .get (f"{BASE_URL }/models").mock (
        return_value =httpx .Response (
        200 ,
        json ={"data":[{"id":"model-a"},{"id":"model-b"}]},
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        info =await provider .get_model_info ("model-a")
        assert isinstance (info ,ModelInfo )
        assert info .id =="model-a"
        assert info .context_length is None 

    @pytest .mark .anyio 
    @respx .mock 
    async def test_raises_error_when_model_not_found (self ):
        respx .get (f"{BASE_URL }/models").mock (
        return_value =httpx .Response (
        200 ,
        json ={"data":[{"id":"model-a"}]},
        )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        with pytest .raises (ModelNotFoundError ):
            await provider .get_model_info ("nonexistent-model")












class TestChatTruncation :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_truncated_false_by_default (self ):
        """ProviderResult.truncated defaults to False."""
        from llmsh .providers .base import ProviderResult 

        result =ProviderResult (content ="hi")
        assert result .truncated is False 

    @pytest .mark .anyio 
    @respx .mock 
    async def test_truncated_true_when_finish_reason_is_length (self ):
        resp ={
        "id":"chatcmpl-abc",
        "choices":[
        {
        "message":{"role":"assistant","content":"partial"},
        "finish_reason":"length",
        }
        ],
        }
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (200 ,json =resp )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        result =await provider .chat (_request ())
        assert result .truncated is True 

    @pytest .mark .anyio 
    @respx .mock 
    async def test_truncated_false_when_finish_reason_is_stop (self ):
        resp ={
        "id":"chatcmpl-abc",
        "choices":[
        {
        "message":{"role":"assistant","content":"complete"},
        "finish_reason":"stop",
        }
        ],
        }
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (200 ,json =resp )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        result =await provider .chat (_request ())
        assert result .truncated is False 

    @pytest .mark .anyio 
    @respx .mock 
    async def test_truncated_false_when_finish_reason_absent (self ):
        resp =_chat_response ("done")
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (200 ,json =resp )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        result =await provider .chat (_request ())
        assert result .truncated is False 







class TestParseSseLineEmptyChoices :
    def test_empty_choices_with_usage_returns_usage (self ):
        """Usage-only SSE chunk (choices=[]) must not crash and must return usage."""
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        chunk ={
        "choices":[],
        "usage":{
        "prompt_tokens":10 ,
        "completion_tokens":302 ,
        "total_tokens":312 ,
        },
        }
        line =f"data: {json .dumps (chunk )}"
        result =provider ._parse_sse_line (line )
        assert result is not None 
        text ,reasoning ,usage ,tc_chunk =result 
        assert text ==""
        assert reasoning ==""
        assert isinstance (usage ,UsageInfo )
        assert usage .input_tokens ==10 
        assert usage .output_tokens ==302 
        assert usage .total_tokens ==312 
        assert tc_chunk is None 

    def test_empty_choices_without_usage_returns_empty (self ):
        """Chunk with empty choices and no usage returns all empty."""
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        chunk ={"choices":[]}
        line =f"data: {json .dumps (chunk )}"
        result =provider ._parse_sse_line (line )
        assert result is not None 
        text ,reasoning ,usage ,tc_chunk =result 
        assert text ==""
        assert reasoning ==""
        assert usage is None 
        assert tc_chunk is None 

    def test_normal_choices_still_work (self ):
        """Normal chunk with choices still returns content correctly."""
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        chunk ={"choices":[{"delta":{"content":"hello"}}]}
        line =f"data: {json .dumps (chunk )}"
        result =provider ._parse_sse_line (line )
        assert result is not None 
        text ,reasoning ,usage ,tc_chunk =result 
        assert text =="hello"
        assert reasoning ==""
        assert usage is None 
        assert tc_chunk is None 
