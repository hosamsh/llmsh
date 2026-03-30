"""Tests for provider reasoning/thinking block support."""

import json 

import httpx 
import pytest 
import respx 

from llmsh .models import ChatMessage ,EndpointConfig 
from llmsh .providers .base import (
ProviderRequest ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
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


def _request ()->ProviderRequest :
    return ProviderRequest (
    messages =[ChatMessage (role ="user",content ="What is 15 * 37?")],
    model ="test-model",
    )


def _sse_reasoning_lines (
*reasoning_chunks :str ,content_chunks :list [str ]|None =None 
)->str :
    """Build SSE text with reasoning_content deltas followed by content deltas."""
    lines :list [str ]=[]
    for chunk in reasoning_chunks :
        data ={"choices":[{"delta":{"reasoning_content":chunk }}]}
        lines .append (f"data: {json .dumps (data )}")
        lines .append ("")
    if content_chunks :
        for chunk in content_chunks :
            data ={"choices":[{"delta":{"content":chunk }}]}
            lines .append (f"data: {json .dumps (data )}")
            lines .append ("")
    lines .append ("data: [DONE]")
    lines .append ("")
    return "\n".join (lines )


def _sse_content_only (*chunks :str )->str :
    """Build SSE text with content-only deltas (no reasoning)."""
    lines :list [str ]=[]
    for chunk in chunks :
        data ={"choices":[{"delta":{"content":chunk }}]}
        lines .append (f"data: {json .dumps (data )}")
        lines .append ("")
    lines .append ("data: [DONE]")
    lines .append ("")
    return "\n".join (lines )







class TestReasoningDeltaParsing :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_reasoning_only_yields_reasoning_deltas (self ):
        """SSE with only reasoning_content yields ReasoningDelta,
        no TextDelta."""
        from llmsh .providers .base import ReasoningDelta 

        sse_text =_sse_reasoning_lines ("I need to"," calculate")
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

        reasoning_events =[e for e in events if isinstance (e ,ReasoningDelta )]
        text_events =[e for e in events if isinstance (e ,TextDelta )]
        assert len (reasoning_events )==2 
        assert reasoning_events [0 ].text =="I need to"
        assert reasoning_events [1 ].text ==" calculate"
        assert len (text_events )==0 

    @pytest .mark .anyio 
    @respx .mock 
    async def test_reasoning_then_content_yields_both (self ):
        """SSE with reasoning then content yields ReasoningDelta
        first, then TextDelta."""
        from llmsh .providers .base import ReasoningDelta 

        sse_text =_sse_reasoning_lines (
        "Let me think...",content_chunks =["The answer is ","555."]
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
        async for event in provider .stream_chat (_request ()):
            events .append (event )

        reasoning_events =[e for e in events if isinstance (e ,ReasoningDelta )]
        text_events =[e for e in events if isinstance (e ,TextDelta )]
        assert len (reasoning_events )==1 
        assert reasoning_events [0 ].text =="Let me think..."
        assert len (text_events )==2 
        assert text_events [0 ].text =="The answer is "
        assert text_events [1 ].text =="555."


        first_reasoning_idx =next (
        i for i ,e in enumerate (events )if isinstance (e ,ReasoningDelta )
        )
        first_text_idx =next (
        i for i ,e in enumerate (events )if isinstance (e ,TextDelta )
        )
        assert first_reasoning_idx <first_text_idx 

    @pytest .mark .anyio 
    @respx .mock 
    async def test_content_only_no_reasoning_delta (self ):
        """SSE with content only (no reasoning_content) yields
        no ReasoningDelta — backward compat."""
        from llmsh .providers .base import ReasoningDelta 

        sse_text =_sse_content_only ("Hello"," world")
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

        reasoning_events =[e for e in events if isinstance (e ,ReasoningDelta )]
        text_events =[e for e in events if isinstance (e ,TextDelta )]
        assert len (reasoning_events )==0 
        assert len (text_events )==2 

    @pytest .mark .anyio 
    @respx .mock 
    async def test_response_completed_contains_only_content (self ):
        """ResponseCompleted.content contains only response text, NOT reasoning text."""
        sse_text =_sse_reasoning_lines (
        "Thinking...",content_chunks =["The answer is 555."]
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
        async for event in provider .stream_chat (_request ()):
            events .append (event )

        completed =[e for e in events if isinstance (e ,ResponseCompleted )]
        assert len (completed )==1 
        assert completed [0 ].content =="The answer is 555."
        assert "Thinking"not in completed [0 ].content 







class TestDoctorReasoningCapability :
    @pytest .mark .anyio 
    async def test_reasoning_detected_when_provider_emits_reasoning_delta (self ):
        """Doctor reasoning test passes when provider emits ReasoningDelta."""
        from typing import AsyncIterator 

        from llmsh .doctor import run_capability_tests 
        from llmsh .models import ModelCapabilities ,ProfileConfig 
        from llmsh .providers .base import (
        BaseProvider ,
        DoctorReport ,
        ProviderEvent ,
        ProviderResult ,
        ReasoningDelta ,
        )

        class ReasoningProvider (BaseProvider ):
            async def chat (self ,request :ProviderRequest )->ProviderResult :
                return ProviderResult (content ="stub")

            async def list_models (self )->list [str ]:
                return ["test-model"]

            async def doctor (self )->DoctorReport :
                return DoctorReport (checks =[])

            async def stream_chat (
            self ,request :ProviderRequest 
            )->AsyncIterator [ProviderEvent ]:
                yield ResponseStarted ()
                yield ReasoningDelta (text ="Let me think...")
                yield TextDelta (text ="Paris")
                yield ResponseCompleted (content ="Paris")

        profile =ProfileConfig (
        name ="default",
        endpoint ="local",
        model ="test-model",
        capabilities =ModelCapabilities (tool_calling =False ),
        )
        endpoint =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        )
        report =await run_capability_tests (profile ,endpoint ,ReasoningProvider ())

        reasoning =next ((r for r in report .results if r .name =="reasoning"),None )
        assert reasoning is not None 
        assert reasoning .passed is True 

    @pytest .mark .anyio 
    async def test_reasoning_not_detected_when_no_reasoning_delta (self ):
        """Doctor reasoning test reports 'no' when provider emits no ReasoningDelta."""
        from typing import AsyncIterator 

        from llmsh .doctor import run_capability_tests 
        from llmsh .models import ModelCapabilities ,ProfileConfig 
        from llmsh .providers .base import (
        BaseProvider ,
        DoctorReport ,
        ProviderEvent ,
        ProviderResult ,
        )

        class TextOnlyProvider (BaseProvider ):
            async def chat (self ,request :ProviderRequest )->ProviderResult :
                return ProviderResult (content ="stub")

            async def list_models (self )->list [str ]:
                return ["test-model"]

            async def doctor (self )->DoctorReport :
                return DoctorReport (checks =[])

            async def stream_chat (
            self ,request :ProviderRequest 
            )->AsyncIterator [ProviderEvent ]:
                yield ResponseStarted ()
                yield TextDelta (text ="Paris")
                yield ResponseCompleted (content ="Paris")

        profile =ProfileConfig (
        name ="default",
        endpoint ="local",
        model ="test-model",
        capabilities =ModelCapabilities (tool_calling =False ),
        )
        endpoint =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        )
        report =await run_capability_tests (profile ,endpoint ,TextOnlyProvider ())

        reasoning =next ((r for r in report .results if r .name =="reasoning"),None )
        assert reasoning is not None 
        assert reasoning .passed is False 







class TestCapabilitySummaryReasoning :
    def test_summary_includes_reasoning_yes (self ):
        """Summary includes reasoning:yes when test passed."""
        from llmsh .doctor import format_capability_summary 
        from llmsh .models import CapabilityReport ,CapabilityTestResult 

        report =CapabilityReport (
        profile_name ="test",
        model ="test-model",
        endpoint ="local",
        results =[
        CapabilityTestResult (name ="reasoning",passed =True ,message ="ok"),
        ],
        tested_at =__import__ ("datetime").datetime .now (
        __import__ ("datetime").timezone .utc 
        ),
        )
        summary =format_capability_summary (report )
        assert "reasoning:yes"in summary 

    def test_summary_includes_reasoning_no (self ):
        """Summary includes reasoning:no when test failed."""
        from llmsh .doctor import format_capability_summary 
        from llmsh .models import CapabilityReport ,CapabilityTestResult 

        report =CapabilityReport (
        profile_name ="test",
        model ="test-model",
        endpoint ="local",
        results =[
        CapabilityTestResult (name ="reasoning",passed =False ,message ="nope"),
        ],
        tested_at =__import__ ("datetime").datetime .now (
        __import__ ("datetime").timezone .utc 
        ),
        )
        summary =format_capability_summary (report )
        assert "reasoning:no"in summary 
