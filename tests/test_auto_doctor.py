"""Tests for auto-doctor on profile switch."""

from __future__ import annotations 

from datetime import UTC ,datetime 
from typing import AsyncIterator 

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import (
CapabilityReport ,
CapabilityTestResult ,
EndpointConfig ,
ModelCapabilities ,
ProfileConfig ,
)
from llmsh .providers .base import (
BaseProvider ,
ProviderEvent ,
ProviderRequest ,
ProviderResult ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
)






def _make_profile (
name :str ="default",
model :str ="test-model",
endpoint :str ="local",
tool_calling :bool =False ,
)->ProfileConfig :
    return ProfileConfig (
    name =name ,
    endpoint =endpoint ,
    model =model ,
    capabilities =ModelCapabilities (tool_calling =tool_calling ),
    )


def _make_endpoint (name :str ="local")->EndpointConfig :
    return EndpointConfig (
    name =name ,
    base_url ="http://localhost:8006/v1",
    auth_mode ="none",
    )


def _make_report (
profile_name :str ="default",
model :str ="test-model",
endpoint :str ="local",
)->CapabilityReport :
    return CapabilityReport (
    profile_name =profile_name ,
    model =model ,
    endpoint =endpoint ,
    results =[
    CapabilityTestResult (
    name ="qa_response",
    passed =True ,
    message ="Response: Paris",
    duration_ms =100 ,
    ),
    CapabilityTestResult (
    name ="tool_calling",
    passed =None ,
    message ="Disabled",
    ),
    CapabilityTestResult (
    name ="tool_inference",
    passed =None ,
    message ="Disabled",
    ),
    CapabilityTestResult (
    name ="speed",
    passed =True ,
    message ="fast (100ms)",
    duration_ms =100 ,
    ),
    ],
    tested_at =datetime .now (UTC ),
    )







class TestIsCacheValid :
    def test_returns_true_when_model_and_endpoint_match (self )->None :
        from llmsh .doctor import _capability_cache ,is_cache_valid 

        _capability_cache .clear ()
        profile =_make_profile (
        name ="p1",model ="gpt-4",endpoint ="cloud",
        )
        _capability_cache ["p1"]=_make_report (
        profile_name ="p1",model ="gpt-4",endpoint ="cloud",
        )

        assert is_cache_valid (profile )is True 

    def test_returns_false_when_model_changed (self )->None :
        from llmsh .doctor import _capability_cache ,is_cache_valid 

        _capability_cache .clear ()
        profile =_make_profile (
        name ="p1",model ="gpt-4o",endpoint ="cloud",
        )
        _capability_cache ["p1"]=_make_report (
        profile_name ="p1",model ="gpt-4",endpoint ="cloud",
        )

        assert is_cache_valid (profile )is False 

    def test_returns_false_when_endpoint_changed (self )->None :
        from llmsh .doctor import _capability_cache ,is_cache_valid 

        _capability_cache .clear ()
        profile =_make_profile (
        name ="p1",model ="gpt-4",endpoint ="new-ep",
        )
        _capability_cache ["p1"]=_make_report (
        profile_name ="p1",model ="gpt-4",endpoint ="cloud",
        )

        assert is_cache_valid (profile )is False 

    def test_returns_false_when_no_cache (self )->None :
        from llmsh .doctor import _capability_cache ,is_cache_valid 

        _capability_cache .clear ()
        profile =_make_profile (name ="p1")

        assert is_cache_valid (profile )is False 







class TestFormatCapabilitySummary :
    def test_all_passing (self )->None :
        from llmsh .doctor import format_capability_summary 

        report =CapabilityReport (
        profile_name ="p1",
        model ="m1",
        endpoint ="e1",
        results =[
        CapabilityTestResult (
        name ="qa_response",
        passed =True ,
        message ="Response: Paris",
        ),
        CapabilityTestResult (
        name ="tool_calling",
        passed =True ,
        message ="Tool called (200ms)",
        ),
        CapabilityTestResult (
        name ="tool_inference",
        passed =True ,
        message ="Tool inferred (300ms)",
        ),
        CapabilityTestResult (
        name ="speed",
        passed =True ,
        message ="fast (100ms)",
        ),
        ],
        tested_at =datetime .now (UTC ),
        )

        result =format_capability_summary (report )
        assert "qa:ok"in result 
        assert "tools:yes"in result 
        assert "inference:yes"in result 
        assert "speed:fast (100ms)"in result 

    def test_tools_not_supported (self )->None :
        from llmsh .doctor import format_capability_summary 

        report =CapabilityReport (
        profile_name ="p1",
        model ="m1",
        endpoint ="e1",
        results =[
        CapabilityTestResult (
        name ="qa_response",
        passed =True ,
        message ="Response: Paris",
        ),
        CapabilityTestResult (
        name ="tool_calling",
        passed =False ,
        message ="Model did not call the tool",
        ),
        CapabilityTestResult (
        name ="tool_inference",
        passed =None ,
        message ="Skipped",
        ),
        CapabilityTestResult (
        name ="speed",
        passed =True ,
        message ="slow (4200ms)",
        ),
        ],
        tested_at =datetime .now (UTC ),
        )

        result =format_capability_summary (report )
        assert "qa:ok"in result 
        assert "tools:no"in result 
        assert "speed:slow (4200ms)"in result 

        assert "inference"not in result 

    def test_tools_skipped (self )->None :
        from llmsh .doctor import format_capability_summary 

        report =_make_report ()
        result =format_capability_summary (report )
        assert "qa:ok"in result 
        assert "tools:disabled"in result 
        assert "speed:fast (100ms)"in result 

    def test_qa_failed (self )->None :
        from llmsh .doctor import format_capability_summary 

        report =CapabilityReport (
        profile_name ="p1",
        model ="m1",
        endpoint ="e1",
        results =[
        CapabilityTestResult (
        name ="qa_response",
        passed =False ,
        message ="Expected 'paris', got: banana",
        ),
        CapabilityTestResult (
        name ="speed",
        passed =True ,
        message ="fast (50ms)",
        ),
        ],
        tested_at =datetime .now (UTC ),
        )

        result =format_capability_summary (report )
        assert "qa:fail"in result 







class TestRunCapabilityTestsFields :
    @pytest .mark .anyio 
    async def test_report_has_model_and_endpoint (self )->None :
        from llmsh .doctor import run_capability_tests 
        from tests .test_doctor_capabilities import ParisProvider 

        profile =_make_profile (
        name ="field-test",model ="qwen2.5:14b",endpoint ="local",
        )
        endpoint =_make_endpoint ()
        report =await run_capability_tests (
        profile ,endpoint ,ParisProvider (),
        )

        assert report .model =="qwen2.5:14b"
        assert report .endpoint =="local"







class TestCacheReuse :
    @pytest .mark .anyio 
    async def test_cached_report_reused_when_valid (self )->None :
        from llmsh .doctor import (
        _capability_cache ,
        get_cached_report ,
        is_cache_valid ,
        run_capability_tests ,
        )
        from tests .test_doctor_capabilities import ParisProvider 

        _capability_cache .clear ()
        profile =_make_profile (
        name ="reuse",model ="m1",endpoint ="e1",
        )
        endpoint =_make_endpoint ()


        first =await run_capability_tests (
        profile ,endpoint ,ParisProvider (),
        )


        assert is_cache_valid (profile )is True 
        cached =get_cached_report ("reuse")
        assert cached is first 

    @pytest .mark .anyio 
    async def test_cache_invalidated_after_model_change (self )->None :
        from llmsh .doctor import (
        _capability_cache ,
        is_cache_valid ,
        run_capability_tests ,
        )
        from tests .test_doctor_capabilities import ParisProvider 

        _capability_cache .clear ()
        profile =_make_profile (
        name ="inv",model ="m1",endpoint ="e1",
        )
        endpoint =_make_endpoint ()

        await run_capability_tests (
        profile ,endpoint ,ParisProvider (),
        )


        new_profile =_make_profile (
        name ="inv",model ="m2",endpoint ="e1",
        )
        assert is_cache_valid (new_profile )is False 







class _StubProvider (BaseProvider ):
    async def stream_chat (
    self ,request :ProviderRequest ,
    )->AsyncIterator [ProviderEvent ]:
        yield ResponseStarted ()
        yield TextDelta (text ="hello")
        yield ResponseCompleted (content ="hello")

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        return ProviderResult (content ="hello")


def _make_config_with_two_profiles ()->AppConfig :
    ep =EndpointConfig (
    name ="local",
    base_url ="http://localhost:8006/v1",
    auth_mode ="none",
    )
    p1 =ProfileConfig (
    name ="default",
    endpoint ="local",
    model ="model-a",
    capabilities =ModelCapabilities (),
    )
    p2 =ProfileConfig (
    name ="other",
    endpoint ="local",
    model ="model-b",
    capabilities =ModelCapabilities (),
    )
    return AppConfig (
    current_profile ="default",
    endpoints ={"local":ep },
    profiles ={"default":p1 ,"other":p2 },
    )


async def _type_and_submit (pilot ,text :str )->None :
    from llmsh .ui .widgets import ChatInput 

    chat_input =pilot .app .screen .query_one (ChatInput )
    await pilot .click (chat_input )
    for ch in text :
        await pilot .press (ch )
    await pilot .press ("enter")
    await pilot .pause ()
    await pilot .pause ()


def _transcript_text (app )->str :
    from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

    pane =app .screen .query_one (TranscriptPane )
    return " ".join (str (w .content )for w in pane .query (MessageWidget ))







class TestAutoDoctorErrorHandling :
    @pytest .mark .anyio 
    async def test_auto_doctor_error_shows_retry_hint (
    self ,tmp_path ,monkeypatch ,
    )->None :
        """When capability tests fail, show retry hint instead of crashing."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        async def failing_tests (*args ,**kwargs ):
            raise ConnectionError ("Network unreachable")

        monkeypatch .setattr (
        "llmsh.doctor.run_capability_tests",failing_tests ,
        )
        monkeypatch .setattr (
        "llmsh.ui.slash.is_cache_valid",lambda p :False ,
        )

        from llmsh .ui .main import LlmshApp 

        config =_make_config_with_two_profiles ()
        core =AppCore (config =config ,provider =_StubProvider ())
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile use other")

            text =_transcript_text (app )
            assert "capability test failed"in text .lower ()
            assert "/doctor test"in text 
