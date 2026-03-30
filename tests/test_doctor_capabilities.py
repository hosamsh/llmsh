"""Tests for profile capability testing via doctor."""

from __future__ import annotations 

import asyncio 
from typing import AsyncIterator 

import pytest 

from llmsh .models import (
CapabilityReport ,
CapabilityTestResult ,
EndpointConfig ,
ModelCapabilities ,
ProfileConfig ,
ToolCall ,
)
from llmsh .providers .base import (
BaseProvider ,
DoctorReport ,
ProviderEvent ,
ProviderRequest ,
ProviderResult ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
ToolCallEvent ,
)






class _CapabilityStubProvider (BaseProvider ):
    """Base stub with doctor/list_models stubs to satisfy BaseProvider."""

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        return ProviderResult (content ="stub")

    async def list_models (self )->list [str ]:
        return ["test-model"]

    async def doctor (self )->DoctorReport :
        return DoctorReport (checks =[])


class ParisProvider (_CapabilityStubProvider ):
    """Returns 'Paris' for any question."""

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        yield ResponseStarted ()
        yield TextDelta (text ="Paris")
        yield ResponseCompleted (content ="Paris")


class GarbageProvider (_CapabilityStubProvider ):
    """Returns nonsense text."""

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        yield ResponseStarted ()
        yield TextDelta (text ="banana 42 xylophone")
        yield ResponseCompleted (content ="banana 42 xylophone")


class ToolCallingProvider (_CapabilityStubProvider ):
    """Returns a ToolCallEvent for any request that includes tools."""

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        yield ResponseStarted ()
        if request .tools :
            tool =request .tools [0 ]

            args ={}
            if tool .name =="check_weather":
                args ={"city":"Tokyo"}
            yield ToolCallEvent (
            tool_call =ToolCall (id ="call_1",name =tool .name ,arguments =args )
            )
        yield ResponseCompleted (content ="")


class TextOnlyProvider (_CapabilityStubProvider ):
    """Returns only text, ignoring tool definitions."""

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        yield ResponseStarted ()
        yield TextDelta (text ="I cannot use tools")
        yield ResponseCompleted (content ="I cannot use tools")


class SlowProvider (_CapabilityStubProvider ):
    """Returns 'Paris' after a configurable delay."""

    def __init__ (self ,delay_seconds :float =0.0 )->None :
        self ._delay =delay_seconds 

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        await asyncio .sleep (self ._delay )
        yield ResponseStarted ()
        yield TextDelta (text ="Paris")
        yield ResponseCompleted (content ="Paris")


class ExplodingProvider (_CapabilityStubProvider ):
    """Raises an exception on stream_chat."""

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        raise ConnectionError ("connection refused")
        yield 







def _make_profile (
name :str ="default",
model :str ="test-model",
tool_calling :bool =True ,
)->ProfileConfig :
    return ProfileConfig (
    name =name ,
    endpoint ="local",
    model =model ,
    capabilities =ModelCapabilities (tool_calling =tool_calling ),
    )


def _make_endpoint ()->EndpointConfig :
    return EndpointConfig (
    name ="local",
    base_url ="http://localhost:8006/v1",
    auth_mode ="none",
    )


def _find_result (report :CapabilityReport ,name :str )->CapabilityTestResult |None :
    return next ((r for r in report .results if r .name ==name ),None )







class TestQATest :
    @pytest .mark .anyio 
    async def test_qa_passes_when_response_contains_paris (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile ()
        report =await run_capability_tests (profile ,_make_endpoint (),ParisProvider ())

        qa =_find_result (report ,"qa_response")
        assert qa is not None 
        assert qa .passed is True 

    @pytest .mark .anyio 
    async def test_qa_fails_when_response_is_garbage (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile ()
        report =await run_capability_tests (
        profile ,_make_endpoint (),GarbageProvider ()
        )

        qa =_find_result (report ,"qa_response")
        assert qa is not None 
        assert qa .passed is False 

    @pytest .mark .anyio 
    async def test_qa_records_duration (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile ()
        report =await run_capability_tests (profile ,_make_endpoint (),ParisProvider ())

        qa =_find_result (report ,"qa_response")
        assert qa is not None 
        assert qa .duration_ms is not None 
        assert qa .duration_ms >=0 







class TestToolCallingExplicit :
    @pytest .mark .anyio 
    async def test_tool_passes_when_provider_returns_tool_call (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile (tool_calling =True )
        report =await run_capability_tests (
        profile ,_make_endpoint (),ToolCallingProvider ()
        )

        tool =_find_result (report ,"tool_calling")
        assert tool is not None 
        assert tool .passed is True 

    @pytest .mark .anyio 
    async def test_tool_fails_when_provider_returns_only_text (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile (tool_calling =True )
        report =await run_capability_tests (
        profile ,_make_endpoint (),TextOnlyProvider ()
        )

        tool =_find_result (report ,"tool_calling")
        assert tool is not None 
        assert tool .passed is False 







class TestToolCallingImplicit :
    @pytest .mark .anyio 
    async def test_implicit_passes_when_tool_called_with_tokyo (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile (tool_calling =True )
        report =await run_capability_tests (
        profile ,_make_endpoint (),ToolCallingProvider ()
        )

        inference =_find_result (report ,"tool_inference")
        assert inference is not None 
        assert inference .passed is True 

    @pytest .mark .anyio 
    async def test_implicit_fails_when_no_tool_call (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile (tool_calling =True )
        report =await run_capability_tests (
        profile ,_make_endpoint (),TextOnlyProvider ()
        )

        inference =_find_result (report ,"tool_inference")
        assert inference is not None 

        assert inference .passed is None 







class TestToolCallingSkipped :
    @pytest .mark .anyio 
    async def test_tool_tests_skipped_when_tool_calling_disabled (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile (tool_calling =False )
        report =await run_capability_tests (profile ,_make_endpoint (),ParisProvider ())

        tool =_find_result (report ,"tool_calling")
        assert tool is not None 
        assert tool .passed is None 
        assert "disabled"in tool .message .lower ()

        inference =_find_result (report ,"tool_inference")
        assert inference is not None 
        assert inference .passed is None 
        assert "disabled"in inference .message .lower ()







class TestSpeedClassification :
    @pytest .mark .anyio 
    async def test_fast_speed_under_2s (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile (tool_calling =False )
        report =await run_capability_tests (profile ,_make_endpoint (),ParisProvider ())

        speed =_find_result (report ,"speed")
        assert speed is not None 
        assert speed .passed is True 
        assert "fast"in speed .message 

    def test_classify_speed_fast (self )->None :
        from llmsh .doctor import _classify_speed 

        assert _classify_speed (0 )=="fast"
        assert _classify_speed (500 )=="fast"
        assert _classify_speed (1999 )=="fast"

    def test_classify_speed_moderate (self )->None :
        from llmsh .doctor import _classify_speed 

        assert _classify_speed (2000 )=="moderate"
        assert _classify_speed (3500 )=="moderate"
        assert _classify_speed (4999 )=="moderate"

    def test_classify_speed_slow (self )->None :
        from llmsh .doctor import _classify_speed 

        assert _classify_speed (5000 )=="slow"
        assert _classify_speed (10000 )=="slow"







class TestErrorHandling :
    @pytest .mark .anyio 
    async def test_qa_fails_on_connection_error (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile (tool_calling =False )
        report =await run_capability_tests (
        profile ,_make_endpoint (),ExplodingProvider ()
        )

        qa =_find_result (report ,"qa_response")
        assert qa is not None 
        assert qa .passed is False 

    @pytest .mark .anyio 
    async def test_no_crash_on_exception (self )->None :
        """run_capability_tests should never raise — errors become FAIL results."""
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile (tool_calling =True )
        report =await run_capability_tests (
        profile ,_make_endpoint (),ExplodingProvider ()
        )


        assert isinstance (report ,CapabilityReport )
        assert len (report .results )>=3 







class TestCache :
    @pytest .mark .anyio 
    async def test_cache_populated_after_run (self )->None :
        from llmsh .doctor import _capability_cache ,run_capability_tests 

        _capability_cache .clear ()

        profile =_make_profile (name ="cache-test")
        await run_capability_tests (profile ,_make_endpoint (),ParisProvider ())

        assert "cache-test"in _capability_cache 

    @pytest .mark .anyio 
    async def test_get_cached_report_returns_stored_report (self )->None :
        from llmsh .doctor import (
        _capability_cache ,
        get_cached_report ,
        run_capability_tests ,
        )

        _capability_cache .clear ()

        profile =_make_profile (name ="cache-test-2")
        report =await run_capability_tests (profile ,_make_endpoint (),ParisProvider ())

        cached =get_cached_report ("cache-test-2")
        assert cached is report 

    @pytest .mark .anyio 
    async def test_get_cached_report_returns_none_when_not_cached (self )->None :
        from llmsh .doctor import _capability_cache ,get_cached_report 

        _capability_cache .clear ()
        assert get_cached_report ("nonexistent")is None 







class TestCapabilityReportModel :
    @pytest .mark .anyio 
    async def test_report_has_profile_name_and_tested_at (self )->None :
        from llmsh .doctor import run_capability_tests 

        profile =_make_profile (name ="myprofile")
        report =await run_capability_tests (profile ,_make_endpoint (),ParisProvider ())

        assert report .profile_name =="myprofile"
        assert report .tested_at is not None 
