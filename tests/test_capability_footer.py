"""Tests for capability footer display (task 049)."""

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






class _StubProvider (BaseProvider ):
    async def stream_chat (
    self ,request :ProviderRequest ,
    )->AsyncIterator [ProviderEvent ]:
        yield ResponseStarted ()
        yield TextDelta (text ="hello")
        yield ResponseCompleted (content ="hello")

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        return ProviderResult (content ="hello")


def _make_core ()->AppCore :
    endpoint =EndpointConfig (
    name ="local",
    base_url ="http://localhost:8006/v1",
    auth_mode ="none",
    provider_type ="openai_compatible",
    )
    profile =ProfileConfig (
    name ="default",
    endpoint ="local",
    model ="test-model",
    capabilities =ModelCapabilities (),
    )
    config =AppConfig (
    current_profile ="default",
    endpoints ={"local":endpoint },
    profiles ={"default":profile },
    )
    return AppCore (config =config ,provider =_StubProvider ())


def _make_report (
qa_passed :bool =True ,
speed_msg :str ="fast (100ms)",
)->CapabilityReport :
    return CapabilityReport (
    profile_name ="test",
    model ="test-model",
    endpoint ="local",
    results =[
    CapabilityTestResult (
    name ="qa_response",passed =qa_passed ,message ="ok",
    ),
    CapabilityTestResult (
    name ="speed",passed =True ,message =speed_msg ,
    ),
    ],
    tested_at =datetime .now (UTC ),
    )







class TestFooterShowsCapabilities :
    @pytest .mark .anyio 
    async def test_footer_shows_capabilities (self )->None :
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            footer =app .screen .query_one (FooterBar )
            report =_make_report ()
            footer .update_capabilities (report )
            text =str (footer .content )
            assert "qa:ok"in text 
            assert "speed:fast"in text 
            assert "Ctrl+C cancel"in text 







class TestFooterClearsIndicators :
    @pytest .mark .anyio 
    async def test_footer_clears_on_none (self )->None :
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            footer =app .screen .query_one (FooterBar )

            report =_make_report ()
            footer .update_capabilities (report )

            footer .update_capabilities (None )
            text =str (footer .content )
            assert "qa:ok"not in text 
            assert text .strip ()=="Ctrl+C cancel  /help commands"







class TestFooterTestingState :
    @pytest .mark .anyio 
    async def test_footer_shows_testing (self )->None :
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            footer =app .screen .query_one (FooterBar )
            footer .set_testing ()
            text =str (footer .content )
            assert "testing..."in text 
            assert "Ctrl+C cancel"in text 







class TestFooterRestoresAfterTesting :
    @pytest .mark .anyio 
    async def test_footer_restores_after_set_testing (self )->None :
        """After set_testing, update_capabilities(None) clears 'testing...'."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            footer =app .screen .query_one (FooterBar )
            footer .set_testing ()
            assert "testing..."in str (footer .content )


            footer .update_capabilities (None )
            text =str (footer .content )
            assert "testing..."not in text 
            assert text .strip ()=="Ctrl+C cancel  /help commands"

    @pytest .mark .anyio 
    async def test_footer_restores_cached_after_set_testing (self )->None :
        """After set_testing, update_capabilities(report) restores indicators."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            footer =app .screen .query_one (FooterBar )
            report =_make_report ()
            footer .set_testing ()
            assert "testing..."in str (footer .content )

            footer .update_capabilities (report )
            text =str (footer .content )
            assert "testing..."not in text 
            assert "qa:ok"in text 
            assert "speed:fast"in text 







class TestFooterWidth :
    @pytest .mark .anyio 
    async def test_footer_fits_80_chars (self )->None :
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            footer =app .screen .query_one (FooterBar )

            report =CapabilityReport (
            profile_name ="test",
            model ="test-model",
            endpoint ="local",
            results =[
            CapabilityTestResult (
            name ="qa_response",passed =True ,message ="ok",
            ),
            CapabilityTestResult (
            name ="tool_calling",passed =True ,
            message ="Tool called (200ms)",
            ),
            CapabilityTestResult (
            name ="speed",passed =True ,
            message ="fast (100ms)",
            ),
            ],
            tested_at =datetime .now (UTC ),
            )
            footer .update_capabilities (report )
            text =str (footer .content )
            assert len (text )<=80 
