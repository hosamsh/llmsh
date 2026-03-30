"""Tests for reasoning/thinking block styling in the transcript."""

from __future__ import annotations 

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
from tests .conftest import StubProvider 


def _make_core (events =None )->AppCore :
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
    provider =StubProvider (events )if events else StubProvider ()
    return AppCore (config =config ,provider =provider )


class TestReasoningWidget :
    """TranscriptPane should manage a separate reasoning widget."""

    @pytest .mark .anyio 
    async def test_begin_reasoning_mounts_widget_with_reasoning_class (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_reasoning ()
            await pilot .pause ()
            reasoning_widgets =pane .query (".reasoning-block")
            assert len (reasoning_widgets )==1 

    @pytest .mark .anyio 
    async def test_append_reasoning_updates_widget_text (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_reasoning ()
            await pilot .pause ()
            pane .append_reasoning ("step 1")
            pane .append_reasoning (" step 2")
            await pilot .pause ()
            assert "step 1"in pane ._reasoning_text 
            assert "step 2"in pane ._reasoning_text 

    @pytest .mark .anyio 
    async def test_end_reasoning_clears_active_reference (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_reasoning ()
            await pilot .pause ()
            pane .end_reasoning ()
            assert pane ._reasoning_widget is None 

    @pytest .mark .anyio 
    async def test_reasoning_widget_is_separate_from_streaming_message (self ):
        """Reasoning block and streaming message are distinct widgets."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_reasoning ()
            await pilot .pause ()
            pane .end_reasoning ()
            pane .begin_streaming ()
            await pilot .pause ()
            pane .append_text ("response text")
            pane .end_streaming ()
            await pilot .pause ()

            reasoning_widgets =pane .query (".reasoning-block")
            msg_widgets =pane .query (MessageWidget )
            assert len (reasoning_widgets )==1 
            assert len (msg_widgets )==1 

            assert "response text"in msg_widgets .first ().content 

    @pytest .mark .anyio 
    async def test_reasoning_widget_appears_before_streaming_message (self ):
        """When reasoning comes first, its widget should be above the response."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_reasoning ()
            await pilot .pause ()
            pane .append_reasoning ("thinking")
            pane .end_reasoning ()
            pane .begin_streaming ()
            await pilot .pause ()
            pane .append_text ("answer")
            pane .end_streaming ()
            await pilot .pause ()


            children =list (pane .children )
            reasoning_idx =None 
            message_idx =None 
            for i ,child in enumerate (children ):
                if "reasoning-block"in child .classes :
                    reasoning_idx =i 
                if isinstance (child ,MessageWidget ):
                    message_idx =i 
            assert reasoning_idx is not None 
            assert message_idx is not None 
            assert reasoning_idx <message_idx 
