"""Tests for chat UI fixes: spacing, focus, and streaming."""

from __future__ import annotations 

import re 

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
from llmsh .providers .base import BaseProvider 
from tests .conftest import StubProvider 


def _make_core (provider :BaseProvider |None =None )->AppCore :
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
    return AppCore (config =config ,provider =provider or StubProvider ())


def _extract_text_lines (svg :str )->dict [int ,str ]:
    """Extract visible text content per line from an SVG screenshot."""
    lines :dict [int ,str ]={}
    for match in re .finditer (
    r'clip-path="url\(#terminal-\d+-line-(\d+)\)">(.*?)</text>',
    svg ,
    re .DOTALL ,
    ):
        line_num =int (match .group (1 ))
        text =match .group (2 ).replace ("&#160;"," ")
        if line_num not in lines :
            lines [line_num ]=""
        lines [line_num ]+=text 
    return lines 


class TestFocusReturnsAfterResponse :
    """Focus should be on ChatInput after the assistant response completes."""

    @pytest .mark .anyio 
    async def test_focus_on_input_after_send_receive_cycle (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause (delay =1.0 )
            await app .workers .wait_for_complete ()
            await pilot .pause (delay =0.5 )
            assert app .screen .focused is chat_input 

    @pytest .mark .anyio 
    async def test_input_not_disabled_after_response (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause (delay =1.0 )
            await app .workers .wait_for_complete ()
            await pilot .pause (delay =0.5 )
            assert not chat_input .disabled 


class TestStreamingUpdatesIncrementally :
    """During streaming, text should appear incrementally, not all at once."""

    @pytest .mark .anyio 
    async def test_each_append_flushes_immediately (self ):
        """Each call to append_text should flush to the widget right away."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_streaming ()
            await pilot .pause ()

            pane .append_text ("first")
            assert pane ._active_message is not None 
            assert "first"in pane ._active_message ._content 

            pane .append_text (" second")
            assert "first second"in pane ._active_message ._content 

            pane .end_streaming ()

    @pytest .mark .anyio 
    async def test_intermediate_text_visible_during_streaming (self ):
        """Transcript should show partial text before streaming ends."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_streaming ()
            await pilot .pause ()

            pane .append_text ("hello")
            await pilot .pause ()

            assert pane ._active_message is not None 
            assert "hello"in pane ._active_message ._content 

            pane .append_text (" world")
            await pilot .pause ()
            assert "hello world"in pane ._active_message ._content 

            pane .end_streaming ()


class TestCompactMessageSpacing :
    """Messages should have minimal vertical spacing - no large gaps."""

    @pytest .mark .anyio 
    async def test_no_blank_lines_between_user_and_assistant (self ):
        """At most 1 blank line between user message and assistant response."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        app =LlmshApp (_make_core ())
        async with app .run_test (size =(80 ,24 ))as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause (delay =1.0 )
            await app .workers .wait_for_complete ()
            await pilot .pause (delay =0.5 )

            svg =app .export_screenshot ()
            lines =_extract_text_lines (svg )


            hi_line =None 
            hello_line =None 
            for num ,text in sorted (lines .items ()):
                stripped =text .strip ()
                if "hi"in stripped and hi_line is None :
                    hi_line =num 
                if "hello"in stripped and hello_line is None :
                    hello_line =num 

            assert hi_line is not None ,"User message 'hi' not found in screenshot"
            assert hello_line is not None ,"Assistant response not found in screenshot"


            gap =hello_line -hi_line -1 
            assert gap <=1 ,(
            f"Too many blank lines ({gap }) between user message (line {hi_line }) "
            f"and assistant response (line {hello_line })"
            )

    @pytest .mark .anyio 
    async def test_screenshot_contains_both_messages (self ):
        """After a conversation, screenshot should show both messages."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        app =LlmshApp (_make_core ())
        async with app .run_test (size =(80 ,24 ))as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause (delay =1.0 )
            await app .workers .wait_for_complete ()
            await pilot .pause (delay =0.5 )

            svg =app .export_screenshot ()
            assert "hi"in svg 
            assert "hello"in svg 
