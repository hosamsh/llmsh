"""Tests for the chat command and SubmitMessage event handling."""

from __future__ import annotations 

import asyncio 
from typing import AsyncIterator 

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig ,UsageInfo 
from llmsh .providers .base import (
ErrorEvent ,
ProviderEvent ,
ProviderRequest ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
TokenUsageEvent ,
)
from tests .conftest import StubProvider 


def _make_core (
provider :StubProvider |None =None ,
profile_name :str ="default",
model :str ="test-model",
)->AppCore :
    """Build a minimal AppCore backed by a StubProvider."""
    endpoint =EndpointConfig (
    name ="local",
    base_url ="http://localhost:8006/v1",
    auth_mode ="none",
    provider_type ="openai_compatible",
    )
    profile =ProfileConfig (
    name =profile_name ,
    endpoint ="local",
    model =model ,
    capabilities =ModelCapabilities (),
    )
    config =AppConfig (
    current_profile =profile_name ,
    endpoints ={"local":endpoint },
    profiles ={profile_name :profile },
    )
    return AppCore (config =config ,provider =provider or StubProvider ())


class SlowProvider (StubProvider ):
    """A provider that yields events with async delays for cancel testing."""

    def __init__ (self ,events :list [ProviderEvent ],delay :float =0.05 )->None :
        super ().__init__ (events =events )
        self .delay =delay 

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        self .requests .append (request )
        for event in self .events :
            await asyncio .sleep (self .delay )
            yield event 


class TestSubmitMessageStreaming :
    @pytest .mark .anyio 
    async def test_submit_yields_two_deltas_transcript_shows_assembled_text (self ):
        """Two TextDelta events are assembled in transcript."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        stub =StubProvider (events =[
        ResponseStarted (),
        TextDelta (text ="hello "),
        TextDelta (text ="world"),
        ResponseCompleted (content ="hello world"),
        ])
        core =_make_core (provider =stub )
        app =LlmshApp (core )

        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            widgets =pane .query (MessageWidget )

            assert len (widgets )>=2 

            assistant_widgets =[
            w for w in widgets if "assistant-message"in w .classes 
            ]
            assert len (assistant_widgets )>=1 
            assert "hello world"in assistant_widgets [-1 ]._content 

    @pytest .mark .anyio 
    async def test_submit_adds_user_message_to_transcript (self ):
        """Submitting a message adds a user message widget to transcript."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        stub =StubProvider (events =[
        ResponseStarted (),
        TextDelta (text ="reply"),
        ResponseCompleted (content ="reply"),
        ])
        core =_make_core (provider =stub )
        app =LlmshApp (core )

        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            user_widgets =[
            w for w in pane .query (MessageWidget )
            if "user-message"in w .classes 
            ]
            assert len (user_widgets )==1 
            assert "hi"in user_widgets [0 ]._content 

    @pytest .mark .anyio 
    async def test_streaming_false_after_completion (self ):
        """After response completes, is_streaming is False and provider was called."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        stub =StubProvider (events =[
        ResponseStarted (),
        TextDelta (text ="done"),
        ResponseCompleted (content ="done"),
        ])
        core =_make_core (provider =stub )
        app =LlmshApp (core )

        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()


            pane =app .screen .query_one (TranscriptPane )
            assert len (pane .query (MessageWidget ))>=1 
            assert core .is_streaming is False 

            assert len (stub .requests )==1 


class TestCancelMidStream :
    @pytest .mark .anyio 
    async def test_cancel_mid_stream_partial_text_visible (self ):
        """Pressing escape mid-stream preserves partial text and stops streaming."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        slow =SlowProvider (
        events =[
        ResponseStarted (),
        TextDelta (text ="first "),
        TextDelta (text ="second "),
        TextDelta (text ="third "),
        TextDelta (text ="fourth"),
        ResponseCompleted (content ="first second third fourth"),
        ],
        delay =0.1 ,
        )
        core =_make_core (provider =slow )
        app =LlmshApp (core )

        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")

            await asyncio .sleep (0.25 )
            await pilot .press ("escape")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            assistant_widgets =[
            w for w in pane .query (MessageWidget )
            if "assistant-message"in w .classes 
            ]

            assert len (assistant_widgets )>=1 
            content =assistant_widgets [-1 ]._content 
            assert "first"in content 

            assert core .is_streaming is False 


class TestErrorHandling :
    @pytest .mark .anyio 
    async def test_error_event_shown_in_transcript (self ):
        """Provider ErrorEvent is shown in transcript without crashing."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        stub =StubProvider (events =[
        ResponseStarted (),
        ErrorEvent (message ="provider exploded"),
        ])
        core =_make_core (provider =stub )
        app =LlmshApp (core )

        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            widgets =pane .query (MessageWidget )

            all_text =" ".join (w ._content for w in widgets )
            assert "provider exploded"in all_text 

    @pytest .mark .anyio 
    async def test_app_responsive_after_error (self ):
        """App remains usable after an error event."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        stub =StubProvider (events =[
        ResponseStarted (),
        ErrorEvent (message ="boom"),
        ])
        core =_make_core (provider =stub )
        app =LlmshApp (core )

        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()


            pane =app .screen .query_one (TranscriptPane )
            all_text =" ".join (w ._content for w in pane .query (MessageWidget ))
            assert "boom"in all_text 

            assert not chat_input .disabled 
            assert core .is_streaming is False 


class TestTokenUsageDisplay :
    @pytest .mark .anyio 
    async def test_token_usage_shown_in_footer (self ):
        """Token usage is displayed in footer after response completes."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 
        from llmsh .ui .widgets import ChatInput 

        stub =StubProvider (events =[
        ResponseStarted (),
        TextDelta (text ="ok"),
        ResponseCompleted (content ="ok"),
        TokenUsageEvent (usage =UsageInfo (total_tokens =42 )),
        ])
        core =_make_core (provider =stub )
        app =LlmshApp (core )

        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            footer =app .screen .query_one (FooterBar )
            footer_text =str (footer .content )
            assert "42"in footer_text 
