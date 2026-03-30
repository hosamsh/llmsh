"""Tests for ChatInput widget and cancel keybinding."""

from __future__ import annotations 

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
from tests .conftest import StubProvider 


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
    return AppCore (config =config ,provider =StubProvider ())


class TestChatInputSubmit :
    @pytest .mark .anyio 
    async def test_type_and_enter_clears_input (self ):
        """After pressing Enter, the input value is empty."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","i")
            await pilot .press ("enter")
            await pilot .pause ()
            assert chat_input .value ==""

    @pytest .mark .anyio 
    async def test_type_and_enter_posts_message_with_text (self ):
        """Pressing Enter posts a SubmitMessage carrying the typed text."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,SubmitMessage 

        received :list [SubmitMessage ]=[]

        class TrackingApp (LlmshApp ):
            def on_submit_message (self ,msg :SubmitMessage )->None :
                received .append (msg )

        app =TrackingApp (_make_core ())
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("h","e","l","l","o")
            await pilot .press ("enter")
            await pilot .pause ()
            assert len (received )==1 
            assert received [0 ].text =="hello"

    @pytest .mark .anyio 
    async def test_enter_with_empty_input_does_not_post_message (self ):
        """Pressing Enter with no text does not post a SubmitMessage."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,SubmitMessage 

        received :list [SubmitMessage ]=[]

        class TrackingApp (LlmshApp ):
            def on_submit_message (self ,msg :SubmitMessage )->None :
                received .append (msg )

        app =TrackingApp (_make_core ())
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            await pilot .press ("enter")
            await pilot .pause ()
            assert len (received )==0 


class TestCancelAction :
    @pytest .mark .anyio 
    async def test_escape_with_no_active_request_is_noop (self ):
        """Pressing Escape when not streaming raises no exception."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        assert not core .is_streaming 

        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await pilot .press ("escape")
            await pilot .pause ()


    @pytest .mark .anyio 
    async def test_escape_when_streaming_calls_cancel (self ):
        """Pressing Escape while streaming calls AppCore.cancel()."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        core .is_streaming =True 

        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await pilot .press ("escape")
            await pilot .pause ()
            assert core ._cancelled is True 


class TestChatInputDisabledState :
    @pytest .mark .anyio 
    async def test_input_enabled_by_default (self ):
        """ChatInput starts enabled (not disabled)."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        app =LlmshApp (_make_core ())
        async with app .run_test ():
            chat_input =app .screen .query_one (ChatInput )
            assert not chat_input .disabled 

    @pytest .mark .anyio 
    async def test_input_disabled_when_streaming_true (self ):
        """Setting is_streaming=True on core disables ChatInput."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import MainScreen 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            screen =app .screen 
            assert isinstance (screen ,MainScreen )
            screen .set_streaming (True )
            await pilot .pause ()
            chat_input =app .screen .query_one (ChatInput )
            assert chat_input .disabled 

    @pytest .mark .anyio 
    async def test_input_re_enabled_when_streaming_false (self ):
        """Setting is_streaming=False re-enables ChatInput."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import MainScreen 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            screen =app .screen 
            assert isinstance (screen ,MainScreen )
            screen .set_streaming (True )
            await pilot .pause ()
            screen .set_streaming (False )
            await pilot .pause ()
            chat_input =app .screen .query_one (ChatInput )
            assert not chat_input .disabled 
