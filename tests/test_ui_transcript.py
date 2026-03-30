"""Tests for chat transcript widgets (MessageWidget and TranscriptPane)."""

from __future__ import annotations 

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import ChatMessage ,EndpointConfig ,ModelCapabilities ,ProfileConfig 
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


class TestMessageWidgetRendering :
    @pytest .mark .anyio 
    async def test_user_message_has_user_css_class (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 

        app =LlmshApp (_make_core ())
        async with app .run_test ():
            widget =MessageWidget (role ="user",content ="hello")
            assert "user-message"in widget .classes 

    @pytest .mark .anyio 
    async def test_assistant_message_has_assistant_css_class (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 

        app =LlmshApp (_make_core ())
        async with app .run_test ():
            widget =MessageWidget (role ="assistant",content ="hello")
            assert "assistant-message"in widget .classes 

    @pytest .mark .anyio 
    async def test_message_widget_renders_content (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 

        app =LlmshApp (_make_core ())
        async with app .run_test ():
            widget =MessageWidget (role ="user",content ="test content")
            assert "test content"in widget .content 

    @pytest .mark .anyio 
    async def test_message_widget_renders_role_label (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 

        app =LlmshApp (_make_core ())
        async with app .run_test ():
            widget =MessageWidget (role ="user",content ="hi")
            assert "user"in widget .content 


class TestTranscriptPaneAddMessage :
    @pytest .mark .anyio 
    async def test_add_user_message_widget_present_in_transcript (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            msg =ChatMessage (role ="user",content ="hello there")
            await pane .add_message (msg )
            await pilot .pause ()
            widgets =pane .query (MessageWidget )
            assert len (widgets )==1 

    @pytest .mark .anyio 
    async def test_add_message_content_is_accessible (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            msg =ChatMessage (role ="user",content ="my message")
            await pane .add_message (msg )
            await pilot .pause ()
            widget =pane .query (MessageWidget ).first ()
            assert "my message"in widget .content 


class TestTranscriptPaneStreaming :
    @pytest .mark .anyio 
    async def test_streaming_sequence_produces_correct_text (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_streaming ()
            await pilot .pause ()
            pane .append_text ("hello")
            pane .append_text (" world")
            pane .end_streaming ()
            await pilot .pause ()
            widgets =pane .query (MessageWidget )
            assert len (widgets )==1 
            assert "hello world"in widgets .first ().content 

    @pytest .mark .anyio 
    async def test_streaming_message_has_streaming_class_during_stream (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_streaming ()
            await pilot .pause ()
            assert pane ._active_message is not None 
            assert "streaming"in pane ._active_message .classes 

    @pytest .mark .anyio 
    async def test_end_streaming_removes_streaming_class (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_streaming ()
            await pilot .pause ()
            pane .append_text ("some text")
            pane .end_streaming ()
            await pilot .pause ()

            assert pane ._active_message is None 

    @pytest .mark .anyio 
    async def test_begin_streaming_twice_is_safe (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )
            pane .begin_streaming ()
            await pilot .pause ()
            pane .begin_streaming ()
            await pilot .pause ()

            streaming_widgets =[
            w for w in pane .query (MessageWidget )if "streaming"in w .classes 
            ]
            assert len (streaming_widgets )==1 


class TestSystemMessageWidget :
    @pytest .mark .anyio 
    async def test_system_message_has_system_css_class (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 

        app =LlmshApp (_make_core ())
        async with app .run_test ():
            widget =MessageWidget (role ="system",content ="hello system")
            assert "system-message"in widget .classes 

    @pytest .mark .anyio 
    async def test_system_message_does_not_have_assistant_css_class (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 

        app =LlmshApp (_make_core ())
        async with app .run_test ():
            widget =MessageWidget (role ="system",content ="hello system")
            assert "assistant-message"not in widget .classes 

    @pytest .mark .anyio 
    async def test_system_message_content_accessible (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 

        app =LlmshApp (_make_core ())
        async with app .run_test ():
            widget =MessageWidget (role ="system",content ="system info text")
            assert "system info text"in widget .content 

    @pytest .mark .anyio 
    async def test_system_message_has_no_role_label_in_composed_statics (self ):
        """System messages render plain text with no 'You:' or 'Assistant:' prefix."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane_widget =MessageWidget (role ="system",content ="plain text")
            await app .screen .mount (pane_widget )
            await pilot .pause ()
            statics =pane_widget .query ("Static")
            rendered =" ".join (str (s .content )for s in statics )
            assert "You:"not in rendered 
            assert "Assistant:"not in rendered 

    @pytest .mark .anyio 
    async def test_slash_command_response_is_system_message (self ):
        """After a slash command, the response widget has system-message CSS class."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/help":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            widgets =pane .query (MessageWidget )
            assert len (widgets )>0 
            system_widgets =[w for w in widgets if "system-message"in w .classes ]
            assert len (system_widgets )>0 
            assert all ("assistant-message"not in w .classes for w in system_widgets )


class TestTranscriptPaneMultipleMessages :
    @pytest .mark .anyio 
    async def test_add_message_after_streaming_both_present (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

        app =LlmshApp (_make_core ())
        async with app .run_test ()as pilot :
            pane =app .screen .query_one (TranscriptPane )


            pane .begin_streaming ()
            await pilot .pause ()
            pane .append_text ("streamed response")
            pane .end_streaming ()
            await pilot .pause ()


            msg =ChatMessage (role ="user",content ="follow up")
            await pane .add_message (msg )
            await pilot .pause ()

            widgets =pane .query (MessageWidget )
            assert len (widgets )==2 
