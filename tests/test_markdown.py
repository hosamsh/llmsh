"""Tests for markdown rendering helper and widget integration."""

from __future__ import annotations 

import pytest 
from textual .widgets import Markdown as TextualMarkdown 

from llmsh .markdown import render_markdown 


class TestRenderMarkdown :
    def test_heading_returns_markdown_widget (self ):
        widget =render_markdown ("# Hello")
        assert isinstance (widget ,TextualMarkdown )

    def test_code_block_returns_markdown_widget (self ):
        widget =render_markdown ("```python\nx = 1\n```")
        assert isinstance (widget ,TextualMarkdown )

    def test_plain_text_returns_markdown_widget (self ):
        widget =render_markdown ("plain text")
        assert isinstance (widget ,TextualMarkdown )

    def test_returns_new_instance_each_call (self ):
        a =render_markdown ("hello")
        b =render_markdown ("hello")
        assert a is not b 


class TestMessageWidgetMarkdown :
    @pytest .mark .anyio 
    async def test_user_message_does_not_contain_markdown_widget (self ):
        from llmsh .app import AppCore 
        from llmsh .config import AppConfig 
        from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 
        from tests .conftest import StubProvider 

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
        core =AppCore (config =config ,provider =StubProvider ())
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            widget =MessageWidget (role ="user",content ="hello world")
            await app .screen .mount (widget )
            await pilot .pause ()
            assert not widget .query (TextualMarkdown )

    @pytest .mark .anyio 
    async def test_assistant_message_contains_markdown_widget (self ):
        from llmsh .app import AppCore 
        from llmsh .config import AppConfig 
        from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget 
        from tests .conftest import StubProvider 

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
        core =AppCore (config =config ,provider =StubProvider ())
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            widget =MessageWidget (role ="assistant",content ="# Hello")
            await app .screen .mount (widget )
            await pilot .pause ()
            assert widget .query (TextualMarkdown )
