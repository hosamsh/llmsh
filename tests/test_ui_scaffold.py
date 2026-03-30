"""Tests for the Textual UI scaffold (LlmshApp and MainScreen)."""

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
from tests .conftest import StubProvider 


def _make_core (profile_name :str ="default",model :str ="test-model")->AppCore :
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
    return AppCore (config =config ,provider =StubProvider ())


class TestLlmshAppMounts :
    @pytest .mark .anyio 
    async def test_app_mounts_without_exception (self ):
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            assert pilot .app is app 

    @pytest .mark .anyio 
    async def test_app_stores_core (self ):
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            assert app .core is core 


class TestHeaderContent :
    @pytest .mark .anyio 
    async def test_header_contains_profile_name (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import HeaderBar 

        core =_make_core (profile_name ="myprofile")
        app =LlmshApp (core )
        async with app .run_test ():
            headers =app .screen .query (HeaderBar )
            assert len (headers )>0 
            header_text =headers .first ().content 
            assert "myprofile"in str (header_text )

    @pytest .mark .anyio 
    async def test_header_contains_model_name (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import HeaderBar 

        core =_make_core (model ="gpt-test-4")
        app =LlmshApp (core )
        async with app .run_test ():
            headers =app .screen .query (HeaderBar )
            assert len (headers )>0 
            header_text =headers .first ().content 
            assert "gpt-test-4"in str (header_text )

    @pytest .mark .anyio 
    async def test_header_contains_app_name (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import HeaderBar 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            headers =app .screen .query (HeaderBar )
            assert len (headers )>0 
            header_text =headers .first ().content 
            assert "llmsh"in str (header_text )


class TestFooterPresent :
    @pytest .mark .anyio 
    async def test_footer_is_present (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            footers =app .screen .query (FooterBar )
            assert len (footers )>0 

    @pytest .mark .anyio 
    async def test_footer_contains_key_hints (self ):
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            footers =app .screen .query (FooterBar )
            footer_text =str (footers .first ().content )
            assert "/help"in footer_text or "Ctrl"in footer_text 


class TestMainScreenLayout :
    @pytest .mark .anyio 
    async def test_transcript_area_is_present (self ):
        from textual .containers import VerticalScroll 

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            scrollables =app .screen .query (VerticalScroll )
            assert len (scrollables )>0 

    @pytest .mark .anyio 
    async def test_input_area_is_present (self ):
        from textual .widgets import Input 

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ():
            inputs =app .screen .query (Input )
            assert len (inputs )>0 
