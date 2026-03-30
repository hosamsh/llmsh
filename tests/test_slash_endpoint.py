"""Tests for /endpoint slash command sub-dispatch."""

from __future__ import annotations 

import tomllib 
from typing import AsyncIterator 

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
from llmsh .providers .base import (
BaseProvider ,
DoctorCheck ,
DoctorReport ,
ProviderEvent ,
ProviderRequest ,
ProviderResult ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
)






class StubProvider (BaseProvider ):
    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        yield ResponseStarted ()
        yield TextDelta (text ="hello")
        yield ResponseCompleted (content ="hello")

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        return ProviderResult (content ="hello")

    async def doctor (self )->DoctorReport :
        return DoctorReport (
        checks =[DoctorCheck (name ="connectivity",passed =True ,message ="OK")]
        )


def _make_config_with_endpoints (
endpoints :dict [str ,EndpointConfig ]|None =None ,
profile_name :str ="default",
model :str ="test-model",
)->AppConfig :
    if endpoints is None :
        ep =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        )
        endpoints ={"local":ep }
    profile =ProfileConfig (
    name =profile_name ,
    endpoint =next (iter (endpoints )),
    model =model ,
    capabilities =ModelCapabilities (),
    )
    return AppConfig (
    current_profile =profile_name ,
    endpoints =endpoints ,
    profiles ={profile_name :profile },
    )


def _make_core (config :AppConfig |None =None )->AppCore :
    if config is None :
        config =_make_config_with_endpoints ()
    return AppCore (config =config ,provider =StubProvider ())


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







class TestEndpointList :
    @pytest .mark .anyio 
    async def test_endpoint_list_shows_endpoints (self ):
        """/endpoint list displays all configured endpoints."""
        from llmsh .ui .main import LlmshApp 

        ep =EndpointConfig (
        name ="myep",
        base_url ="http://example.com/v1",
        auth_mode ="api_key",
        api_key_env ="MY_KEY",
        )
        config =_make_config_with_endpoints (endpoints ={"myep":ep })
        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint list")

            text =_transcript_text (app )
            assert "myep"in text 
            assert "http://example.com/v1"in text 
            assert "api_key"in text 

    @pytest .mark .anyio 
    async def test_endpoint_list_empty (self ):
        """/endpoint list with no endpoints configured shows an informative message."""
        from llmsh .config import AppConfig 
        from llmsh .models import ProfileConfig 
        from llmsh .ui .main import LlmshApp 


        profile =ProfileConfig (
        name ="default",
        endpoint ="missing",
        model ="test",
        capabilities =ModelCapabilities (),
        )
        config =AppConfig (
        current_profile ="default",
        endpoints ={},
        profiles ={"default":profile },
        )
        core =AppCore (config =config ,provider =StubProvider ())
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint list")

            text =_transcript_text (app )
            assert "no endpoints"in text .lower ()

    @pytest .mark .anyio 
    async def test_endpoint_no_args_shows_list (self ):
        """/endpoint with no args defaults to list behavior."""
        from llmsh .ui .main import LlmshApp 

        ep =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        )
        config =_make_config_with_endpoints (endpoints ={"local":ep })
        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint")

            text =_transcript_text (app )
            assert "local"in text 
            assert "http://localhost:8006/v1"in text 







class TestEndpointAdd :
    @pytest .mark .anyio 
    async def test_endpoint_add_starts_flow (self ):
        """/endpoint add starts an interactive flow (placeholder changes)."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint add")


            assert app .screen ._active_flow is not None 

            chat_input =app .screen .query_one (ChatInput )
            assert chat_input .placeholder !="Type a message..."

    @pytest .mark .anyio 
    async def test_endpoint_add_flow_completes_no_auth (self ,tmp_path ,monkeypatch ):
        """Full add flow (no auth) completes and saves the new endpoint."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint add")


            await _type_and_submit (pilot ,"http://newhost.local/v1")

            await _type_and_submit (pilot ,"none")

            await _type_and_submit (pilot ,"newep")
            await pilot .pause ()


            assert app .screen ._active_flow is None 


            assert config_file .exists ()
            data =tomllib .loads (config_file .read_text ())
            assert "newep"in data ["endpoints"]
            assert data ["endpoints"]["newep"]["base_url"]=="http://newhost.local/v1"
            assert data ["endpoints"]["newep"]["auth_mode"]=="none"


            text =_transcript_text (app )
            assert "newep"in text 
            assert "added"in text .lower ()

    @pytest .mark .anyio 
    async def test_endpoint_add_flow_completes_with_api_key (
    self ,tmp_path ,monkeypatch 
    ):
        """Full add flow with api_key auth saves api_key_env in config."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint add")

            await _type_and_submit (pilot ,"https://api.example.com/v1")
            await _type_and_submit (pilot ,"api_key")
            await _type_and_submit (pilot ,"MY_API_KEY")
            await _type_and_submit (pilot ,"example-ep")
            await pilot .pause ()

            assert app .screen ._active_flow is None 
            assert config_file .exists ()
            data =tomllib .loads (config_file .read_text ())
            assert "example-ep"in data ["endpoints"]
            assert data ["endpoints"]["example-ep"]["auth_mode"]=="api_key"
            assert data ["endpoints"]["example-ep"]["api_key_env"]=="MY_API_KEY"

    @pytest .mark .anyio 
    async def test_endpoint_add_validates_url (self ,tmp_path ,monkeypatch ):
        """Invalid URL shows error and stays in flow."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint add")

            await _type_and_submit (pilot ,"ftp://bad")

            text =_transcript_text (app )
            assert "http"in text .lower ()

            assert app .screen ._active_flow is not None 

    @pytest .mark .anyio 
    async def test_endpoint_add_validates_name_uniqueness (self ,tmp_path ,monkeypatch ):
        """Using an existing endpoint name shows error and re-asks."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 


        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint add")

            await _type_and_submit (pilot ,"http://other.local/v1")
            await _type_and_submit (pilot ,"none")

            await _type_and_submit (pilot ,"local")

            text =_transcript_text (app )
            assert "already"in text .lower ()or "exists"in text .lower ()

            assert app .screen ._active_flow is not None 







class TestEndpointRemove :
    @pytest .mark .anyio 
    async def test_endpoint_remove_valid (self ,tmp_path ,monkeypatch ):
        """/endpoint remove <name> removes an existing endpoint from config."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .config import save_config 
        from llmsh .ui .main import LlmshApp 

        ep =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        )
        config =_make_config_with_endpoints (endpoints ={"local":ep })

        save_config (config ,config_file )

        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint remove local")

            text =_transcript_text (app )
            assert "removed"in text .lower ()
            assert "local"in text 


            data =tomllib .loads (config_file .read_text ())
            assert "local"not in data .get ("endpoints",{})

    @pytest .mark .anyio 
    async def test_endpoint_remove_unknown (self ,tmp_path ,monkeypatch ):
        """/endpoint remove <name> with unknown name shows error."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint remove nonexistent")

            text =_transcript_text (app )
            assert "not found"in text .lower ()or "nonexistent"in text .lower ()

    @pytest .mark .anyio 
    async def test_endpoint_remove_no_name (self ):
        """/endpoint remove with no name shows usage hint."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint remove")

            text =_transcript_text (app )
            assert "usage"in text .lower ()or "/endpoint remove"in text .lower ()

    @pytest .mark .anyio 
    async def test_endpoint_remove_warns_about_referencing_profiles (
    self ,tmp_path ,monkeypatch 
    ):
        """/endpoint remove warns when profiles reference the removed endpoint."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .config import save_config 
        from llmsh .ui .main import LlmshApp 

        ep =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        )
        profile =ProfileConfig (
        name ="myprofile",
        endpoint ="local",
        model ="test",
        capabilities =ModelCapabilities (),
        )
        config =AppConfig (
        current_profile ="myprofile",
        endpoints ={"local":ep },
        profiles ={"myprofile":profile },
        )
        save_config (config ,config_file )
        core =AppCore (config =config ,provider =StubProvider ())
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/endpoint remove local")

            text =_transcript_text (app )
            assert "removed"in text .lower ()
            assert "myprofile"in text .lower ()
