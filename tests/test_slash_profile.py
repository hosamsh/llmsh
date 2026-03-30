"""Tests for /profile slash command sub-dispatch."""

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


def _make_endpoint (name :str ="local")->EndpointConfig :
    return EndpointConfig (
    name =name ,
    base_url ="http://localhost:8006/v1",
    auth_mode ="none",
    )


def _make_config (
profiles :dict [str ,ProfileConfig ]|None =None ,
endpoints :dict [str ,EndpointConfig ]|None =None ,
current_profile :str ="default",
)->AppConfig :
    if endpoints is None :
        ep =_make_endpoint ("local")
        endpoints ={"local":ep }
    if profiles is None :
        profile =ProfileConfig (
        name =current_profile ,
        endpoint =next (iter (endpoints )),
        model ="test-model",
        capabilities =ModelCapabilities (),
        )
        profiles ={current_profile :profile }
    return AppConfig (
    current_profile =current_profile ,
    endpoints =endpoints ,
    profiles =profiles ,
    )


def _make_config_two_profiles ()->AppConfig :
    ep =_make_endpoint ("local")
    p1 =ProfileConfig (
    name ="default",
    endpoint ="local",
    model ="model-a",
    capabilities =ModelCapabilities (),
    )
    p2 =ProfileConfig (
    name ="other",
    endpoint ="local",
    model ="model-b",
    capabilities =ModelCapabilities (),
    )
    return AppConfig (
    current_profile ="default",
    endpoints ={"local":ep },
    profiles ={"default":p1 ,"other":p2 },
    )


def _make_core (config :AppConfig |None =None )->AppCore :
    if config is None :
        config =_make_config ()
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







class TestProfileList :
    @pytest .mark .anyio 
    async def test_profile_list_shows_profiles_with_active_marker (self ):
        """/profile list displays all profiles and marks the active one."""
        from llmsh .ui .main import LlmshApp 

        config =_make_config_two_profiles ()
        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile list")

            text =_transcript_text (app )
            assert "default"in text 
            assert "other"in text 

            assert "active"in text .lower ()

    @pytest .mark .anyio 
    async def test_profile_list_no_args_defaults_to_list (self ):
        """/profile with no args shows list (not 'current profile')."""
        from llmsh .ui .main import LlmshApp 

        config =_make_config_two_profiles ()
        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile")

            text =_transcript_text (app )

            assert "default"in text 
            assert "other"in text 

    @pytest .mark .anyio 
    async def test_profile_list_shows_endpoint_and_model (self ):
        """/profile list shows endpoint and model for each profile."""
        from llmsh .ui .main import LlmshApp 

        config =_make_config_two_profiles ()
        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile list")

            text =_transcript_text (app )
            assert "model-a"in text 
            assert "model-b"in text 
            assert "local"in text 







class TestProfileAdd :
    @pytest .mark .anyio 
    async def test_profile_add_starts_flow (self ):
        """/profile add starts an interactive flow (placeholder changes)."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile add")


            assert app .screen ._active_flow is not None 

            chat_input =app .screen .query_one (ChatInput )
            assert chat_input .placeholder !="Type a message..."

    @pytest .mark .anyio 
    async def test_profile_add_flow_completes (self ,tmp_path ,monkeypatch ):
        """Full add flow completes and saves the new profile."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile add")


            await _type_and_submit (pilot ,"myprofile")

            await _type_and_submit (pilot ,"local")

            await _type_and_submit (pilot ,"gpt-4")
            await pilot .pause ()


            assert app .screen ._active_flow is None 


            assert config_file .exists ()
            data =tomllib .loads (config_file .read_text ())
            assert "myprofile"in data ["profiles"]
            assert data ["profiles"]["myprofile"]["endpoint"]=="local"
            assert data ["profiles"]["myprofile"]["model"]=="gpt-4"


            text =_transcript_text (app )
            assert "myprofile"in text 
            assert "created"in text .lower ()or "added"in text .lower ()

    @pytest .mark .anyio 
    async def test_profile_add_validates_endpoint_exists (self ,tmp_path ,monkeypatch ):
        """Invalid endpoint shows error and stays in flow."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile add")


            await _type_and_submit (pilot ,"myprofile")

            await _type_and_submit (pilot ,"nonexistent-ep")

            text =_transcript_text (app )
            assert "not found"in text .lower ()or "nonexistent"in text .lower ()

            assert app .screen ._active_flow is not None 

    @pytest .mark .anyio 
    async def test_profile_add_validates_name_unique (self ,tmp_path ,monkeypatch ):
        """Duplicate profile name shows error and re-asks."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile add")


            await _type_and_submit (pilot ,"default")

            text =_transcript_text (app )
            assert "already"in text .lower ()or "exists"in text .lower ()

            assert app .screen ._active_flow is not None 

    @pytest .mark .anyio 
    async def test_profile_add_validates_name_alphanumeric (self ,tmp_path ,monkeypatch ):
        """Name with invalid chars shows error and re-asks."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile add")


            await _type_and_submit (pilot ,"my profile!")

            text =_transcript_text (app )
            assert "alphanumeric"in text .lower ()or "invalid"in text .lower ()
            assert app .screen ._active_flow is not None 







class TestProfileUse :
    @pytest .mark .anyio 
    async def test_profile_use_switches_profile (self ,tmp_path ,monkeypatch ):
        """/profile use <name> switches the active profile."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 

        config =_make_config_two_profiles ()
        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile use other")

            assert core .profile .name =="other"

    @pytest .mark .anyio 
    async def test_profile_use_updates_header (self ,tmp_path ,monkeypatch ):
        """/profile use <name> reflects the new profile in the header."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import HeaderBar 

        config =_make_config_two_profiles ()
        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile use other")

            header_text =str (app .screen .query_one (HeaderBar ).content )
            assert "other"in header_text 

    @pytest .mark .anyio 
    async def test_profile_use_clears_budget_display (self ,tmp_path ,monkeypatch ):
        """/profile use <name> clears stale budget text from footer."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 

        config =_make_config_two_profiles ()
        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :

            footer =app .screen .query_one (FooterBar )
            footer .show_budget (0.65 )
            assert "budget: 65%"in footer ._render_text ()

            await _type_and_submit (pilot ,"/profile use other")


            assert "budget"not in footer ._render_text ()

    @pytest .mark .anyio 
    async def test_profile_use_unknown_shows_error (self ):
        """/profile use <nonexistent> shows error message."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile use nosuchprofile")

            text =_transcript_text (app )
            assert "not found"in text .lower ()or "nosuchprofile"in text .lower ()

    @pytest .mark .anyio 
    async def test_profile_use_no_name_shows_usage (self ):
        """/profile use without a name shows usage hint."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile use")

            text =_transcript_text (app )
            assert "usage"in text .lower ()or "/profile use"in text .lower ()







class TestProfileSetModel :
    @pytest .mark .anyio 
    async def test_profile_set_model_updates_config (self ,tmp_path ,monkeypatch ):
        """/profile set-model <name> <model> updates the model in config."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .config import save_config 
        from llmsh .ui .main import LlmshApp 

        config =_make_config_two_profiles ()
        save_config (config ,config_file )

        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile set-model other new-model-x")

            text =_transcript_text (app )
            assert "new-model-x"in text 
            assert "set"in text .lower ()or "updated"in text .lower ()

            data =tomllib .loads (config_file .read_text ())
            assert data ["profiles"]["other"]["model"]=="new-model-x"

    @pytest .mark .anyio 
    async def test_profile_set_model_updates_header_if_active (
    self ,tmp_path ,monkeypatch 
    ):
        """/profile set-model on active profile refreshes the header."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .config import save_config 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import HeaderBar 

        config =_make_config ()
        save_config (config ,config_file )

        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile set-model default brand-new-model")

            header_text =str (app .screen .query_one (HeaderBar ).content )
            assert "brand-new-model"in header_text 

    @pytest .mark .anyio 
    async def test_profile_set_model_unknown_profile (self ):
        """/profile set-model with unknown profile shows error."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile set-model nosuchprofile gpt-4")

            text =_transcript_text (app )
            assert "not found"in text .lower ()or "nosuchprofile"in text .lower ()

    @pytest .mark .anyio 
    async def test_profile_set_model_missing_args (self ):
        """/profile set-model without enough args shows usage hint."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile set-model")

            text =_transcript_text (app )
            assert "usage"in text .lower ()or "set-model"in text .lower ()







class TestProfileRemove :
    @pytest .mark .anyio 
    async def test_profile_remove_valid (self ,tmp_path ,monkeypatch ):
        """/profile remove <name> removes a non-active profile."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .config import save_config 
        from llmsh .ui .main import LlmshApp 

        config =_make_config_two_profiles ()
        save_config (config ,config_file )

        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile remove other")

            text =_transcript_text (app )
            assert "removed"in text .lower ()
            assert "other"in text 

            data =tomllib .loads (config_file .read_text ())
            assert "other"not in data .get ("profiles",{})

    @pytest .mark .anyio 
    async def test_profile_remove_active_rejected (self ):
        """/profile remove <active> shows error and does not remove it."""
        from llmsh .ui .main import LlmshApp 

        config =_make_config_two_profiles ()
        core =_make_core (config )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile remove default")

            text =_transcript_text (app )
            assert (
            "cannot remove"in text .lower ()
            or "active"in text .lower ()
            or "switch"in text .lower ()
            )

            assert "default"in core ._config .profiles 

    @pytest .mark .anyio 
    async def test_profile_remove_unknown_shows_error (self ):
        """/profile remove <nonexistent> shows error."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile remove nosuchprofile")

            text =_transcript_text (app )
            assert "not found"in text .lower ()or "nosuchprofile"in text .lower ()

    @pytest .mark .anyio 
    async def test_profile_remove_no_name_shows_usage (self ):
        """/profile remove without a name shows usage hint."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile remove")

            text =_transcript_text (app )
            assert "usage"in text .lower ()or "/profile remove"in text .lower ()







class TestProfileUnknownSubcommand :
    @pytest .mark .anyio 
    async def test_profile_unknown_subcommand_shows_usage (self ):
        """/profile <unknown> shows usage hint."""
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await _type_and_submit (pilot ,"/profile bogus")

            text =_transcript_text (app )
            assert "usage"in text .lower ()or "/profile"in text .lower ()
