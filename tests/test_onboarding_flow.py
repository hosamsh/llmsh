"""Tests for the first-run onboarding flow (SetupFlow)."""

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
    def __init__ (self ,events :list [ProviderEvent ]|None =None )->None :
        self .events :list [ProviderEvent ]=events or [
        ResponseStarted (),
        TextDelta (text ="hello"),
        ResponseCompleted (content ="hello"),
        ]

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        for event in self .events :
            yield event 

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        return ProviderResult (content ="hello")

    async def doctor (self )->DoctorReport :
        return DoctorReport (
        checks =[DoctorCheck (name ="connectivity",passed =True ,message ="OK")]
        )


def _make_core (
profile_name :str ="default",
model :str ="test-model",
)->AppCore :
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


async def _type_and_submit (pilot ,text :str )->None :
    """Type text into the focused input and press enter."""
    from llmsh .ui .widgets import ChatInput 

    chat_input =pilot .app .screen .query_one (ChatInput )
    await pilot .click (chat_input )
    for ch in text :
        await pilot .press (ch )
    await pilot .press ("enter")
    await pilot .pause ()
    await pilot .pause ()


def _transcript_text (app )->str :
    """Return all transcript message content as a single string."""
    from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

    pane =app .screen .query_one (TranscriptPane )
    widgets =pane .query (MessageWidget )
    return " ".join (str (w .content )for w in widgets )







class TestSetupFlowPostsWelcomeMessage :
    @pytest .mark .anyio 
    async def test_setup_flow_posts_welcome_message (self ):
        """Starting SetupFlow posts a welcome message to the transcript."""
        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            text =_transcript_text (app )
            assert "welcome"in text .lower ()
            assert "no configuration found"in text .lower ()


class TestSetupFlowValidUrlAdvancesToAuth :
    @pytest .mark .anyio 
    async def test_setup_flow_valid_url_advances_to_auth (self ):
        """Entering a valid URL advances the flow to the auth question."""
        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            await _type_and_submit (pilot ,"http://localhost:8006/v1")

            text =_transcript_text (app )
            assert "api key"in text .lower ()


class TestSetupFlowInvalidUrlShowsError :
    @pytest .mark .anyio 
    async def test_setup_flow_invalid_url_shows_error (self ):
        """Entering a URL without http:// shows an error, stays on step 1."""
        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            await _type_and_submit (pilot ,"ftp://bad-url")

            text =_transcript_text (app )
            assert "must start with http"in text .lower ()

            assert app .screen ._active_flow is flow 


class TestSetupFlowAuthYesAsksForEnv :
    @pytest .mark .anyio 
    async def test_setup_flow_auth_yes_asks_for_env (self ):
        """Answering 'y' to auth question asks for environment variable."""
        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            await _type_and_submit (pilot ,"https://api.openai.com/v1")
            await _type_and_submit (pilot ,"y")

            text =_transcript_text (app )
            assert "environment variable"in text .lower ()


class TestSetupFlowAuthNoSkipsToModel :
    @pytest .mark .anyio 
    async def test_setup_flow_auth_no_skips_to_model (self ):
        """Answering 'n' to auth question skips to model question."""
        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            await _type_and_submit (pilot ,"http://localhost:8006/v1")
            await _type_and_submit (pilot ,"n")

            text =_transcript_text (app )
            assert "model"in text .lower ()


class TestSetupFlowCompletesAndSavesConfig :
    @pytest .mark .anyio 
    async def test_setup_flow_completes_and_saves_config (self ,tmp_path ,monkeypatch ):
        """Full flow through all steps writes a valid config file."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()


            await _type_and_submit (pilot ,"http://localhost:8006/v1")

            await _type_and_submit (pilot ,"n")

            await _type_and_submit (pilot ,"llama-3")

            await _type_and_submit (pilot ,"local")
            await pilot .pause ()


            assert app .screen ._active_flow is None 


            assert config_file .exists ()
            data =tomllib .loads (config_file .read_text ())
            assert data ["current_profile"]=="local"
            assert "local"in data ["endpoints"]
            assert data ["endpoints"]["local"]["base_url"]=="http://localhost:8006/v1"
            assert "local"in data ["profiles"]
            assert data ["profiles"]["local"]["model"]=="llama-3"


            text =_transcript_text (app )
            assert "configuration saved"in text .lower ()


class TestSetupFlowWithAuthSavesApiKeyEnv :
    @pytest .mark .anyio 
    async def test_setup_flow_with_auth_saves_api_key_env (self ,tmp_path ,monkeypatch ):
        """Flow with auth=yes saves the API key env variable in config."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            await _type_and_submit (pilot ,"https://api.openai.com/v1")
            await _type_and_submit (pilot ,"y")
            await _type_and_submit (pilot ,"OPENAI_API_KEY")
            await _type_and_submit (pilot ,"gpt-4")
            await _type_and_submit (pilot ,"openai")
            await pilot .pause ()

            assert config_file .exists ()
            data =tomllib .loads (config_file .read_text ())
            assert data ["endpoints"]["openai"]["auth_mode"]=="api_key"
            assert data ["endpoints"]["openai"]["api_key_env"]=="OPENAI_API_KEY"


class TestSetupFlowCancelShowsMessage :
    @pytest .mark .anyio 
    async def test_setup_flow_cancel_shows_message (self ):
        """Pressing Escape during SetupFlow shows cancel message."""
        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            await pilot .press ("escape")
            await pilot .pause ()
            await pilot .pause ()

            assert app .screen ._active_flow is None 
            text =_transcript_text (app )
            assert "setup cancelled"in text .lower ()


class TestChatLaunchesWithoutConfig :
    @pytest .mark .anyio 
    async def test_chat_launches_without_config (self ):
        """LlmshApp with core=None opens without crashing."""
        from llmsh .ui .main import LlmshApp 

        app =LlmshApp (core =None )
        async with app .run_test ()as pilot :
            await pilot .pause ()

            assert app .is_running 


class TestOnboardingStartsAutomaticallyWhenNoConfig :
    @pytest .mark .anyio 
    async def test_onboarding_starts_automatically_when_no_config (self ):
        """When core is None, onboarding flow starts automatically on mount."""
        from llmsh .ui .main import LlmshApp 

        app =LlmshApp (core =None )
        async with app .run_test ()as pilot :
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()

            text =_transcript_text (app )
            assert "welcome"in text .lower ()
            assert "no configuration found"in text .lower ()
            assert app .screen ._active_flow is not None 


class TestHeaderUpdatesAfterOnboarding :
    @pytest .mark .anyio 
    async def test_header_updates_after_onboarding (self ,tmp_path ,monkeypatch ):
        """After completing onboarding, HeaderBar shows the profile and model."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import HeaderBar 

        app =LlmshApp (core =None )
        async with app .run_test ()as pilot :
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()


            await _type_and_submit (pilot ,"http://localhost:8006/v1")
            await _type_and_submit (pilot ,"n")
            await _type_and_submit (pilot ,"llama-3")
            await _type_and_submit (pilot ,"myprofile")
            await pilot .pause ()

            header =app .screen .query_one (HeaderBar )
            header_text =str (header .render ())
            assert "myprofile"in header_text 
            assert "llama-3"in header_text 


class TestPlaceholderChangesBetweenSteps :
    @pytest .mark .anyio 
    async def test_placeholder_changes_between_steps (self ):
        """After submitting the URL step, placeholder changes to 'y or n'."""
        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            chat_input =app .screen .query_one (ChatInput )


            await _type_and_submit (pilot ,"http://localhost:8006/v1")


            assert chat_input .placeholder =="y or n"


class TestSetupFlowInitialPlaceholderIsStep1 :
    @pytest .mark .anyio 
    async def test_setup_flow_initial_placeholder_is_step1 (self ):
        """After start_flow, placeholder reflects step 1 (URL), not the default."""
        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            chat_input =app .screen .query_one (ChatInput )
            assert chat_input .placeholder =="e.g., http://localhost:8006/v1"


class TestCancelFlowNoDuplicateMessage :
    @pytest .mark .anyio 
    async def test_cancel_flow_no_duplicate_message (self ):
        """Cancelling shows only the flow's cancel message, not a generic one too."""
        from llmsh .ui .flows .setup import SetupFlow 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =SetupFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            await pilot .press ("escape")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            widgets =list (pane .query (MessageWidget ))
            cancel_messages =[
            w for w in widgets if "cancelled"in str (w .content ).lower ()
            ]

            assert len (cancel_messages )==1 
            assert "flow cancelled"not in str (cancel_messages [0 ].content ).lower ()
