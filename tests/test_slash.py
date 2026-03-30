"""Tests for slash command parser and handler dispatch."""

from __future__ import annotations 

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
extra_profiles :dict |None =None ,
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
    profiles ={profile_name :profile }
    if extra_profiles :
        for ep_name ,ep_model in extra_profiles .items ():
            profiles [ep_name ]=ProfileConfig (
            name =ep_name ,
            endpoint ="local",
            model =ep_model ,
            capabilities =ModelCapabilities (),
            )
    config =AppConfig (
    current_profile =profile_name ,
    endpoints ={"local":endpoint },
    profiles =profiles ,
    )
    return AppCore (config =config ,provider =StubProvider ())







class TestParseSlashCommand :
    def test_bare_command_returns_command_and_empty_args (self ):
        from llmsh .ui .slash import parse_slash_command 

        result =parse_slash_command ("/help")
        assert result ==("help",[])

    def test_command_with_single_arg (self ):
        from llmsh .ui .slash import parse_slash_command 

        result =parse_slash_command ("/profile local")
        assert result ==("profile",["local"])

    def test_command_with_arg_profile (self ):
        from llmsh .ui .slash import parse_slash_command 

        result =parse_slash_command ("/profile local")
        assert result ==("profile",["local"])

    def test_non_slash_input_returns_none (self ):
        from llmsh .ui .slash import parse_slash_command 

        result =parse_slash_command ("hello world")
        assert result is None 

    def test_bare_slash_returns_none (self ):
        from llmsh .ui .slash import parse_slash_command 

        result =parse_slash_command ("/")
        assert result is None 

    def test_leading_whitespace_is_stripped (self ):
        from llmsh .ui .slash import parse_slash_command 

        result =parse_slash_command ("  /help  ")
        assert result ==("help",[])

    def test_multiple_args (self ):
        from llmsh .ui .slash import parse_slash_command 

        result =parse_slash_command ("/profile set-model default gpt-4")
        assert result ==("profile",["set-model","default","gpt-4"])

    def test_empty_string_returns_none (self ):
        from llmsh .ui .slash import parse_slash_command 

        result =parse_slash_command ("")
        assert result is None 







class TestSlashClear :
    @pytest .mark .anyio 
    async def test_clear_empties_transcript (self ):
        """After /clear, the transcript pane has no message widgets."""
        from llmsh .models import ChatMessage 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :

            pane =app .screen .query_one (TranscriptPane )
            await pane .add_message (ChatMessage (role ="user",content ="hello"))
            await pilot .pause ()
            assert len (pane .query (MessageWidget ))==1 


            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/clear":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()


            assert len (pane .query (MessageWidget ))==0 

    @pytest .mark .anyio 
    async def test_clear_resets_core_messages (self ):
        """After /clear, core._messages is reset (system prompt preserved)."""
        from llmsh .models import ChatMessage 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()

        core ._messages .append (ChatMessage (role ="user",content ="hello"))
        core ._messages .append (ChatMessage (role ="assistant",content ="hi"))
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/clear":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()


            non_system =[m for m in core ._messages if m .role !="system"]
            assert non_system ==[]

    @pytest .mark .anyio 
    async def test_clear_resets_budget_display (self ):
        """After /clear, the footer budget text is cleared."""
        from llmsh .budget import ContextBudget 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import FooterBar 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()

        core ._budget =ContextBudget (context_length =4096 ,reserved_output =1024 )
        core ._budget .update_usage (2000 )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            footer =app .screen .query_one (FooterBar )

            footer .show_budget (core ._budget .utilization )
            assert "budget"in footer ._render_text ()

            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/clear":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()


            assert "budget"not in footer ._render_text ()


class TestSlashHelp :
    @pytest .mark .anyio 
    async def test_help_shows_grouped_categories_in_transcript (self ):
        """After /help, the transcript contains grouped command categories."""
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
            combined =" ".join (str (w .content )for w in widgets )

            assert (
            "Chat"in combined 
            or "Configuration"in combined 
            or "Analysis"in combined 
            )

    @pytest .mark .anyio 
    async def test_help_includes_key_commands (self ):
        """/help output includes endpoint, profile, and session commands."""
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
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "endpoint"in combined 
            assert "profile"in combined 
            assert "session"in combined 

    @pytest .mark .anyio 
    async def test_help_does_not_mention_model_command (self ):
        """/help output does not list /model as a standalone command."""
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
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "/model"not in combined 


class TestSlashCancelNoFlow :
    @pytest .mark .anyio 
    async def test_cancel_with_no_active_flow_shows_nothing_to_cancel (self ):
        """/cancel with no active flow shows 'Nothing to cancel' message."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/cancel":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert (
            "nothing to cancel"in combined .lower ()
            or "cancel"in combined .lower ()
            )


class TestSlashUnknownCommand :
    @pytest .mark .anyio 
    async def test_unknown_command_shows_error_in_transcript (self ):
        """An unrecognised /command adds an error message to the transcript."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/notacommand":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            widgets =pane .query (MessageWidget )
            assert len (widgets )>0 
            combined =" ".join (str (w .content )for w in widgets )
            assert "unknown"in combined .lower ()or "error"in combined .lower ()

    @pytest .mark .anyio 
    async def test_model_command_is_not_recognized (self ):
        """/model is no longer a valid command — shows Unknown command."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/model":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "unknown"in combined .lower ()



class TestSlashProfile :
    @pytest .mark .anyio 
    async def test_profile_without_args_shows_list (self ):
        """/profile with no args shows the profile list (including active profile)."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core (profile_name ="myprofile")
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/profile":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "myprofile"in combined 

    @pytest .mark .anyio 
    async def test_profile_use_calls_switch_profile (self ,tmp_path ,monkeypatch ):
        """/profile use <name> switches the profile on core."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core (
        profile_name ="default",
        extra_profiles ={"other":"other-model"},
        )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/profile use other":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            assert core .profile .name =="other"

    @pytest .mark .anyio 
    async def test_profile_use_updates_header (self ,tmp_path ,monkeypatch ):
        """/profile use <name> reflects the new profile in the header."""
        config_file =tmp_path /"config.toml"
        monkeypatch .setattr ("llmsh.config.config_path",lambda :config_file )

        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .screens import HeaderBar 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core (
        profile_name ="default",
        extra_profiles ={"other":"other-model"},
        )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/profile use other":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            header_text =str (app .screen .query_one (HeaderBar ).content )
            assert "other"in header_text 


class TestSlashSave :
    @pytest .mark .anyio 
    async def test_save_shows_confirmation_in_transcript (self ,tmp_path ):
        """/save shows a confirmation message in transcript."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :

            import llmsh .paths as paths_mod 

            original =paths_mod .sessions_dir 
            paths_mod .sessions_dir =lambda :tmp_path /"sessions"
            try :
                chat_input =app .screen .query_one (ChatInput )
                await pilot .click (chat_input )
                for ch in "/save":
                    await pilot .press (ch )
                await pilot .press ("enter")
                await pilot .pause ()
                await pilot .pause ()

                pane =app .screen .query_one (TranscriptPane )
                combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
                assert "saved"in combined .lower ()or "session"in combined .lower ()
            finally :
                paths_mod .sessions_dir =original 


class TestSlashLoad :
    @pytest .mark .anyio 
    async def test_load_shows_sessions_list_in_transcript (self ,tmp_path ):
        """/load shows a numbered list of recent sessions."""
        from llmsh .sessions .store import SessionStore 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        store =SessionStore (sessions_path )
        rec =store .create (profile ="default",model ="test-model")
        store .save (rec )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            import llmsh .paths as paths_mod 

            original =paths_mod .sessions_dir 
            paths_mod .sessions_dir =lambda :sessions_path 
            try :
                chat_input =app .screen .query_one (ChatInput )
                await pilot .click (chat_input )
                for ch in "/load":
                    await pilot .press (ch )
                await pilot .press ("enter")
                await pilot .pause ()
                await pilot .pause ()

                pane =app .screen .query_one (TranscriptPane )
                combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))

                assert (
                "session"in combined .lower ()
                or "new session"in combined .lower ()
                )
            finally :
                paths_mod .sessions_dir =original 


class TestSlashCopy :
    @pytest .mark .anyio 
    async def test_copy_shows_placeholder_message (self ):
        """/copy shows a placeholder message since clipboard is not yet done."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/copy":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            widgets =pane .query (MessageWidget )
            assert len (widgets )>0 
            combined =" ".join (str (w .content )for w in widgets )

            assert "hello"not in combined .lower ()

            assert (
            "clipboard"in combined .lower ()
            or "coming soon"in combined .lower ()
            or "not available"in combined .lower ()
            )


class TestSlashDoctor :
    @pytest .mark .anyio 
    async def test_doctor_shows_results_in_transcript (self ):
        """/doctor runs checks and displays results."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/doctor":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            widgets =pane .query (MessageWidget )
            assert len (widgets )>0 
            combined =" ".join (str (w .content )for w in widgets )


            assert (
            "connectivity"in combined .lower ()
            or "doctor"in combined .lower ()
            or "check"in combined .lower ()
            )
            assert "hello"not in combined .lower ()


class TestSlashSummarize :
    def test_summarize_in_command_names (self ):
        """'summarize' is registered in _COMMAND_NAMES."""
        from llmsh .ui .slash import _COMMAND_NAMES 

        assert "summarize"in _COMMAND_NAMES 

    @pytest .mark .anyio 
    async def test_summarize_missing_args_shows_usage (self ):
        """/summarize with no args shows usage message."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/summarize":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "usage"in combined .lower ()

    @pytest .mark .anyio 
    async def test_summarize_path_only_shows_usage (self ):
        """/summarize with only a path (no instruction) shows usage."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/summarize somefile.txt":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "usage"in combined .lower ()

    @pytest .mark .anyio 
    async def test_summarize_nonexistent_path_shows_error (self ):
        """/summarize with a nonexistent path shows 'Not found' error."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/summarize /no/such/file.txt find errors":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "not found"in combined .lower ()

    @pytest .mark .anyio 
    async def test_summarize_valid_file_shows_result (self ,tmp_path ,monkeypatch ):
        """/summarize with a valid file calls summarize_file and shows the result."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 


        test_file =tmp_path /"data.txt"
        test_file .write_text ("some log data here")


        async def fake_summarize_file (**kwargs ):
            return ("Summary: found 3 warnings",1 ,0 )

        monkeypatch .setattr ("llmsh.summarize.summarize_file",fake_summarize_file )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            cmd =f"/summarize {test_file } find warnings"
            for ch in cmd :
                await pilot .press (ch )
            await pilot .press ("enter")

            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()
            await app .workers .wait_for_complete ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "summary: found 3 warnings"in combined .lower ()

    @pytest .mark .anyio 
    async def test_summarize_shows_truncation_warning (self ,tmp_path ,monkeypatch ):
        """/summarize reports truncation count when responses were truncated."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        test_file =tmp_path /"data.txt"
        test_file .write_text ("some data")

        async def fake_summarize_file (**kwargs ):
            return ("Result here",2 ,3 )

        monkeypatch .setattr ("llmsh.summarize.summarize_file",fake_summarize_file )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            cmd =f"/summarize {test_file } analyze"
            for ch in cmd :
                await pilot .press (ch )
            await pilot .press ("enter")

            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()
            await app .workers .wait_for_complete ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "truncated"in combined .lower ()
            assert "result here"in combined .lower ()


class TestNoConfigMessage :
    @pytest .mark .anyio 
    async def test_message_with_no_core_shows_helpful_error (self ):
        """Sending a normal message with no AppCore shows a helpful system message."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        app =LlmshApp (None )
        async with app .run_test ()as pilot :

            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )

            app .screen ._active_flow =None 
            for ch in "hello world":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "endpoint"in combined .lower ()or "configuration"in combined .lower ()
