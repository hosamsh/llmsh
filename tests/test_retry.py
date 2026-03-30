"""Tests for auto-save and /retry on connection failure."""

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
ErrorEvent ,
ProviderEvent ,
ProviderRequest ,
ProviderResult ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
)






class StubProvider (BaseProvider ):
    """Provider that yields configurable events and tracks requests."""

    def __init__ (self ,events :list [ProviderEvent ]|None =None )->None :
        self .events :list [ProviderEvent ]=events or [
        ResponseStarted (),
        TextDelta (text ="hello"),
        ResponseCompleted (content ="hello"),
        ]
        self .requests :list [ProviderRequest ]=[]

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        self .requests .append (request )
        for event in self .events :
            yield event 

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        self .requests .append (request )
        return ProviderResult (content ="hello")

    async def doctor (self )->DoctorReport :
        return DoctorReport (
        checks =[DoctorCheck (name ="connectivity",passed =True ,message ="OK")]
        )


class ErrorProvider (BaseProvider ):
    """Provider that raises an exception during streaming."""

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        raise ConnectionError ("server down")
        yield 

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        raise ConnectionError ("server down")

    async def doctor (self )->DoctorReport :
        return DoctorReport (checks =[])


def _make_config ()->AppConfig :
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
    return AppConfig (
    current_profile ="default",
    endpoints ={"local":endpoint },
    profiles ={"default":profile },
    )


def _make_core (
provider :BaseProvider |None =None ,
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
    return AppCore (config =config ,provider =provider or StubProvider ())







class TestErrorPopsUserMessage :
    @pytest .mark .anyio 
    async def test_exception_in_provider_pops_user_message (self ):
        """When the provider raises, the user message is removed from history."""
        core =_make_core (provider =ErrorProvider ())

        events =[]
        async for event in core .send_message ("hello"):
            events .append (event )


        assert any (isinstance (e ,ErrorEvent )for e in events )

        assert len (core ._messages )==0 

    @pytest .mark .anyio 
    async def test_successful_response_keeps_both_messages (self ):
        """After a successful exchange, both user and assistant are in history."""
        core =_make_core ()

        async for _ in core .send_message ("hello"):
            pass 

        assert len (core ._messages )==2 
        assert core ._messages [0 ].role =="user"
        assert core ._messages [1 ].role =="assistant"

    @pytest .mark .anyio 
    async def test_error_then_retry_has_clean_history (self ):
        """After an error pops the user msg, a retry re-adds it cleanly."""
        stub =StubProvider ()
        core =_make_core (provider =ErrorProvider ())


        async for _ in core .send_message ("hello"):
            pass 
        assert len (core ._messages )==0 


        core ._provider =stub 
        async for _ in core .send_message ("hello"):
            pass 

        roles =[m .role for m in core ._messages ]
        assert roles ==["user","assistant"]







class TestAutoSave :
    @pytest .mark .anyio 
    async def test_auto_save_creates_session_file (self ,tmp_path ,monkeypatch ):
        """After a successful streaming response, a session file is auto-saved."""
        import llmsh .ui .screens as screens_mod 

        monkeypatch .setattr (screens_mod ,"sessions_dir",lambda :tmp_path /"sessions")

        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "hello":
                await pilot .press (ch )
            await pilot .press ("enter")

            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()


            sessions_path =tmp_path /"sessions"
            json_files =list (sessions_path .glob ("*.json"))

            data_files =[f for f in json_files if f .stem !="index"]
            assert len (data_files )>=1 

    @pytest .mark .anyio 
    async def test_auto_save_updates_existing_session (self ,tmp_path ,monkeypatch ):
        """A second message updates the same session, not a new one."""
        import llmsh .ui .screens as screens_mod 

        monkeypatch .setattr (screens_mod ,"sessions_dir",lambda :tmp_path /"sessions")

        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )


            await pilot .click (chat_input )
            for ch in "first":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()


            await pilot .click (chat_input )
            for ch in "second":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()

            sessions_path =tmp_path /"sessions"
            json_files =list (sessions_path .glob ("*.json"))
            data_files =[f for f in json_files if f .stem !="index"]

            assert len (data_files )==1 







class TestRetryCommand :
    @pytest .mark .anyio 
    async def test_retry_no_messages_shows_hint (self ):
        """/retry with no previous messages shows 'Nothing to retry.'"""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/retry":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "nothing to retry"in combined .lower ()

    @pytest .mark .anyio 
    async def test_retry_shows_retrying_message (self ):
        """/retry shows 'Retrying...' system message."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )


            await pilot .click (chat_input )
            for ch in "hello":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()


            await pilot .click (chat_input )
            for ch in "/retry":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "retrying"in combined .lower ()

    @pytest .mark .anyio 
    async def test_retry_resends_last_user_message (self ):
        """/retry sends the last user message to the provider again."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        stub =StubProvider ()
        core =_make_core (provider =stub )
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )


            await pilot .click (chat_input )
            for ch in "hello":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()


            await pilot .click (chat_input )
            for ch in "/retry":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()


            assert len (stub .requests )>=2 

            first_user_msgs =[
            m .content for m in stub .requests [0 ].messages if m .role =="user"
            ]
            second_user_msgs =[
            m .content for m in stub .requests [1 ].messages if m .role =="user"
            ]
            assert "hello"in first_user_msgs 
            assert "hello"in second_user_msgs 


class TestHelpIncludesRetry :
    @pytest .mark .anyio 
    async def test_help_mentions_retry (self ):
        """/help includes /retry in output."""
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
            assert "/retry"in combined 
