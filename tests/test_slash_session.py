"""Tests for /session slash command and its sub-commands."""

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


def _make_core (profile_name :str ="default",model :str ="test-model")->AppCore :
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


def _patch_sessions_dir (monkeypatch ,path ):
    import llmsh .ui .slash as slash_mod 
    monkeypatch .setattr (slash_mod ,"sessions_dir",lambda :path )







class TestSessionList :
    @pytest .mark .anyio 
    async def test_session_list_shows_sessions (self ,tmp_path ,monkeypatch ):
        """/session list shows recent sessions in transcript."""
        from llmsh .sessions .store import SessionStore 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        store =SessionStore (sessions_path )
        rec =store .create (profile ="default",model ="test-model")
        store .save (rec )

        _patch_sessions_dir (monkeypatch ,sessions_path )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/session list":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "session"in combined .lower ()







class TestSessionSave :
    @pytest .mark .anyio 
    async def test_session_save_shows_confirmation (self ,tmp_path ,monkeypatch ):
        """/session save saves the session and shows a confirmation."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        _patch_sessions_dir (monkeypatch ,sessions_path )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/session save":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "saved"in combined .lower ()or "session"in combined .lower ()


        assert any (sessions_path .glob ("*.json"))







class TestSessionLoad :
    @pytest .mark .anyio 
    async def test_session_load_restores_messages (self ,tmp_path ,monkeypatch ):
        """/session load <id> loads the session into the transcript."""
        from llmsh .models import ChatMessage 
        from llmsh .sessions .store import SessionStore 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        store =SessionStore (sessions_path )
        rec =store .create (profile ="default",model ="test-model")
        rec .messages =[ChatMessage (role ="user",content ="hello from loaded session")]
        store .save (rec )

        _patch_sessions_dir (monkeypatch ,sessions_path )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            cmd =f"/session load {rec .id [:8 ]}"
            for ch in cmd :
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "hello from loaded session"in combined 







class TestSessionDelete :
    @pytest .mark .anyio 
    async def test_session_delete_valid_removes_file (self ,tmp_path ,monkeypatch ):
        """/session delete <id> removes the session file and shows confirmation."""
        from llmsh .sessions .store import SessionStore 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        store =SessionStore (sessions_path )
        rec =store .create (profile ="default",model ="test-model")
        store .save (rec )
        session_file =sessions_path /f"{rec .id }.json"
        assert session_file .exists ()

        _patch_sessions_dir (monkeypatch ,sessions_path )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            cmd =f"/session delete {rec .id }"
            for ch in cmd :
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "deleted"in combined .lower ()or "session"in combined .lower ()

        assert not session_file .exists ()

    @pytest .mark .anyio 
    async def test_session_delete_unknown_shows_error (self ,tmp_path ,monkeypatch ):
        """/session delete <unknown-id> shows an error message."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        sessions_path .mkdir ()
        _patch_sessions_dir (monkeypatch ,sessions_path )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/session delete nonexistent-id":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "not found"in combined .lower ()or "error"in combined .lower ()

    @pytest .mark .anyio 
    async def test_session_delete_no_id_shows_usage (self ,tmp_path ,monkeypatch ):
        """/session delete with no id shows a usage hint."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        sessions_path .mkdir ()
        _patch_sessions_dir (monkeypatch ,sessions_path )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/session delete":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "usage"in combined .lower ()or "delete"in combined .lower ()







class TestSessionNoArgs :
    @pytest .mark .anyio 
    async def test_session_no_args_defaults_to_list (self ,tmp_path ,monkeypatch ):
        """/session with no args behaves like /session list."""
        from llmsh .sessions .store import SessionStore 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        store =SessionStore (sessions_path )
        rec =store .create (profile ="default",model ="test-model")
        store .save (rec )

        _patch_sessions_dir (monkeypatch ,sessions_path )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/session":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "session"in combined .lower ()







class TestSaveAlias :
    @pytest .mark .anyio 
    async def test_save_alias_still_works (self ,tmp_path ,monkeypatch ):
        """/save continues to work as an alias for /session save."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        _patch_sessions_dir (monkeypatch ,sessions_path )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
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

        assert any (sessions_path .glob ("*.json"))







class TestLoadAlias :
    @pytest .mark .anyio 
    async def test_load_alias_still_works (self ,tmp_path ,monkeypatch ):
        """/load continues to work as an alias for /session load."""
        from llmsh .sessions .store import SessionStore 
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        sessions_path =tmp_path /"sessions"
        store =SessionStore (sessions_path )
        rec =store .create (profile ="default",model ="test-model")
        store .save (rec )

        _patch_sessions_dir (monkeypatch ,sessions_path )

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/load":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            pane =app .screen .query_one (TranscriptPane )
            combined =" ".join (str (w .content )for w in pane .query (MessageWidget ))
            assert "session"in combined .lower ()
