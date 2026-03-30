"""Tests for the session CLI command group."""

from __future__ import annotations 

from datetime import UTC ,datetime 
from unittest .mock import MagicMock 

from typer .testing import CliRunner 

from llmsh .cli import app 
from llmsh .errors import SessionNotFoundError 
from llmsh .sessions .store import SessionMeta 

runner =CliRunner ()

NOW =datetime (2025 ,6 ,15 ,10 ,30 ,0 ,tzinfo =UTC )


def _make_store (sessions :list [SessionMeta ]|None =None )->MagicMock :
    store =MagicMock ()
    store .list .return_value =sessions or []
    return store 


class TestSessionList :
    def test_lists_sessions_with_id_timestamp_preview (self ,monkeypatch ):
        sessions =[
        SessionMeta (
        id ="abc12345-0000-0000-0000-000000000000",
        title ="Hello world chat",
        profile ="default",
        model ="gpt-4",
        updated_at =NOW ,
        ),
        ]
        store =_make_store (sessions )
        monkeypatch .setattr (
        "llmsh.commands.session.SessionStore",lambda d :store ,
        )

        result =runner .invoke (app ,["session","list"])

        assert result .exit_code ==0 
        assert "abc12345"in result .stdout 
        assert "Hello world chat"in result .stdout 
        assert "2025-06-15"in result .stdout 

    def test_empty_list_prints_message (self ,monkeypatch ):
        store =_make_store ([])
        monkeypatch .setattr (
        "llmsh.commands.session.SessionStore",lambda d :store ,
        )

        result =runner .invoke (app ,["session","list"])

        assert result .exit_code ==0 
        assert "No sessions"in result .stdout 

    def test_multiple_sessions_all_shown (self ,monkeypatch ):
        sessions =[
        SessionMeta (
        id ="aaa00000-0000-0000-0000-000000000000",
        title ="First",
        profile ="default",
        model ="gpt-4",
        updated_at =NOW ,
        ),
        SessionMeta (
        id ="bbb00000-0000-0000-0000-000000000000",
        title ="Second",
        profile ="default",
        model ="gpt-4",
        updated_at =NOW ,
        ),
        ]
        store =_make_store (sessions )
        monkeypatch .setattr (
        "llmsh.commands.session.SessionStore",lambda d :store ,
        )

        result =runner .invoke (app ,["session","list"])

        assert result .exit_code ==0 
        assert "First"in result .stdout 
        assert "Second"in result .stdout 


class TestSessionShow :
    def test_prints_session_transcript (self ,monkeypatch ):
        from llmsh .models import ChatMessage ,SessionRecord 

        session =SessionRecord (
        id ="abc12345-0000-0000-0000-000000000000",
        title ="Test session",
        profile ="default",
        model ="gpt-4",
        created_at =NOW ,
        updated_at =NOW ,
        messages =[
        ChatMessage (role ="user",content ="Hello"),
        ChatMessage (role ="assistant",content ="Hi there!"),
        ],
        usage =[],
        )
        store =MagicMock ()
        store .load .return_value =session 
        monkeypatch .setattr (
        "llmsh.commands.session.SessionStore",lambda d :store ,
        )

        result =runner .invoke (app ,["session","show","abc12345"])

        assert result .exit_code ==0 
        assert "user"in result .stdout .lower ()or "User"in result .stdout 
        assert "Hello"in result .stdout 
        assert "Hi there!"in result .stdout 
        store .load .assert_called_once_with ("abc12345")

    def test_not_found_exits_nonzero (self ,monkeypatch ):
        store =MagicMock ()
        store .load .side_effect =SessionNotFoundError ("nope")
        monkeypatch .setattr (
        "llmsh.commands.session.SessionStore",lambda d :store ,
        )

        result =runner .invoke (app ,["session","show","nonexistent"])

        assert result .exit_code !=0 


class TestSessionDelete :
    def test_deletes_session_exits_zero (self ,monkeypatch ):
        store =MagicMock ()
        monkeypatch .setattr (
        "llmsh.commands.session.SessionStore",lambda d :store ,
        )

        result =runner .invoke (app ,["session","delete","abc12345"])

        assert result .exit_code ==0 
        store .delete .assert_called_once_with ("abc12345")
        assert "Deleted"in result .stdout or "deleted"in result .stdout .lower ()

    def test_not_found_exits_nonzero (self ,monkeypatch ):
        store =MagicMock ()
        store .delete .side_effect =SessionNotFoundError ("nope")
        monkeypatch .setattr (
        "llmsh.commands.session.SessionStore",lambda d :store ,
        )

        result =runner .invoke (app ,["session","delete","nonexistent"])

        assert result .exit_code !=0 
