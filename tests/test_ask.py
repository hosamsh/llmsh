"""Tests for the ask command."""

import json 

from typer .testing import CliRunner 

from llmsh .app import AppCore 
from llmsh .cli import app 
from llmsh .config import AppConfig 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig ,UsageInfo 
from llmsh .providers .base import (
ErrorEvent ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
TokenUsageEvent ,
)
from tests .conftest import StubProvider 


def _make_config (profiles :dict |None =None )->AppConfig :
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
    profiles =profiles or {"default":profile },
    )


def _make_core (events =None ,config =None )->AppCore :
    return AppCore (
    config =config or _make_config (),
    provider =StubProvider (events =events ),
    )


runner =CliRunner ()


class TestAskPositionalPrompt :
    def test_prints_response_to_stdout_exits_zero (self ,monkeypatch ):
        core =_make_core ()
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask","hello"])

        assert result .exit_code ==0 
        assert "hello world"in result .stdout 

    def test_sends_prompt_to_provider (self ,monkeypatch ):
        stub =StubProvider ()
        core =AppCore (config =_make_config (),provider =stub )
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        runner .invoke (app ,["ask","my prompt"])

        assert stub .requests [0 ].messages [0 ].content =="my prompt"


class TestAskStdin :
    def test_reads_prompt_from_stdin (self ,monkeypatch ):
        core =_make_core ()
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask","--stdin"],input ="hello from stdin\n")

        assert result .exit_code ==0 
        assert "hello world"in result .stdout 


class TestAskJsonOutput :
    def test_json_output_has_required_fields (self ,monkeypatch ):
        core =_make_core ()
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask","--json","hello"])

        assert result .exit_code ==0 
        data =json .loads (result .stdout )
        assert "response"in data 
        assert "model"in data 
        assert data ["response"]=="hello world"
        assert data ["model"]=="test-model"

    def test_json_output_has_profile (self ,monkeypatch ):
        core =_make_core ()
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask","--json","hello"])

        data =json .loads (result .stdout )
        assert "profile"in data 

    def test_json_usage_is_null_when_no_token_event (self ,monkeypatch ):
        core =_make_core ()
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask","--json","hello"])

        data =json .loads (result .stdout )
        assert data ["usage"]is None 

    def test_json_usage_populated_when_token_event_received (self ,monkeypatch ):
        events =[
        ResponseStarted (),
        TextDelta (text ="hello world"),
        ResponseCompleted (content ="hello world"),
        TokenUsageEvent (
        usage =UsageInfo (input_tokens =5 ,output_tokens =10 ,total_tokens =15 )
        ),
        ]
        core =_make_core (events =events )
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask","--json","hello"])

        data =json .loads (result .stdout )
        assert data ["usage"]is not None 
        assert data ["usage"]["input_tokens"]==5 
        assert data ["usage"]["output_tokens"]==10 


class TestAskProfileOverride :
    def test_unknown_profile_exits_one_with_stderr (self ,monkeypatch ):
        from llmsh .errors import ProfileNotFoundError 

        def failing_factory (**kw ):
            raise ProfileNotFoundError ("Profile not found: nonexistent")

        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",failing_factory )

        result =runner .invoke (app ,["ask","--profile","nonexistent","hello"])

        assert result .exit_code ==1 
        assert "nonexistent"in result .stderr 

    def test_profile_flag_passed_to_factory (self ,monkeypatch ):
        received ={}

        def fake_make_core (**kw ):
            received .update (kw )
            return _make_core ()

        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",fake_make_core )

        runner .invoke (app ,["ask","--profile","myprofile","hello"])

        assert received .get ("profile")=="myprofile"



class TestAskNoPrompt :
    def test_no_prompt_no_stdin_exits_one (self ,monkeypatch ):
        core =_make_core ()
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask"])

        assert result .exit_code ==1 

    def test_no_prompt_no_stdin_stderr_contains_hint (self ,monkeypatch ):
        core =_make_core ()
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask"])

        assert result .stderr !=""


class TestAskSystemPrompt :
    def test_system_flag_sends_system_message_first (self ,monkeypatch ):
        stub =StubProvider ()

        def factory (**kw ):
            return AppCore (
            config =_make_config (),
            provider =stub ,
            system_prompt =kw .get ("system_prompt"),
            )

        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",factory )

        runner .invoke (app ,["ask","--system","You are a pirate","hello"])

        messages =stub .requests [0 ].messages 
        assert messages [0 ].role =="system"
        assert messages [0 ].content =="You are a pirate"
        assert messages [1 ].role =="user"
        assert messages [1 ].content =="hello"

    def test_no_system_flag_sends_no_system_message (self ,monkeypatch ):
        stub =StubProvider ()
        core =AppCore (config =_make_config (),provider =stub )
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        runner .invoke (app ,["ask","hello"])

        messages =stub .requests [0 ].messages 
        assert all (m .role !="system"for m in messages )

    def test_system_with_json_produces_valid_json (self ,monkeypatch ):
        core =_make_core ()
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask","--system","Be brief","--json","hello"])

        assert result .exit_code ==0 
        data =json .loads (result .stdout )
        assert data ["response"]=="hello world"

    def test_system_with_stdin (self ,monkeypatch ):
        stub =StubProvider ()

        def factory (**kw ):
            return AppCore (
            config =_make_config (),
            provider =stub ,
            system_prompt =kw .get ("system_prompt"),
            )

        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",factory )

        result =runner .invoke (
        app ,["ask","--system","You are helpful","--stdin"],input ="hi\n"
        )

        assert result .exit_code ==0 
        messages =stub .requests [0 ].messages 
        assert messages [0 ].role =="system"
        assert messages [0 ].content =="You are helpful"

    def test_system_flag_passed_to_factory (self ,monkeypatch ):
        received ={}

        def fake_make_core (**kw ):
            received .update (kw )
            return _make_core ()

        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",fake_make_core )

        runner .invoke (app ,["ask","--system","test prompt","hello"])

        assert received .get ("system_prompt")=="test prompt"


class TestAskErrorEvent :
    def test_provider_error_event_goes_to_stderr_exit_one (self ,monkeypatch ):
        events =[ErrorEvent (message ="provider exploded")]
        core =_make_core (events =events )
        monkeypatch .setattr ("llmsh.commands.ask._make_app_core",lambda **kw :core )

        result =runner .invoke (app ,["ask","hello"])

        assert result .exit_code ==1 
        assert "provider exploded"in result .stderr 
