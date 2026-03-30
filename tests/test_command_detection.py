"""Tests for slash command detection via LLM tool."""

import pytest 

from llmsh .app import AppCore ,CommandSuggestionEvent 
from llmsh .config import AppConfig 
from llmsh .models import (
EndpointConfig ,
ModelCapabilities ,
ProfileConfig ,
ToolCall ,
)
from llmsh .providers .base import (
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
ToolCallEvent ,
)
from llmsh .ui .slash import build_command_detection_tool ,get_command_names 
from tests .conftest import StubProvider 






def _make_core ()->AppCore :
    return AppCore (config =_make_config (),provider =StubProvider ())


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


def _make_config (tool_calling :bool =False )->AppConfig :
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
    capabilities =ModelCapabilities (tool_calling =tool_calling ),
    )
    return AppConfig (
    current_profile ="default",
    endpoints ={"local":endpoint },
    profiles ={"default":profile },
    )


class TestBuildCommandDetectionTool :
    def test_includes_all_registered_command_names (self ):
        tool =build_command_detection_tool ()
        expected_names =get_command_names ()
        for name in expected_names :
            assert f"/{name }"in tool .description 

    def test_tool_name_is_suggest_slash_command (self ):
        tool =build_command_detection_tool ()
        assert tool .name =="suggest_slash_command"

    def test_has_command_parameter (self ):
        tool =build_command_detection_tool ()
        param_names =[p .name for p in tool .parameters ]
        assert "command"in param_names 

    def test_has_reasoning_parameter_optional (self ):
        tool =build_command_detection_tool ()
        reasoning =[p for p in tool .parameters if p .name =="reasoning"][0 ]
        assert reasoning .required is False 


class TestGetCommandNames :
    def test_returns_known_commands (self ):
        names =get_command_names ()
        assert "help"in names 
        assert "profile"in names 
        assert "doctor"in names 
        assert "retry"in names 


class TestToolInjection :
    @pytest .mark .anyio 
    async def test_tool_included_when_tool_calling_true (self ):
        stub =StubProvider (events =[
        ResponseStarted (),
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ])
        core =AppCore (config =_make_config (tool_calling =True ),provider =stub )

        async for _ in core .send_message ("hello"):
            pass 

        request =stub .requests [0 ]
        assert request .tools is not None 
        assert len (request .tools )==1 
        assert request .tools [0 ].name =="suggest_slash_command"

    @pytest .mark .anyio 
    async def test_tool_not_included_when_tool_calling_false (self ):
        stub =StubProvider (events =[
        ResponseStarted (),
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ])
        core =AppCore (config =_make_config (tool_calling =False ),provider =stub )

        async for _ in core .send_message ("hello"):
            pass 

        request =stub .requests [0 ]
        assert request .tools is None 


class TestToolCallInterception :
    @pytest .mark .anyio 
    async def test_suggest_slash_command_yields_command_suggestion_event (self ):
        tool_call =ToolCall (
        id ="tc1",
        name ="suggest_slash_command",
        arguments ={
        "command":"/profile use qwen",
        "reasoning":"looks like a command",
        },
        )
        stub =StubProvider (events =[
        ResponseStarted (),
        ToolCallEvent (tool_call =tool_call ),
        ResponseCompleted (content =""),
        ])
        core =AppCore (config =_make_config (tool_calling =True ),provider =stub )

        events =[]
        async for event in core .send_message ("profile use qwen"):
            events .append (event )

        suggestions =[e for e in events if isinstance (e ,CommandSuggestionEvent )]
        assert len (suggestions )==1 
        assert suggestions [0 ].command =="/profile use qwen"
        assert suggestions [0 ].reasoning =="looks like a command"

    @pytest .mark .anyio 
    async def test_suggest_slash_command_not_yielded_as_tool_call_event (self ):
        tool_call =ToolCall (
        id ="tc1",
        name ="suggest_slash_command",
        arguments ={"command":"/help"},
        )
        stub =StubProvider (events =[
        ResponseStarted (),
        ToolCallEvent (tool_call =tool_call ),
        ResponseCompleted (content =""),
        ])
        core =AppCore (config =_make_config (tool_calling =True ),provider =stub )

        events =[]
        async for event in core .send_message ("help"):
            events .append (event )

        tool_call_events =[e for e in events if isinstance (e ,ToolCallEvent )]
        assert len (tool_call_events )==0 

    @pytest .mark .anyio 
    async def test_unknown_tool_call_is_ignored (self ):
        tool_call =ToolCall (
        id ="tc1",
        name ="unknown_tool",
        arguments ={"foo":"bar"},
        )
        stub =StubProvider (events =[
        ResponseStarted (),
        ToolCallEvent (tool_call =tool_call ),
        TextDelta (text ="response"),
        ResponseCompleted (content ="response"),
        ])
        core =AppCore (config =_make_config (tool_calling =True ),provider =stub )

        events =[]
        async for event in core .send_message ("hello"):
            events .append (event )


        suggestions =[e for e in events if isinstance (e ,CommandSuggestionEvent )]
        assert len (suggestions )==0 

        text_deltas =[e for e in events if isinstance (e ,TextDelta )]
        assert len (text_deltas )==1 

    @pytest .mark .anyio 
    async def test_text_only_response_no_suggestion (self ):
        stub =StubProvider (events =[
        ResponseStarted (),
        TextDelta (text ="hello world"),
        ResponseCompleted (content ="hello world"),
        ])
        core =AppCore (config =_make_config (tool_calling =True ),provider =stub )

        events =[]
        async for event in core .send_message ("hello"):
            events .append (event )

        suggestions =[e for e in events if isinstance (e ,CommandSuggestionEvent )]
        assert len (suggestions )==0 
        text_deltas =[e for e in events if isinstance (e ,TextDelta )]
        assert len (text_deltas )==1 

    @pytest .mark .anyio 
    async def test_malformed_tool_call_arguments_handled_gracefully (self ):
        tool_call =ToolCall (
        id ="tc1",
        name ="suggest_slash_command",
        arguments ={},
        )
        stub =StubProvider (events =[
        ResponseStarted (),
        ToolCallEvent (tool_call =tool_call ),
        ResponseCompleted (content =""),
        ])
        core =AppCore (config =_make_config (tool_calling =True ),provider =stub )

        events =[]
        async for event in core .send_message ("hello"):
            events .append (event )


        suggestions =[e for e in events if isinstance (e ,CommandSuggestionEvent )]
        assert len (suggestions )==0 

    @pytest .mark .anyio 
    async def test_invalid_command_name_not_suggested (self ):
        tool_call =ToolCall (
        id ="tc1",
        name ="suggest_slash_command",
        arguments ={"command":"/nonexistent foo"},
        )
        stub =StubProvider (events =[
        ResponseStarted (),
        ToolCallEvent (tool_call =tool_call ),
        ResponseCompleted (content =""),
        ])
        core =AppCore (config =_make_config (tool_calling =True ),provider =stub )

        events =[]
        async for event in core .send_message ("nonexistent foo"):
            events .append (event )

        suggestions =[e for e in events if isinstance (e ,CommandSuggestionEvent )]
        assert len (suggestions )==0 

    @pytest .mark .anyio 
    async def test_command_without_leading_slash_not_suggested (self ):
        tool_call =ToolCall (
        id ="tc1",
        name ="suggest_slash_command",
        arguments ={"command":"profile use qwen"},
        )
        stub =StubProvider (events =[
        ResponseStarted (),
        ToolCallEvent (tool_call =tool_call ),
        ResponseCompleted (content =""),
        ])
        core =AppCore (config =_make_config (tool_calling =True ),provider =stub )

        events =[]
        async for event in core .send_message ("profile use qwen"):
            events .append (event )

        suggestions =[e for e in events if isinstance (e ,CommandSuggestionEvent )]
        assert len (suggestions )==0 







class TestCommandConfirmFlow :
    @pytest .mark .anyio 
    async def test_command_confirm_flow_y_executes_command (self ):
        """Typing 'y' runs the suggested command (e.g. /help shows help text)."""
        from llmsh .ui .flows .command_confirm import CommandConfirmFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await app .screen .start_flow (CommandConfirmFlow ("/help"))
            await pilot .pause ()

            await _type_and_submit (pilot ,"y")

            text =_transcript_text (app )
            assert "commands"in text .lower ()
            assert app .screen ._active_flow is None 

    @pytest .mark .anyio 
    async def test_command_confirm_flow_n_cancels (self ):
        """Typing 'n' cancels the flow without running the command."""
        from llmsh .ui .flows .command_confirm import CommandConfirmFlow 
        from llmsh .ui .main import LlmshApp 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            await app .screen .start_flow (CommandConfirmFlow ("/help"))
            await pilot .pause ()

            await _type_and_submit (pilot ,"n")

            text =_transcript_text (app )
            assert "cancelled"in text .lower ()
            assert app .screen ._active_flow is None 
