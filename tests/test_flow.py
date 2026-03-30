"""Tests for the interactive flow framework."""

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
from llmsh .ui .flow import InteractiveFlow 
from llmsh .ui .slash import show_system_message 






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


class TwoStepTestFlow (InteractiveFlow ):
    """A concrete flow for testing: collects two inputs then completes."""

    def __init__ (self )->None :
        self .collected :list [str ]=[]
        self .started =False 

    async def start (self ,screen :MainScreen )->None :
        self .started =True 
        await show_system_message (screen ,"Step 1: What is your name?")

    async def handle_input (self ,text :str ,screen :MainScreen )->bool :
        self .collected .append (text )
        if len (self .collected )==1 :
            await show_system_message (screen ,"Step 2: Confirm? (y/n)")
            return False 
        return True 

    @property 
    def placeholder (self )->str :
        return "Enter your name..."

    async def cancel (self ,screen :MainScreen )->None :
        await show_system_message (screen ,"Test flow cancelled.")







class TestStartFlowChangesPlaceholder :
    @pytest .mark .anyio 
    async def test_start_flow_changes_placeholder (self ):
        """Starting a flow changes the ChatInput placeholder."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            chat_input =app .screen .query_one (ChatInput )
            assert chat_input .placeholder =="Type a message..."

            flow =TwoStepTestFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            assert chat_input .placeholder =="Enter your name..."


class TestInputDuringFlowRoutesToHandleInput :
    @pytest .mark .anyio 
    async def test_input_during_flow_routes_to_handle_input (self ):
        """User input during an active flow is routed to the flow's handle_input."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =TwoStepTestFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "Alice":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            assert flow .collected ==["Alice"]


class TestHandleInputReturningTrueEndsFlow :
    @pytest .mark .anyio 
    async def test_handle_input_returning_true_ends_flow (self ):
        """When handle_input returns True, the flow ends and _active_flow is None."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =TwoStepTestFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            chat_input =app .screen .query_one (ChatInput )


            await pilot .click (chat_input )
            for ch in "Alice":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()


            await pilot .click (chat_input )
            for ch in "y":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            assert app .screen ._active_flow is None 
            assert flow .collected ==["Alice","y"]


class TestHandleInputReturningFalseKeepsFlow :
    @pytest .mark .anyio 
    async def test_handle_input_returning_false_keeps_flow (self ):
        """After step 1 (returning False), the flow is still active."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =TwoStepTestFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "Alice":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            assert app .screen ._active_flow is flow 


class TestEscapeCancelsFlow :
    @pytest .mark .anyio 
    async def test_escape_cancels_flow (self ):
        """Pressing Escape during an active flow cancels it."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =TwoStepTestFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            await pilot .press ("escape")
            await pilot .pause ()
            await pilot .pause ()

            assert app .screen ._active_flow is None 


            pane =app .screen .query_one (TranscriptPane )
            widgets =pane .query (MessageWidget )
            combined =" ".join (str (w .content )for w in widgets )
            assert "cancelled"in combined .lower ()


class TestCancelCommandCancelsFlow :
    @pytest .mark .anyio 
    async def test_cancel_command_cancels_flow (self ):
        """/cancel during a flow cancels it."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =TwoStepTestFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/cancel":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            assert app .screen ._active_flow is None 

            pane =app .screen .query_one (TranscriptPane )
            widgets =pane .query (MessageWidget )
            combined =" ".join (str (w .content )for w in widgets )
            assert "cancelled"in combined .lower ()


class TestCancelWithNoFlow :
    @pytest .mark .anyio 
    async def test_cancel_with_no_flow (self ):
        """/cancel when no flow is active shows 'Nothing to cancel'."""
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
            widgets =pane .query (MessageWidget )
            combined =" ".join (str (w .content )for w in widgets )
            assert "nothing to cancel"in combined .lower ()


class TestSlashCommandsWorkDuringFlow :
    @pytest .mark .anyio 
    async def test_slash_commands_work_during_flow (self ):
        """/help during a flow still shows help, not routed to flow."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput ,MessageWidget ,TranscriptPane 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =TwoStepTestFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "/help":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()


            assert flow .collected ==[]


            assert app .screen ._active_flow is flow 


            pane =app .screen .query_one (TranscriptPane )
            widgets =pane .query (MessageWidget )
            combined =" ".join (str (w .content )for w in widgets )
            assert "commands"in combined .lower ()or "available"in combined .lower ()


class TestNormalMessageWithoutFlow :
    @pytest .mark .anyio 
    async def test_normal_message_without_flow (self ):
        """Without a flow active, messages go to the LLM (existing behavior)."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            assert app .screen ._active_flow is None 

            chat_input =app .screen .query_one (ChatInput )
            await pilot .click (chat_input )
            for ch in "hello":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()
            await pilot .pause ()




            assert app .screen ._active_flow is None 


class TestEndFlowRestoresPlaceholder :
    @pytest .mark .anyio 
    async def test_end_flow_restores_placeholder (self ):
        """After a flow completes, the placeholder returns to the default."""
        from llmsh .ui .main import LlmshApp 
        from llmsh .ui .widgets import ChatInput 

        core =_make_core ()
        app =LlmshApp (core )
        async with app .run_test ()as pilot :
            flow =TwoStepTestFlow ()
            await app .screen .start_flow (flow )
            await pilot .pause ()

            chat_input =app .screen .query_one (ChatInput )
            assert chat_input .placeholder =="Enter your name..."


            await pilot .click (chat_input )
            for ch in "Alice":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            await pilot .click (chat_input )
            for ch in "y":
                await pilot .press (ch )
            await pilot .press ("enter")
            await pilot .pause ()
            await pilot .pause ()

            assert chat_input .placeholder =="Type a message..."
