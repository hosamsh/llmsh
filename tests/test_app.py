"""Tests for AppCore orchestration layer."""

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .errors import ProfileNotFoundError 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
from llmsh .providers .base import (
CancelledEvent ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
)
from tests .conftest import StubProvider 


def _make_config ()->AppConfig :
    """Build a minimal in-memory AppConfig for testing."""
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


class TestSendMessage :
    @pytest .mark .anyio 
    async def test_yields_two_text_deltas (self ):
        stub =StubProvider (events =[
        TextDelta (text ="hello "),
        TextDelta (text ="world"),
        ResponseCompleted (content ="hello world"),
        ])
        core =AppCore (config =_make_config (),provider =stub )

        events =[]
        async for event in core .send_message ("Hi"):
            events .append (event )

        text_deltas =[e for e in events if isinstance (e ,TextDelta )]
        assert len (text_deltas )==2 
        assert text_deltas [0 ].text =="hello "
        assert text_deltas [1 ].text =="world"

    @pytest .mark .anyio 
    async def test_accumulates_messages_for_multi_turn (self ):
        stub =StubProvider (events =[
        TextDelta (text ="reply"),
        ResponseCompleted (content ="reply"),
        ])
        core =AppCore (config =_make_config (),provider =stub )


        async for _ in core .send_message ("First"):
            pass 

        async for _ in core .send_message ("Second"):
            pass 


        last_request =stub .requests [-1 ]
        roles =[m .role for m in last_request .messages ]
        assert roles ==["user","assistant","user"]

    @pytest .mark .anyio 
    async def test_sets_is_streaming_flag (self ):
        stub =StubProvider (events =[
        TextDelta (text ="x"),
        ResponseCompleted (content ="x"),
        ])
        core =AppCore (config =_make_config (),provider =stub )

        assert core .is_streaming is False 
        async for _ in core .send_message ("test"):
            assert core .is_streaming is True 
        assert core .is_streaming is False 


class TestMaxTokensPropagation :
    @pytest .mark .anyio 
    async def test_default_max_tokens_is_1024_when_profile_has_none (self ):
        stub =StubProvider (events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ])
        core =AppCore (config =_make_config (),provider =stub )

        async for _ in core .send_message ("test"):
            pass 

        assert stub .requests [-1 ].max_tokens ==1024 

    @pytest .mark .anyio 
    async def test_profile_max_tokens_overrides_default (self ):
        config =_make_config ()
        config .profiles ["default"].max_tokens =2048 
        stub =StubProvider (events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ])
        core =AppCore (config =config ,provider =stub )

        async for _ in core .send_message ("test"):
            pass 

        assert stub .requests [-1 ].max_tokens ==2048 


class TestCancel :
    @pytest .mark .anyio 
    async def test_cancel_stops_stream_and_yields_cancelled_event (self ):
        """Calling cancel() during streaming should stop iteration
        and the stream should contain a CancelledEvent."""
        events_to_yield =[
        ResponseStarted (),
        TextDelta (text ="a"),
        TextDelta (text ="b"),
        TextDelta (text ="c"),
        ResponseCompleted (content ="abc"),
        ]
        stub =StubProvider (events =events_to_yield )
        core =AppCore (config =_make_config (),provider =stub )

        collected :list =[]

        async def consume ():
            async for event in core .send_message ("test"):
                collected .append (event )
                if isinstance (event ,TextDelta )and event .text =="a":
                    core .cancel ()

        await consume ()


        has_cancelled =any (isinstance (e ,CancelledEvent )for e in collected )
        assert has_cancelled 

        text_deltas =[e for e in collected if isinstance (e ,TextDelta )]
        assert len (text_deltas )<3 


class TestSwitchProfile :
    @pytest .mark .anyio 
    async def test_switch_to_unknown_profile_raises (self ):
        stub =StubProvider ()
        core =AppCore (config =_make_config (),provider =stub )

        with pytest .raises (ProfileNotFoundError ):
            core .switch_profile ("nonexistent")

    @pytest .mark .anyio 
    async def test_switch_profile_updates_active_profile (self ):
        config =_make_config ()

        config .profiles ["other"]=ProfileConfig (
        name ="other",
        endpoint ="local",
        model ="other-model",
        capabilities =ModelCapabilities (),
        )
        stub =StubProvider ()
        core =AppCore (config =config ,provider =stub )

        core .switch_profile ("other")
        assert core .profile .name =="other"
        assert core .model =="other-model"


