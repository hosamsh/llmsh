"""Tests for pre-flight budget check and overflow prevention."""

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import (
EndpointConfig ,
ModelCapabilities ,
ModelInfo ,
ProfileConfig ,
UsageInfo ,
)
from llmsh .providers .base import (
BudgetWarningEvent ,
ErrorEvent ,
ResponseCompleted ,
TextDelta ,
TokenUsageEvent ,
)
from tests .conftest import BudgetStubProvider 


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


class TestPreflightWarning :
    @pytest .mark .anyio 
    async def test_budget_at_95_percent_triggers_warning (self ):
        """When utilization is above 0.9, a BudgetWarningEvent is yielded."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =950 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )
        core =AppCore (config =_make_config (),provider =stub )


        async for _ in core .send_message ("init"):
            pass 



        assert core .budget is not None 
        core .budget ._used =930 

        events =[]
        async for event in core .send_message ("next message"):
            events .append (event )

        warning_events =[e for e in events if isinstance (e ,BudgetWarningEvent )]
        assert len (warning_events )==1 
        assert warning_events [0 ].utilization >0.9 
        assert warning_events [0 ].remaining_tokens >=0 
        msg =warning_events [0 ].message .lower ()
        assert "remaining"in msg or "full"in msg 

    @pytest .mark .anyio 
    async def test_budget_at_70_percent_no_warning (self ):
        """When utilization is below 0.9, no BudgetWarningEvent is yielded."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =700 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )
        core =AppCore (config =_make_config (),provider =stub )


        async for _ in core .send_message ("init"):
            pass 



        assert core .budget is not None 
        core .budget ._used =680 

        events =[]
        async for event in core .send_message ("next message"):
            events .append (event )

        warning_events =[e for e in events if isinstance (e ,BudgetWarningEvent )]
        assert len (warning_events )==0 


class TestPreflightOverflowBlock :
    @pytest .mark .anyio 
    async def test_overflow_blocked_with_error_event (self ):
        """When the message would not fit, an ErrorEvent is yielded and no API call."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =970 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )
        core =AppCore (config =_make_config (),provider =stub )


        async for _ in core .send_message ("init"):
            pass 



        assert core .budget is not None 
        core .budget ._used =975 


        stub .requests .clear ()


        long_message ="x"*100 
        events =[]
        async for event in core .send_message (long_message ):
            events .append (event )

        error_events =[e for e in events if isinstance (e ,ErrorEvent )]
        assert len (error_events )==1 
        assert error_events [0 ].error_type =="context_overflow"

        assert len (stub .requests )==0 

    @pytest .mark .anyio 
    async def test_blocked_message_rolled_back (self ):
        """When overflow is detected, the user message is removed from _messages."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =970 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )
        core =AppCore (config =_make_config (),provider =stub )


        async for _ in core .send_message ("init"):
            pass 

        assert core .budget is not None 
        core .budget ._used =975 

        message_count_before =len (core ._messages )

        long_message ="x"*100 
        async for _ in core .send_message (long_message ):
            pass 


        assert len (core ._messages )==message_count_before 

        assert core ._messages [-1 ].role =="assistant"


class TestConnectionErrorClassification :
    @pytest .mark .anyio 
    async def test_connection_error_high_utilization_classified_context_overflow (self ):
        """High utilization + connection error -> context_overflow."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =850 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )
        core =AppCore (config =_make_config (),provider =stub )


        async for _ in core .send_message ("init"):
            pass 

        assert core .budget is not None 

        core .budget ._used =800 


        stub ._error_on_stream =ConnectionError ("closed connection unexpectedly")
        stub .requests .clear ()

        events =[]
        async for event in core .send_message ("trigger error"):
            events .append (event )

        error_events =[e for e in events if isinstance (e ,ErrorEvent )]
        assert len (error_events )==1 
        assert error_events [0 ].error_type =="context_overflow"

    @pytest .mark .anyio 
    async def test_connection_error_no_budget_falls_back_to_char_heuristic (self ):
        """Without budget, the char-count heuristic is used for classification."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ],
        model_info =None ,
        )
        core =AppCore (config =_make_config (),provider =stub )


        async for _ in core .send_message ("init"):
            pass 

        assert core .budget is None 


        for i in range (50 ):
            core ._messages .append (
            __import__ ("llmsh.models",fromlist =["ChatMessage"]).ChatMessage (
            role ="user",content ="x"*100 
            )
            )

        stub ._error_on_stream =ConnectionError ("closed connection unexpectedly")
        stub .requests .clear ()

        events =[]
        async for event in core .send_message ("more text"):
            events .append (event )

        error_events =[e for e in events if isinstance (e ,ErrorEvent )]
        assert len (error_events )==1 
        assert error_events [0 ].error_type =="context_overflow"


class TestNoBudgetSkipsPreflight :
    @pytest .mark .anyio 
    async def test_no_budget_skips_preflight_checks (self ):
        """When budget is None, no warning or error events are emitted pre-flight."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ],
        model_info =None ,
        )
        core =AppCore (config =_make_config (),provider =stub )

        events =[]
        async for event in core .send_message ("hello"):
            events .append (event )

        assert core .budget is None 
        warning_events =[e for e in events if isinstance (e ,BudgetWarningEvent )]
        error_events =[e for e in events if isinstance (e ,ErrorEvent )]
        assert len (warning_events )==0 
        assert len (error_events )==0 

        completed =[e for e in events if isinstance (e ,ResponseCompleted )]
        assert len (completed )==1 
