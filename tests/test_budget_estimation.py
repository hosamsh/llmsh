"""Tests for budget estimation fallback when API doesn't report usage."""

import pytest 

from llmsh .app import AppCore 
from llmsh .budget import ContextBudget 
from llmsh .config import AppConfig 
from llmsh .models import (
ChatMessage ,
EndpointConfig ,
ModelCapabilities ,
ModelInfo ,
ProfileConfig ,
UsageInfo ,
)
from llmsh .providers .base import (
CompactionEvent ,
ResponseCompleted ,
TextDelta ,
TokenUsageEvent ,
)
from tests .conftest import BudgetStubProvider 


def _make_config (max_tokens :int |None =None )->AppConfig :
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
    max_tokens =max_tokens ,
    )
    return AppConfig (
    current_profile ="default",
    endpoints ={"local":endpoint },
    profiles ={"default":profile },
    )


class TestEstimateTokensRatio :
    def test_estimate_tokens_uses_div_3 (self ):
        """estimate_tokens returns len(text) // 3."""
        b =ContextBudget (context_length =8000 ,reserved_output =2000 )
        text ="a"*99 
        assert b .estimate_tokens (text )==33 


class TestTokenUsageEventReceived :
    @pytest .mark .anyio 
    async def test_api_usage_used_when_token_event_received (self ):
        """When TokenUsageEvent IS received, API value is used."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =150 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =8192 ),
        )
        core =AppCore (config =_make_config (),provider =stub )

        async for _ in core .send_message ("hello"):
            pass 

        assert core .budget is not None 

        assert core .budget .used ==150 


class TestFallbackEstimation :
    @pytest .mark .anyio 
    async def test_no_token_event_triggers_estimation (self ):
        """When no TokenUsageEvent after ResponseCompleted, budget.used is estimated."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),

        ],
        model_info =ModelInfo (id ="test-model",context_length =8192 ),
        )
        core =AppCore (config =_make_config (),provider =stub )

        async for _ in core .send_message ("hello"):
            pass 

        assert core .budget is not None 

        assert core .budget .used >0 

    @pytest .mark .anyio 
    async def test_estimation_sums_all_messages (self ):
        """Estimation should sum ALL messages, not just the latest."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="response one"),
        ResponseCompleted (content ="response one"),

        ],
        model_info =ModelInfo (id ="test-model",context_length =8192 ),
        )
        core =AppCore (config =_make_config (),provider =stub )


        async for _ in core .send_message ("first message"):
            pass 

        assert core .budget is not None 
        used_after_first =core .budget .used 


        stub .events =[
        TextDelta (text ="response two"),
        ResponseCompleted (content ="response two"),
        ]


        async for _ in core .send_message ("second message"):
            pass 

        used_after_second =core .budget .used 


        assert used_after_second >used_after_first 


        expected =sum (
        core .budget .estimate_tokens (m .content )for m in core ._messages 
        )
        assert core .budget .used ==expected 


class TestCompactionWithEstimation :
    @pytest .mark .anyio 
    async def test_compaction_triggers_when_estimated_usage_exceeds_threshold (self ):
        """Compaction triggers when estimated usage exceeds 80% threshold.

        Flow: many messages without TokenUsageEvent -> fallback estimation
        sets budget.used high -> next send_message triggers compaction.
        """




        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="ok"),
        ResponseCompleted (content ="ok"),

        ],
        model_info =ModelInfo (id ="test-model",context_length =4000 ),
        )
        core =AppCore (config =_make_config (max_tokens =200 ),provider =stub )


        async for _ in core .send_message ("init"):
            pass 

        assert core .budget is not None 




        for _ in range (10 ):
            core ._messages .append (
            ChatMessage (role ="user",content ="x"*500 )
            )
            core ._messages .append (
            ChatMessage (role ="assistant",content ="y"*500 )
            )



        stub .events =[
        TextDelta (text ="ok"),
        ResponseCompleted (content ="ok"),
        ]
        async for _ in core .send_message ("push over threshold"):
            pass 


        assert core .budget .utilization >0.8 


        stub .events =[
        TextDelta (text ="after compaction"),
        ResponseCompleted (content ="after compaction"),
        ]
        events =[]
        async for event in core .send_message ("trigger compaction"):
            events .append (event )

        compaction_events =[e for e in events if isinstance (e ,CompactionEvent )]
        assert len (compaction_events )==1 
