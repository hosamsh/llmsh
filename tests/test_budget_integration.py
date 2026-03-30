"""Tests for ContextBudget integration into AppCore."""

import pytest 

from llmsh .app import AppCore 
from llmsh .budget import ContextBudget 
from llmsh .config import AppConfig 
from llmsh .models import (
EndpointConfig ,
ModelCapabilities ,
ModelInfo ,
ProfileConfig ,
UsageInfo ,
)
from llmsh .providers .base import (
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


class TestBudgetCreation :
    @pytest .mark .anyio 
    async def test_budget_created_on_first_send_when_context_length_available (self ):
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ],
        model_info =ModelInfo (id ="test-model",context_length =8192 ),
        )
        core =AppCore (config =_make_config (),provider =stub )

        assert core .budget is None 

        async for _ in core .send_message ("hello"):
            pass 

        assert core .budget is not None 
        assert isinstance (core .budget ,ContextBudget )
        assert core .budget .context_length ==8192 

    @pytest .mark .anyio 
    async def test_budget_is_none_when_get_model_info_raises (self ):
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ],
        model_info_error =RuntimeError ("API unavailable"),
        )
        core =AppCore (config =_make_config (),provider =stub )

        async for _ in core .send_message ("hello"):
            pass 

        assert core .budget is None 

    @pytest .mark .anyio 
    async def test_budget_is_none_when_context_length_is_none (self ):
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ],
        model_info =ModelInfo (id ="test-model",context_length =None ),
        )
        core =AppCore (config =_make_config (),provider =stub )

        async for _ in core .send_message ("hello"):
            pass 

        assert core .budget is None 


class TestBudgetUsageUpdate :
    @pytest .mark .anyio 
    async def test_budget_used_updated_from_token_usage_event (self ):
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


class TestClearMessages :
    @pytest .mark .anyio 
    async def test_budget_reset_after_clear (self ):
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =200 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =8192 ),
        )
        core =AppCore (config =_make_config (),provider =stub )

        async for _ in core .send_message ("hello"):
            pass 

        assert core .budget is not None 
        assert core .budget .used ==200 

        core .clear_messages ()

        assert core .budget .used ==0 

    @pytest .mark .anyio 
    async def test_clear_messages_preserves_system_prompt (self ):
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ],
        model_info =ModelInfo (id ="test-model",context_length =8192 ),
        )
        core =AppCore (
        config =_make_config (),provider =stub ,system_prompt ="You are helpful."
        )

        async for _ in core .send_message ("hello"):
            pass 

        core .clear_messages ()


        assert len (core ._messages )==1 
        assert core ._messages [0 ].role =="system"
        assert core ._messages [0 ].content =="You are helpful."


class TestModelInfoCalledOnce :
    @pytest .mark .anyio 
    async def test_get_model_info_called_only_once_across_multiple_sends (self ):
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        ],
        model_info =ModelInfo (id ="test-model",context_length =8192 ),
        )
        core =AppCore (config =_make_config (),provider =stub )

        async for _ in core .send_message ("first"):
            pass 
        async for _ in core .send_message ("second"):
            pass 
        async for _ in core .send_message ("third"):
            pass 

        assert len (stub .get_model_info_calls )==1 


class TestSwitchProfileResetsBudget :
    def test_switch_profile_resets_budget_and_model_info_fetched (self ):
        config =_make_config ()

        config .profiles ["other"]=ProfileConfig (
        name ="other",
        endpoint ="local",
        model ="other-model",
        capabilities =ModelCapabilities (),
        )
        stub =BudgetStubProvider (
        model_info =ModelInfo (id ="test-model",context_length =8192 ),
        )
        core =AppCore (config =config ,provider =stub )

        core ._budget =ContextBudget (8192 ,1024 )
        core ._budget .update_usage (500 )
        core ._model_info_fetched =True 

        core .switch_profile ("other")

        assert core ._budget is None 
        assert core ._model_info_fetched is False 
