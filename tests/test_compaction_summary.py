"""Tests for LLM-powered conversation summary on compaction."""

import pytest 

from llmsh .app import AppCore 
from llmsh .compaction import summarize_dropped 
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
ProviderRequest ,
ProviderResult ,
ResponseCompleted ,
TextDelta ,
TokenUsageEvent ,
)
from tests .conftest import BudgetStubProvider ,StubProvider 


def _msg (role :str ,content :str )->ChatMessage :
    return ChatMessage (role =role ,content =content )


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


class TestSummarizeDropped :
    @pytest .mark .anyio 
    async def test_calls_provider_chat_with_correct_prompt_structure (self ):
        """summarize_dropped should call provider.chat with messages containing
        the dropped conversation content."""
        stub =StubProvider (
        events =[ResponseCompleted (content ="A summary of the conversation.")]
        )
        dropped =[
        _msg ("user","What is Python?"),
        _msg ("assistant","Python is a programming language."),
        ]
        await summarize_dropped (stub ,"test-model",dropped )

        assert len (stub .requests )==1 
        request =stub .requests [0 ]
        assert request .model =="test-model"

        prompt_text =" ".join (m .content for m in request .messages )
        assert "What is Python?"in prompt_text 
        assert "Python is a programming language."in prompt_text 

    @pytest .mark .anyio 
    async def test_returns_summary_text (self ):
        """summarize_dropped should return the content from provider.chat."""
        stub =StubProvider (
        events =[ResponseCompleted (content ="The user asked about Python.")]
        )
        dropped =[_msg ("user","hello"),_msg ("assistant","hi")]
        result =await summarize_dropped (stub ,"test-model",dropped )

        assert result =="The user asked about Python."

    @pytest .mark .anyio 
    async def test_uses_max_tokens_256 (self ):
        """The summary request should use max_tokens=256."""
        stub =StubProvider (
        events =[ResponseCompleted (content ="summary")]
        )
        dropped =[_msg ("user","hello")]
        await summarize_dropped (stub ,"test-model",dropped )

        assert len (stub .requests )==1 
        assert stub .requests [0 ].max_tokens ==256 

    @pytest .mark .anyio 
    async def test_uses_non_streaming_chat (self ):
        """summarize_dropped should call provider.chat (non-streaming),
        not stream_chat. We verify by checking that chat was called
        (requests list populated) and stream_chat was not separately invoked."""
        chat_called =False 
        stream_called =False 

        class TrackingProvider (StubProvider ):
            async def chat (self ,request :ProviderRequest )->ProviderResult :
                nonlocal chat_called 
                chat_called =True 
                return await super ().chat (request )

            async def stream_chat (self ,request ):
                nonlocal stream_called 
                stream_called =True 
                async for event in super ().stream_chat (request ):
                    yield event 

        provider =TrackingProvider (
        events =[ResponseCompleted (content ="summary")]
        )
        dropped =[_msg ("user","hello")]
        await summarize_dropped (provider ,"test-model",dropped )

        assert chat_called is True 
        assert stream_called is False 


class TestCompactionSummaryIntegration :
    @pytest .mark .anyio 
    async def test_summary_replaces_simple_marker (self ):
        """When compaction drops messages, the marker should contain
        the LLM-generated summary."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =850 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )

        summary_text ="User discussed Python basics."

        async def mock_chat (request :ProviderRequest )->ProviderResult :

            return ProviderResult (content =summary_text )

        stub .chat =mock_chat 

        core =AppCore (
        config =_make_config (),provider =stub ,system_prompt ="You are helpful"
        )


        async for _ in core .send_message ("init"):
            pass 

        assert core .budget is not None 

        for i in range (10 ):
            core ._messages .append (_msg ("user",f"msg{i }"))
            core ._messages .append (_msg ("assistant",f"reply{i }"))

        core .budget ._used =800 

        events =[]
        async for event in core .send_message ("next message"):
            events .append (event )

        compaction_events =[e for e in events if isinstance (e ,CompactionEvent )]
        assert len (compaction_events )==1 

        marker =core ._messages [1 ]
        assert marker .role =="system"
        assert summary_text in marker .content 

    @pytest .mark .anyio 
    async def test_compaction_event_reflects_summary (self ):
        """CompactionEvent.message should indicate a summary was generated."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =850 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )
        summary_text ="Brief summary of conversation."
        async def mock_chat (request :ProviderRequest )->ProviderResult :
            return ProviderResult (content =summary_text )

        stub .chat =mock_chat 

        core =AppCore (
        config =_make_config (),provider =stub ,system_prompt ="You are helpful"
        )

        async for _ in core .send_message ("init"):
            pass 

        assert core .budget is not None 
        for i in range (10 ):
            core ._messages .append (_msg ("user",f"msg{i }"))
            core ._messages .append (_msg ("assistant",f"reply{i }"))

        core .budget ._used =800 

        events =[]
        async for event in core .send_message ("trigger"):
            events .append (event )

        compaction_events =[e for e in events if isinstance (e ,CompactionEvent )]
        assert len (compaction_events )==1 
        assert "Summary"in compaction_events [0 ].message 

    @pytest .mark .anyio 
    async def test_fallback_to_simple_marker_on_summary_failure (self ):
        """When provider.chat raises, the simple marker should be used instead."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =850 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )

        async def failing_chat (request :ProviderRequest )->ProviderResult :
            raise RuntimeError ("API unavailable")

        stub .chat =failing_chat 

        core =AppCore (
        config =_make_config (),provider =stub ,system_prompt ="You are helpful"
        )

        async for _ in core .send_message ("init"):
            pass 

        assert core .budget is not None 
        for i in range (10 ):
            core ._messages .append (_msg ("user",f"msg{i }"))
            core ._messages .append (_msg ("assistant",f"reply{i }"))

        core .budget ._used =800 

        events =[]
        async for event in core .send_message ("trigger"):
            events .append (event )


        compaction_events =[e for e in events if isinstance (e ,CompactionEvent )]
        assert len (compaction_events )==1 
        assert compaction_events [0 ].messages_dropped >0 

        marker =core ._messages [1 ]
        assert marker .role =="system"
        assert "removed"in marker .content .lower ()

    @pytest .mark .anyio 
    async def test_summary_marker_has_system_role (self ):
        """The summary marker message should have role='system'."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =850 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )

        async def mock_chat (request :ProviderRequest )->ProviderResult :
            return ProviderResult (content ="a summary")

        stub .chat =mock_chat 

        core =AppCore (
        config =_make_config (),provider =stub ,system_prompt ="You are helpful"
        )

        async for _ in core .send_message ("init"):
            pass 

        assert core .budget is not None 
        for i in range (10 ):
            core ._messages .append (_msg ("user",f"msg{i }"))
            core ._messages .append (_msg ("assistant",f"reply{i }"))

        core .budget ._used =800 

        async for _ in core .send_message ("trigger"):
            pass 

        marker =core ._messages [1 ]
        assert marker .role =="system"

    @pytest .mark .anyio 
    async def test_summary_call_does_not_modify_messages (self ):
        """The summary call through provider.chat should not add anything
        to _messages. Only the marker from compact_messages is modified."""
        stub =BudgetStubProvider (
        events =[
        TextDelta (text ="hi"),
        ResponseCompleted (content ="hi"),
        TokenUsageEvent (usage =UsageInfo (input_tokens =850 )),
        ],
        model_info =ModelInfo (id ="test-model",context_length =2000 ),
        )

        async def mock_chat (request :ProviderRequest )->ProviderResult :
            return ProviderResult (content ="a summary")

        stub .chat =mock_chat 

        core =AppCore (
        config =_make_config (),provider =stub ,system_prompt ="You are helpful"
        )

        async for _ in core .send_message ("init"):
            pass 

        assert core .budget is not None 
        for i in range (10 ):
            core ._messages .append (_msg ("user",f"msg{i }"))
            core ._messages .append (_msg ("assistant",f"reply{i }"))

        core .budget ._used =800 






        async for _ in core .send_message ("trigger"):
            pass 



        roles =[m .role for m in core ._messages ]


        system_count =sum (1 for r in roles if r =="system")
        assert system_count ==2 
