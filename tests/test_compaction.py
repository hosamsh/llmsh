"""Tests for message window compaction."""

import pytest 

from llmsh .app import AppCore 
from llmsh .compaction import compact_messages 
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


class TestCompactMessagesLongConversation :
    def test_drops_middle_messages_when_conversation_longer_than_keep_recent (self ):
        """Long conversation should have middle messages dropped."""
        messages =[
        _msg ("system","You are helpful"),
        _msg ("user","msg1"),
        _msg ("assistant","reply1"),
        _msg ("user","msg2"),
        _msg ("assistant","reply2"),
        _msg ("user","msg3"),
        _msg ("assistant","reply3"),
        _msg ("user","msg4"),
        _msg ("assistant","reply4"),
        ]
        compacted ,num_dropped =compact_messages (messages ,keep_recent =4 )

        assert num_dropped ==4 
        assert len (compacted )==6 

    def test_preserves_system_prompt (self ):
        """System prompt (first message) is always kept."""
        messages =[
        _msg ("system","You are helpful"),
        _msg ("user","msg1"),
        _msg ("assistant","reply1"),
        _msg ("user","msg2"),
        _msg ("assistant","reply2"),
        _msg ("user","msg3"),
        _msg ("assistant","reply3"),
        _msg ("user","msg4"),
        _msg ("assistant","reply4"),
        ]
        compacted ,_ =compact_messages (messages ,keep_recent =4 )
        assert compacted [0 ].role =="system"
        assert compacted [0 ].content =="You are helpful"


class TestCompactMessagesFewMessages :
    def test_returns_unchanged_when_fewer_than_keep_recent (self ):
        """If not enough messages to drop, return unchanged."""
        messages =[
        _msg ("system","You are helpful"),
        _msg ("user","msg1"),
        _msg ("assistant","reply1"),
        ]
        compacted ,num_dropped =compact_messages (messages ,keep_recent =6 )
        assert num_dropped ==0 
        assert compacted ==messages 

    def test_returns_unchanged_when_exactly_threshold (self ):
        """If exactly at threshold (nothing to drop), return unchanged."""
        messages =[
        _msg ("system","You are helpful"),
        _msg ("user","msg1"),
        _msg ("assistant","reply1"),
        _msg ("user","msg2"),
        _msg ("assistant","reply2"),
        _msg ("user","msg3"),
        _msg ("assistant","reply3"),
        ]


        compacted ,num_dropped =compact_messages (messages ,keep_recent =6 )
        assert num_dropped ==0 
        assert compacted ==messages 


class TestCompactMessagesMarker :
    def test_inserts_marker_at_correct_position (self ):
        """Marker message should be right after system prompt."""
        messages =[
        _msg ("system","You are helpful"),
        _msg ("user","msg1"),
        _msg ("assistant","reply1"),
        _msg ("user","msg2"),
        _msg ("assistant","reply2"),
        _msg ("user","msg3"),
        _msg ("assistant","reply3"),
        _msg ("user","msg4"),
        _msg ("assistant","reply4"),
        ]
        compacted ,_ =compact_messages (messages ,keep_recent =4 )

        assert compacted [1 ].role =="system"
        assert "removed"in compacted [1 ].content .lower ()

    def test_marker_content_includes_dropped_count (self ):
        """Marker message should include the number of dropped messages."""
        messages =[
        _msg ("system","You are helpful"),
        _msg ("user","msg1"),
        _msg ("assistant","reply1"),
        _msg ("user","msg2"),
        _msg ("assistant","reply2"),
        _msg ("user","msg3"),
        _msg ("assistant","reply3"),
        _msg ("user","msg4"),
        _msg ("assistant","reply4"),
        ]
        compacted ,num_dropped =compact_messages (messages ,keep_recent =4 )
        assert str (num_dropped )in compacted [1 ].content 

    def test_no_system_prompt_marker_at_start (self ):
        """Without a system prompt, the marker is at position 0."""
        messages =[
        _msg ("user","msg1"),
        _msg ("assistant","reply1"),
        _msg ("user","msg2"),
        _msg ("assistant","reply2"),
        _msg ("user","msg3"),
        _msg ("assistant","reply3"),
        _msg ("user","msg4"),
        _msg ("assistant","reply4"),
        ]
        compacted ,num_dropped =compact_messages (messages ,keep_recent =4 )
        assert num_dropped ==4 

        assert compacted [0 ].role =="system"
        assert "4"in compacted [0 ].content 


class TestCompactionTriggersInSendMessage :
    @pytest .mark .anyio 
    async def test_compaction_triggers_at_80_percent_utilization (self ):
        """When budget utilization > 0.8, compaction should trigger."""
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

        for i in range (10 ):
            core ._messages .append (_msg ("user",f"msg{i }"))
            core ._messages .append (_msg ("assistant",f"reply{i }"))


        core .budget ._used =800 

        events =[]
        async for event in core .send_message ("next message"):
            events .append (event )

        compaction_events =[e for e in events if isinstance (e ,CompactionEvent )]
        assert len (compaction_events )==1 
        assert compaction_events [0 ].messages_dropped >0 

    @pytest .mark .anyio 
    async def test_no_compaction_at_70_percent_utilization (self ):
        """When budget utilization is 0.7, compaction should NOT trigger."""
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

        for i in range (10 ):
            core ._messages .append (_msg ("user",f"msg{i }"))
            core ._messages .append (_msg ("assistant",f"reply{i }"))


        core .budget ._used =680 

        events =[]
        async for event in core .send_message ("next message"):
            events .append (event )

        compaction_events =[e for e in events if isinstance (e ,CompactionEvent )]
        assert len (compaction_events )==0 


class TestCompactionEventContents :
    @pytest .mark .anyio 
    async def test_compaction_event_has_correct_dropped_count (self ):
        """CompactionEvent should report exactly how many messages were dropped."""
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

        for i in range (10 ):
            core ._messages .append (_msg ("user",f"msg{i }"))
            core ._messages .append (_msg ("assistant",f"reply{i }"))

        message_count_before =len (core ._messages )
        core .budget ._used =800 

        events =[]
        async for event in core .send_message ("trigger"):
            events .append (event )

        compaction_events =[e for e in events if isinstance (e ,CompactionEvent )]
        assert len (compaction_events )==1 



        expected_dropped =message_count_before +1 -6 
        assert compaction_events [0 ].messages_dropped ==expected_dropped 
        assert str (expected_dropped )in compaction_events [0 ].message 
