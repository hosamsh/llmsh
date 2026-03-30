"""Tests for map-reduce file summarization engine."""

from __future__ import annotations 

import pytest 

from llmsh .errors import LlmshError 
from llmsh .providers .base import (
BaseProvider ,
ProviderRequest ,
ProviderResult ,
)
from llmsh .summarize import (
SummarizePlan ,
_batch_findings ,
chunk_text ,
estimate_plan ,
map_chunk ,
reduce_findings ,
summarize_file ,
)


class RecordingProvider (BaseProvider ):
    """Provider that records calls and returns canned responses in sequence.

    Responses can be strings (non-truncated) or tuples of (str, bool)
    where the bool indicates truncation.
    """

    def __init__ (self ,responses :list [str |tuple [str ,bool ]])->None :
        self .requests :list [ProviderRequest ]=[]
        self ._responses =list (responses )
        self ._call_index =0 

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        self .requests .append (request )
        if self ._call_index <len (self ._responses ):
            entry =self ._responses [self ._call_index ]
        else :
            entry =""
        self ._call_index +=1 
        if isinstance (entry ,tuple ):
            return ProviderResult (content =entry [0 ],truncated =entry [1 ])
        return ProviderResult (content =entry )





def test_chunk_text_short_text_returns_single_chunk ()->None :
    text ="line one\nline two\n"
    result =chunk_text (text ,max_chars =100 )
    assert result ==[text ]


def test_chunk_text_splits_at_line_boundaries ()->None :
    lines =[f"line {i }\n"for i in range (10 )]
    text ="".join (lines )
    result =chunk_text (text ,max_chars =21 )
    assert len (result )>1 
    assert "".join (result )==text 
    for chunk in result :
        for line in chunk .splitlines (keepends =True ):
            assert line .endswith ("\n")or chunk ==result [-1 ]


def test_chunk_text_empty_text_returns_empty_list ()->None :
    result =chunk_text ("",max_chars =100 )
    assert result ==[]


def test_chunk_text_single_long_line_returns_one_chunk ()->None :
    long_line ="x"*500 
    result =chunk_text (long_line ,max_chars =100 )
    assert len (result )==1 
    assert result [0 ]==long_line 


def test_chunk_text_respects_max_chars_boundary ()->None :
    lines =[f"{'a'*20 }\n"for _ in range (20 )]
    text ="".join (lines )
    max_chars =50 
    result =chunk_text (text ,max_chars =max_chars )
    for chunk in result :
        lines_in_chunk =chunk .splitlines (keepends =True )
        if len (lines_in_chunk )==1 :
            continue 
        assert len (chunk )<=max_chars 





@pytest .mark .anyio 
async def test_map_chunk_calls_provider_with_instruction_and_chunk ()->None :
    provider =RecordingProvider (["finding A"])
    content_result ,truncated =await map_chunk (
    provider ,"test-model","find errors","chunk content",2 ,5 ,max_tokens =456 
    )
    assert content_result =="finding A"
    assert truncated is False 
    assert len (provider .requests )==1 
    req =provider .requests [0 ]
    assert req .model =="test-model"
    assert req .max_tokens ==456 
    content =req .messages [0 ].content 
    assert "find errors"in content 
    assert "chunk content"in content 
    assert "3 of 5"in content 


@pytest .mark .anyio 
async def test_map_chunk_has_no_state_parameter ()->None :
    """map_chunk must not accept or include inter-chunk state."""
    provider =RecordingProvider (["finding"])
    await map_chunk (provider ,"m","task","chunk",0 ,1 ,max_tokens =100 )
    content =provider .requests [0 ].messages [0 ].content 
    assert "extraction state"not in content .lower ()
    assert "previous chunks"not in content .lower ()


@pytest .mark .anyio 
async def test_map_chunk_returns_truncated_flag_from_provider ()->None :
    """map_chunk passes through the truncated flag from provider."""
    provider =RecordingProvider ([("partial finding",True )])
    content_result ,truncated =await map_chunk (
    provider ,"m","task","chunk",0 ,1 ,max_tokens =100 
    )
    assert content_result =="partial finding"
    assert truncated is True 





@pytest .mark .anyio 
async def test_reduce_findings_calls_provider_with_instruction_and_batch ()->None :
    provider =RecordingProvider (["merged findings"])
    content_result ,truncated =await reduce_findings (
    provider ,"test-model","find errors",["finding 1","finding 2"],
    max_tokens =1024 ,
    )
    assert content_result =="merged findings"
    assert truncated is False 
    assert len (provider .requests )==1 
    req =provider .requests [0 ]
    assert req .model =="test-model"
    assert req .max_tokens ==1024 
    content =req .messages [0 ].content 
    assert "find errors"in content 
    assert "finding 1"in content 
    assert "finding 2"in content 


@pytest .mark .anyio 
async def test_reduce_findings_returns_truncated_flag_from_provider ()->None :
    """reduce_findings passes through the truncated flag from provider."""
    provider =RecordingProvider ([("partial merge",True )])
    content_result ,truncated =await reduce_findings (
    provider ,"m","task",["f1","f2"],max_tokens =100 
    )
    assert content_result =="partial merge"
    assert truncated is True 





def test_batch_findings_groups_within_budget ()->None :

    findings =["a"*30 ,"b"*30 ,"c"*30 ,"d"*30 ]

    batches =_batch_findings (findings ,safe_input =50 ,prompt_overhead =10 )
    assert len (batches )==1 
    assert batches [0 ]==findings 


def test_batch_findings_starts_new_batch_on_overflow ()->None :

    findings =["a"*30 ,"b"*30 ,"c"*30 ]

    batches =_batch_findings (findings ,safe_input =30 ,prompt_overhead =5 )
    assert len (batches )==2 
    assert batches [0 ]==["a"*30 ,"b"*30 ]
    assert batches [1 ]==["c"*30 ]


def test_batch_findings_single_large_finding_own_batch ()->None :

    small ="a"*30 
    large ="b"*300 

    batches =_batch_findings ([small ,large ,small ],safe_input =30 ,prompt_overhead =5 )

    assert len (batches )==3 
    assert batches [0 ]==[small ]
    assert batches [1 ]==[large ]
    assert batches [2 ]==[small ]





def test_estimate_plan_single_chunk ()->None :
    plan =estimate_plan (
    num_chunks =1 ,safe_input =1024 ,
    prompt_overhead =100 ,map_max_output =456 ,
    )
    assert plan .total_chunks ==1 
    assert plan .estimated_reduce_rounds ==0 
    assert plan .estimated_total_calls ==1 


def test_estimate_plan_multiple_chunks ()->None :




    plan =estimate_plan (
    num_chunks =4 ,safe_input =1024 ,
    prompt_overhead =100 ,map_max_output =456 ,
    )
    assert plan .total_chunks ==4 
    assert plan .estimated_reduce_rounds ==2 
    assert plan .estimated_total_calls ==7 


def test_estimate_plan_many_chunks ()->None :



    plan =estimate_plan (
    num_chunks =8 ,safe_input =1024 ,
    prompt_overhead =100 ,map_max_output =456 ,
    )
    assert plan .total_chunks ==8 
    assert plan .estimated_reduce_rounds ==3 
    assert plan .estimated_total_calls ==15 





@pytest .mark .anyio 
async def test_summarize_file_single_chunk_no_reduce ()->None :
    """Single chunk: 1 map call, no reduce, returns finding directly."""
    provider =RecordingProvider (["the finding"])
    answer ,chunks ,truncated_calls =await summarize_file (
    provider ,"test-model","summarize","short text\n",
    context_length =8000 ,max_output_tokens =1024 ,
    )
    assert answer =="the finding"
    assert chunks ==1 
    assert truncated_calls ==0 
    assert len (provider .requests )==1 


@pytest .mark .anyio 
async def test_summarize_file_multi_chunk_maps_then_reduces ()->None :
    """Multiple chunks: map all, then reduce until one result."""




    lines =[f"{'x'*80 }\n"for _ in range (50 )]
    file_text ="".join (lines )

    responses =[f"finding-{i }"for i in range (50 )]+["reduced"]*50 
    provider =RecordingProvider (responses )
    answer ,chunks ,truncated_calls =await summarize_file (
    provider ,"test-model","summarize",file_text ,
    context_length =500 ,max_output_tokens =100 ,
    )
    assert chunks >1 
    assert truncated_calls ==0 

    map_calls =chunks 
    total_calls =len (provider .requests )
    assert total_calls >map_calls 


@pytest .mark .anyio 
async def test_summarize_file_single_item_batch_no_llm_call ()->None :
    """Single-item batch passes through without LLM call."""



    lines =[f"{'a'*95 }\n"for _ in range (10 )]
    file_text ="".join (lines )
    provider =RecordingProvider ([f"f{i }"for i in range (20 )]+["reduced"]*20 )
    answer ,chunks ,truncated_calls =await summarize_file (
    provider ,"test-model","summarize",file_text ,
    context_length =400 ,max_output_tokens =100 ,
    )
    assert chunks >1 

    for req in provider .requests [chunks :]:
        content =req .messages [0 ].content 

        assert "Findings set 1"in content 
        assert "Findings set 2"in content 


@pytest .mark .anyio 
async def test_summarize_file_calls_on_plan ()->None :
    """on_plan is called once before execution with estimated plan."""
    provider =RecordingProvider (["finding"])
    plans :list [SummarizePlan ]=[]

    def on_plan (plan :SummarizePlan )->None :
        plans .append (plan )

    answer ,chunks ,truncated_calls =await summarize_file (
    provider ,"test-model","summarize","short\n",
    context_length =8000 ,max_output_tokens =1024 ,
    on_plan =on_plan ,
    )
    assert len (plans )==1 
    assert plans [0 ].total_chunks ==1 
    assert plans [0 ].estimated_reduce_rounds ==0 


@pytest .mark .anyio 
async def test_summarize_file_calls_on_progress_with_phase ()->None :
    """on_progress reports phase and position."""



    lines =[f"{'z'*95 }\n"for _ in range (10 )]
    file_text ="".join (lines )
    responses =[f"f{i }"for i in range (20 )]+["reduced"]*20 
    provider =RecordingProvider (responses )
    progress :list [tuple [str ,int ,int ]]=[]

    def on_progress (phase :str ,current :int ,total :int )->None :
        progress .append ((phase ,current ,total ))

    answer ,chunks ,truncated_calls =await summarize_file (
    provider ,"test-model","summarize",file_text ,
    context_length =400 ,max_output_tokens =100 ,
    on_progress =on_progress ,
    )

    map_events =[p for p in progress if p [0 ]=="map"]
    assert len (map_events )>=2 

    assert map_events [0 ][1 ]==1 

    reduce_events =[p for p in progress if p [0 ].startswith ("reduce")]
    assert len (reduce_events )>=1 


@pytest .mark .anyio 
async def test_summarize_file_works_with_2048_context ()->None :
    """2048-context model must not raise 'too small'."""
    provider =RecordingProvider (["finding"])
    answer ,chunks ,truncated_calls =await summarize_file (
    provider ,"test-model","summarize","short text\n",
    context_length =2048 ,max_output_tokens =1024 ,
    )
    assert answer =="finding"
    assert chunks ==1 


@pytest .mark .anyio 
async def test_summarize_file_works_with_16384_context ()->None :
    """16384-context model produces positive chunk budget with 60% cap."""


    provider =RecordingProvider (["finding"])
    answer ,chunks ,truncated_calls =await summarize_file (
    provider ,"test-model","summarize","short text\n",
    context_length =16384 ,max_output_tokens =1024 ,
    )
    assert answer =="finding"
    assert chunks ==1 


@pytest .mark .anyio 
async def test_summarize_file_raises_on_small_context ()->None :
    with pytest .raises (LlmshError ,match ="too small"):
        await summarize_file (
        RecordingProvider ([]),
        "test-model","summarize","some text",
        context_length =100 ,max_output_tokens =90 ,
        )


@pytest .mark .anyio 
async def test_summarize_file_returns_3_tuple ()->None :
    """Returns (answer, chunks_processed, truncated_calls)."""
    provider =RecordingProvider (["the answer"])
    result =await summarize_file (
    provider ,"test-model","summarize","text\n",
    context_length =8000 ,max_output_tokens =1024 ,
    )
    assert isinstance (result ,tuple )
    assert len (result )==3 
    answer ,chunks ,truncated_calls =result 
    assert isinstance (answer ,str )
    assert isinstance (chunks ,int )
    assert isinstance (truncated_calls ,int )
    assert truncated_calls ==0 


@pytest .mark .anyio 
async def test_summarize_file_counts_truncated_calls ()->None :
    """Truncated calls are counted across map and reduce phases."""


    lines =[f"{'z'*95 }\n"for _ in range (10 )]
    file_text ="".join (lines )

    responses :list [str |tuple [str ,bool ]]=[("finding-0",True )]
    responses +=["finding-1"]

    responses +=[("reduced",True )]
    responses +=["reduced"]*20 
    provider =RecordingProvider (responses )
    answer ,chunks ,truncated_calls =await summarize_file (
    provider ,"test-model","summarize",file_text ,
    context_length =600 ,max_output_tokens =100 ,
    )
    assert truncated_calls >=2 


@pytest .mark .anyio 
async def test_summarize_file_uses_available_output_for_max_tokens ()->None :
    """Map and reduce calls should use dynamic max_tokens, not a fixed value."""



    provider =RecordingProvider (["finding"])
    await summarize_file (
    provider ,"test-model","summarize","short text\n",
    context_length =10000 ,max_output_tokens =500 ,
    )
    req =provider .requests [0 ]
    assert req .max_tokens ==4000 





@pytest .mark .anyio 
async def test_summarize_file_raises_when_cancelled_before_map ()->None :
    """summarize_file raises LlmshError when cancelled callback returns True."""
    provider =RecordingProvider (["finding"])
    with pytest .raises (LlmshError ,match ="cancelled"):
        await summarize_file (
        provider ,"test-model","summarize","short text\n",
        context_length =8000 ,max_output_tokens =1024 ,
        cancelled =lambda :True ,
        )

    assert len (provider .requests )==0 


@pytest .mark .anyio 
async def test_summarize_file_cancellation_checked_between_map_calls ()->None :
    """Cancellation is checked between each map call, not just at the start."""
    call_count =0 

    def cancel_after_first ()->bool :
        return call_count >=1 



    lines =[f"{'x'*95 }\n"for _ in range (10 )]
    file_text ="".join (lines )

    class CountingProvider (RecordingProvider ):
        async def chat (self ,request :ProviderRequest )->ProviderResult :
            nonlocal call_count 
            result =await super ().chat (request )
            call_count +=1 
            return result 

    provider =CountingProvider ([f"finding-{i }"for i in range (20 )])

    with pytest .raises (LlmshError ,match ="cancelled"):
        await summarize_file (
        provider ,"test-model","summarize",file_text ,
        context_length =400 ,max_output_tokens =100 ,
        cancelled =cancel_after_first ,
        )

    assert call_count ==1 


@pytest .mark .anyio 
async def test_summarize_file_cancellation_checked_before_reduce ()->None :
    """Cancellation is checked before each reduce batch."""

    lines =[f"{'y'*95 }\n"for _ in range (10 )]
    file_text ="".join (lines )
    responses =[f"finding-{i }"for i in range (20 )]+["reduced"]*20 
    provider =RecordingProvider (responses )

    map_done =False 

    def cancel_at_reduce ()->bool :
        return map_done 


    progress :list [tuple [str ,int ,int ]]=[]

    def on_progress (phase :str ,current :int ,total :int )->None :
        nonlocal map_done 
        progress .append ((phase ,current ,total ))
        if phase =="map"and current ==total :
            map_done =True 

    with pytest .raises (LlmshError ,match ="cancelled"):
        await summarize_file (
        provider ,"test-model","summarize",file_text ,
        context_length =400 ,max_output_tokens =100 ,
        cancelled =cancel_at_reduce ,
        on_progress =on_progress ,
        )

    map_events =[p for p in progress if p [0 ]=="map"]
    assert len (map_events )>=2 
    reduce_events =[p for p in progress if p [0 ].startswith ("reduce")]
    assert len (reduce_events )==0 


@pytest .mark .anyio 
async def test_summarize_file_not_cancelled_when_callback_returns_false ()->None :
    """summarize_file completes normally when cancelled returns False."""
    provider =RecordingProvider (["the finding"])
    answer ,chunks ,truncated =await summarize_file (
    provider ,"test-model","summarize","short text\n",
    context_length =8000 ,max_output_tokens =1024 ,
    cancelled =lambda :False ,
    )
    assert answer =="the finding"
    assert chunks ==1 
