"""Map-reduce file summarization engine.

Processes files larger than model context by splitting into chunks,
extracting findings independently (map), then merging in batches (reduce)
until one result remains.
"""

from __future__ import annotations 

from collections .abc import Callable 
from dataclasses import dataclass 

from llmsh .errors import LlmshError 
from llmsh .models import ChatMessage 
from llmsh .providers .base import BaseProvider ,ProviderRequest 


def _estimate_tokens (text :str )->int :
    return len (text )//3 


def chunk_text (text :str ,max_chars :int )->list [str ]:
    """Split text into chunks of at most max_chars, breaking at line boundaries.

    If a single line exceeds max_chars, it becomes its own chunk.
    """
    if not text :
        return []
    lines =text .splitlines (keepends =True )
    chunks :list [str ]=[]
    current =""
    for line in lines :
        if current and len (current )+len (line )>max_chars :
            chunks .append (current )
            current =""
        current +=line 
    if current :
        chunks .append (current )
    return chunks 


async def map_chunk (
provider :BaseProvider ,
model :str ,
instruction :str ,
chunk :str ,
chunk_index :int ,
total_chunks :int ,
max_tokens :int ,
)->tuple [str ,bool ]:
    """Map phase: extract findings from a single chunk independently.

    Returns (findings, truncated).
    """
    prompt =(
    f"Task: {instruction }\n\n"
    f"Chunk {chunk_index +1 } of {total_chunks }.\n\n"
    f"Extract all information relevant to the task from this chunk.\n"
    f"Output ONLY the relevant findings from this chunk.\n\n"
    f"---\n{chunk }\n---"
    )
    request =ProviderRequest (
    messages =[ChatMessage (role ="user",content =prompt )],
    model =model ,
    max_tokens =max_tokens ,
    )
    result =await provider .chat (request )
    return result .content ,result .truncated 


async def reduce_findings (
provider :BaseProvider ,
model :str ,
instruction :str ,
findings_batch :list [str ],
max_tokens :int ,
)->tuple [str ,bool ]:
    """Reduce phase: merge a batch of findings into a single consolidated summary.

    Returns (merged_findings, truncated).
    """
    sections =""
    for i ,finding in enumerate (findings_batch ):
        sections +=f"\nFindings set {i +1 }:\n{finding }\n"
    prompt =(
    f"Merge the following sets of findings into a single consolidated summary.\n"
    f"Task: {instruction }\n"
    f"{sections }\n"
    f"Combine all findings into one unified summary. "
    f"Deduplicate, aggregate counts, and organize clearly."
    )
    request =ProviderRequest (
    messages =[ChatMessage (role ="user",content =prompt )],
    model =model ,
    max_tokens =max_tokens ,
    )
    result =await provider .chat (request )
    return result .content ,result .truncated 


def _batch_findings (
findings :list [str ],safe_input :int ,prompt_overhead :int 
)->list [list [str ]]:
    """Group findings into batches that fit within context budget."""
    batches :list [list [str ]]=[]
    current_batch :list [str ]=[]
    current_tokens =prompt_overhead 
    for finding in findings :
        finding_tokens =_estimate_tokens (finding )
        if current_batch and current_tokens +finding_tokens >safe_input :
            batches .append (current_batch )
            current_batch =[finding ]
            current_tokens =prompt_overhead +finding_tokens 
        else :
            current_batch .append (finding )
            current_tokens +=finding_tokens 
    if current_batch :
        batches .append (current_batch )
    return batches 


@dataclass 
class SummarizePlan :
    total_chunks :int 
    estimated_reduce_rounds :int 
    estimated_total_calls :int 


def estimate_plan (
num_chunks :int ,
safe_input :int ,
prompt_overhead :int ,
map_max_output :int ,
)->SummarizePlan :
    """Estimate the execution plan based on worst-case assumptions."""
    findings_per_batch =max (2 ,(safe_input -prompt_overhead )//map_max_output )
    reduce_rounds =0 
    current_count =num_chunks 
    total_reduce_calls =0 
    while current_count >1 :
        reduce_rounds +=1 
        current_count =-(-current_count //findings_per_batch )
        total_reduce_calls +=current_count 
    return SummarizePlan (num_chunks ,reduce_rounds ,num_chunks +total_reduce_calls )


async def _run_map_phase (
provider :BaseProvider ,
model :str ,
instruction :str ,
chunks :list [str ],
available_output :int ,
on_progress :Callable [[str ,int ,int ],None ]|None ,
cancelled :Callable [[],bool ]|None ,
)->tuple [list [str ],int ]:
    """Map phase: extract findings from each chunk.

    Returns (findings, truncated_count).
    """
    truncated_count =0 
    findings :list [str ]=[]
    total_chunks =len (chunks )
    for i ,chunk in enumerate (chunks ):
        if cancelled is not None and cancelled ():
            raise LlmshError ("Summarization cancelled")
        finding ,truncated =await map_chunk (
        provider ,model ,instruction ,chunk ,i ,total_chunks ,available_output ,
        )
        findings .append (finding )
        if truncated :
            truncated_count +=1 
        if on_progress is not None :
            on_progress ("map",i +1 ,total_chunks )
    return findings ,truncated_count 


async def _run_reduce_phase (
provider :BaseProvider ,
model :str ,
instruction :str ,
findings :list [str ],
safe_input :int ,
prompt_overhead :int ,
available_output :int ,
on_progress :Callable [[str ,int ,int ],None ]|None ,
cancelled :Callable [[],bool ]|None ,
)->tuple [list [str ],int ]:
    """Reduce phase: merge findings until one remains.

    Returns (findings, truncated_count).
    """
    truncated_count =0 
    reduce_round =0 
    while len (findings )>1 :
        reduce_round +=1 
        batches =_batch_findings (findings ,safe_input ,prompt_overhead )
        new_findings :list [str ]=[]
        batch_count =sum (1 for b in batches if len (b )>1 )
        progress_index =0 
        for batch in batches :
            if cancelled is not None and cancelled ():
                raise LlmshError ("Summarization cancelled")
            if len (batch )==1 :
                new_findings .append (batch [0 ])
            else :
                merged ,truncated =await reduce_findings (
                provider ,model ,instruction ,batch ,available_output ,
                )
                new_findings .append (merged )
                if truncated :
                    truncated_count +=1 
                progress_index +=1 
                if on_progress is not None :
                    on_progress (f"reduce-{reduce_round }",progress_index ,batch_count )
        findings =new_findings 
    return findings ,truncated_count 


async def summarize_file (
provider :BaseProvider ,
model :str ,
instruction :str ,
file_text :str ,
context_length :int ,
max_output_tokens :int =1024 ,
on_plan :Callable [[SummarizePlan ],None ]|None =None ,
on_progress :Callable [[str ,int ,int ],None ]|None =None ,
cancelled :Callable [[],bool ]|None =None ,
)->tuple [str ,int ,int ]:
    """Orchestrate map-reduce summarization of a file.

    Returns a tuple of (answer, chunks_processed, truncated_calls).
    """
    safe_input =int (context_length *0.60 )
    available_output =context_length -safe_input 
    prompt_overhead =_estimate_tokens (instruction )+100 
    chunk_budget =safe_input -prompt_overhead 
    max_chars =chunk_budget *3 

    if max_chars <100 :
        raise LlmshError ("Context window too small for chunked summarization")

    chunks =chunk_text (file_text ,max_chars )
    total_chunks =len (chunks )

    est_finding =min (available_output ,1024 )
    plan =estimate_plan (total_chunks ,safe_input ,prompt_overhead ,est_finding )
    if on_plan is not None :
        on_plan (plan )

    findings ,map_truncated =await _run_map_phase (
    provider ,model ,instruction ,chunks ,available_output ,on_progress ,cancelled ,
    )
    findings ,reduce_truncated =await _run_reduce_phase (
    provider ,model ,instruction ,findings ,safe_input ,prompt_overhead ,
    available_output ,on_progress ,cancelled ,
    )

    return findings [0 ],total_chunks ,map_truncated +reduce_truncated 
