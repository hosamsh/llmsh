from __future__ import annotations 

from llmsh .models import ChatMessage 
from llmsh .providers .base import BaseProvider ,ProviderRequest 


def compact_messages (
messages :list [ChatMessage ],
keep_recent :int =6 ,
)->tuple [list [ChatMessage ],int ]:
    """Compact a message list by dropping middle messages.

    Returns (compacted_messages, num_dropped).

    Strategy:
    1. Always keep messages[0] if it's a system prompt
    2. Always keep the last `keep_recent` messages
    3. Drop everything in between
    4. Insert a marker message for the dropped portion
    """
    has_system =len (messages )>0 and messages [0 ].role =="system"
    prefix_len =1 if has_system else 0 

    if len (messages )<prefix_len +keep_recent +1 :
        return messages ,0 

    prefix =messages [:prefix_len ]
    recent =messages [-keep_recent :]
    num_dropped =len (messages )-prefix_len -keep_recent 

    marker =ChatMessage (
    role ="system",
    content =f"[{num_dropped } earlier messages removed to fit context window]",
    )
    return prefix +[marker ]+recent ,num_dropped 


async def summarize_dropped (
provider :BaseProvider ,
model :str ,
dropped_messages :list [ChatMessage ],
)->str :
    """Ask the LLM to summarize the dropped conversation segment."""
    conversation ="\n".join (
    f"{m .role }: {m .content }"for m in dropped_messages 
    )
    prompt =ChatMessage (
    role ="user",
    content =(
    "Summarize the following conversation concisely"
    " in a few sentences:\n\n"+conversation 
    ),
    )
    request =ProviderRequest (
    messages =[prompt ],
    model =model ,
    max_tokens =256 ,
    )
    result =await provider .chat (request )
    return result .content 
