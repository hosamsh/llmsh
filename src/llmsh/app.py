from __future__ import annotations 

from dataclasses import dataclass 
from pathlib import Path 
from typing import AsyncIterator 

from llmsh .budget import ContextBudget 
from llmsh .compaction import compact_messages ,summarize_dropped 
from llmsh .config import (
AppConfig ,
get_active_profile ,
get_endpoint ,
load_config ,
resolve_api_key ,
)
from llmsh .errors import ProfileNotFoundError 
from llmsh .models import ChatMessage ,ProfileConfig 
from llmsh .providers .base import (
BaseProvider ,
BudgetWarningEvent ,
CancelledEvent ,
CompactionEvent ,
ErrorEvent ,
ProviderEvent ,
ProviderRequest ,
ResponseCompleted ,
TokenUsageEvent ,
ToolCallEvent ,
)
from llmsh .providers .openai_compatible import OpenAICompatibleProvider 
from llmsh .ui .slash import build_command_detection_tool ,get_command_names 


@dataclass 
class CommandSuggestionEvent :
    command :str 
    reasoning :str |None =None 


class AppCore :
    def __init__ (
    self ,
    config :AppConfig |None =None ,
    config_path :Path |None =None ,
    provider :BaseProvider |None =None ,
    system_prompt :str |None =None ,
    )->None :
        self ._config =config or load_config (config_path )
        self ._profile =get_active_profile (self ._config )
        self ._model =self ._profile .model 
        self ._messages :list [ChatMessage ]=[]
        if system_prompt is not None :
            self ._messages .append (ChatMessage (role ="system",content =system_prompt ))
        self ._provider =provider or self ._create_provider ()
        self ._budget :ContextBudget |None =None 
        self ._model_info_fetched =False 
        self ._cancelled =False 
        self ._got_usage =False 
        self .is_streaming =False 
        self .session =None 

    @property 
    def profile (self )->ProfileConfig :
        return self ._profile 

    @property 
    def model (self )->str :
        return self ._model 

    @property 
    def budget (self )->ContextBudget |None :
        return self ._budget 

    def _create_provider (self )->BaseProvider :
        endpoint =get_endpoint (self ._config ,self ._profile )
        api_key =resolve_api_key (endpoint )
        return OpenAICompatibleProvider (endpoint ,api_key )

    async def _init_budget (self ,max_tokens :int )->None :
        if self ._model_info_fetched :
            return 
        self ._model_info_fetched =True 
        try :
            info =await self ._provider .get_model_info (self ._model )
            if info .context_length is not None :
                self ._budget =ContextBudget (info .context_length ,max_tokens )
        except Exception :
            pass 

    def _preflight_check (
    self ,prompt :str 
    )->ErrorEvent |BudgetWarningEvent |None :
        """Check budget before sending. Returns an event to yield, or None."""
        if self ._budget is None :
            return None 
        estimated =self ._budget .estimate_tokens (prompt )
        if not self ._budget .would_fit (estimated ):
            remaining =self ._budget .remaining 
            return ErrorEvent (
            message =(
            f"Message too large for remaining context"
            f" ({remaining } tokens left)."
            " Use /clear to start fresh,"
            " or /save first to keep your history."
            ),
            error_type ="context_overflow",
            )
        if self ._budget .utilization >0.9 :
            remaining =self ._budget .remaining 
            pct =int (self ._budget .utilization *100 )
            return BudgetWarningEvent (
            utilization =self ._budget .utilization ,
            remaining_tokens =remaining ,
            message =(
            f"Context is {pct }% full"
            f" ({remaining } tokens remaining)."
            " Consider /save and /clear."
            ),
            )
        return None 

    def _classify_stream_error (self ,exc :Exception )->ErrorEvent :
        """Classify a streaming exception into an appropriate ErrorEvent."""
        if self ._budget is not None and self ._budget .utilization >0.8 :
            return ErrorEvent (
            message =(
            "The conversation may be too long"
            " for this model's context window."
            ),
            error_type ="context_overflow",
            )
        error_msg =str (exc ).lower ()
        total_chars =sum (len (m .content )for m in self ._messages )
        connection_keywords =(
        "closed connection","incomplete","reset","broken pipe"
        )
        context_overflow =total_chars >4000 and any (
        kw in error_msg for kw in connection_keywords 
        )
        if context_overflow :
            return ErrorEvent (
            message =(
            "The conversation may be too long"
            " for this model's context window."
            ),
            error_type ="context_overflow",
            )
        return ErrorEvent (
        message =f"Connection failed: {exc }",
        error_type ="connection",
        )

    async def _try_compact (self )->CompactionEvent |None :
        """Compact messages if budget utilization > 80%. Returns event or None."""
        if self ._budget is None or self ._budget .utilization <=0.8 :
            return None 
        original =list (self ._messages )
        compacted ,num_dropped =compact_messages (self ._messages )
        if num_dropped ==0 :
            return None 
        self ._messages =compacted 

        has_system =original [0 ].role =="system"if original else False 
        start =1 if has_system else 0 
        dropped =original [start :start +num_dropped ]
        try :
            summary =await summarize_dropped (
            self ._provider ,self ._model ,dropped 
            )
            marker =self ._messages [1 if has_system else 0 ]
            marker .content =(
            f"[Summary of {num_dropped } earlier messages]\n\n{summary }"
            )
            return CompactionEvent (
            messages_dropped =num_dropped ,
            message =f"[Summary of {num_dropped } earlier messages]",
            )
        except Exception :
            return CompactionEvent (
            messages_dropped =num_dropped ,
            message =(
            f"[{num_dropped } earlier messages removed"
            " to fit context window]"
            ),
            )

    def _build_request (self )->ProviderRequest :
        """Build the provider request from current state."""
        tools =None 
        if self ._profile .capabilities .tool_calling :
            tools =[build_command_detection_tool ()]
        max_tokens =self ._profile .max_tokens or 1024 
        return ProviderRequest (
        messages =list (self ._messages ),model =self ._model ,tools =tools ,
        max_tokens =max_tokens ,
        )

    def _process_stream_event (
    self ,event :ProviderEvent 
    )->ProviderEvent |CommandSuggestionEvent |None :
        """Process a single stream event. Returns event to yield, or None to skip."""
        if isinstance (event ,ToolCallEvent ):
            return self ._handle_tool_call (event )
        if isinstance (event ,TokenUsageEvent ):
            if self ._budget is not None and event .usage .input_tokens is not None :
                self ._budget .update_usage (event .usage .input_tokens )
                self ._got_usage =True 
        if isinstance (event ,ResponseCompleted ):
            self ._messages .append (
            ChatMessage (role ="assistant",content =event .content )
            )
        return event 

    def _estimate_usage_fallback (self )->None :
        """Estimate token usage from message history when provider didn't report it."""
        if self ._budget is None :
            return 
        estimated =sum (
        self ._budget .estimate_tokens (m .content )for m in self ._messages 
        )
        self ._budget .update_usage (estimated )

    def _pop_unanswered_user_message (self )->None :
        """Remove trailing user message if the assistant never responded."""
        if self ._messages and self ._messages [-1 ].role =="user":
            self ._messages .pop ()

    async def send_message (
    self ,prompt :str 
    )->AsyncIterator [ProviderEvent |CommandSuggestionEvent ]:
        self ._messages .append (ChatMessage (role ="user",content =prompt ))
        max_tokens =self ._profile .max_tokens or 1024 
        await self ._init_budget (max_tokens )

        preflight =self ._preflight_check (prompt )
        if isinstance (preflight ,ErrorEvent ):
            yield preflight 
            self ._messages .pop ()
            return 

        compaction =await self ._try_compact ()
        if compaction is not None :
            yield compaction 
        if isinstance (preflight ,BudgetWarningEvent ):
            yield preflight 
        request =self ._build_request ()
        self ._cancelled =False 
        self ._got_usage =False 
        self .is_streaming =True 
        got_response =False 
        try :
            async for event in self ._provider .stream_chat (request ):
                if self ._cancelled :
                    yield CancelledEvent ()
                    return 
                result =self ._process_stream_event (event )
                if result is not None :
                    yield result 
                if isinstance (event ,ResponseCompleted ):
                    got_response =True 
        except Exception as exc :
            yield self ._classify_stream_error (exc )
        finally :
            self .is_streaming =False 
            if not got_response :
                self ._pop_unanswered_user_message ()
        if got_response and not self ._got_usage :
            self ._estimate_usage_fallback ()

    def _handle_tool_call (self ,event :ToolCallEvent )->CommandSuggestionEvent |None :
        tc =event .tool_call 
        if tc .name !="suggest_slash_command":
            return None 
        command =tc .arguments .get ("command")
        if not isinstance (command ,str )or not command .startswith ("/"):
            return None 
        parts =command [1 :].split ()
        if not parts or parts [0 ]not in get_command_names ():
            return None 
        return CommandSuggestionEvent (
        command =command ,
        reasoning =tc .arguments .get ("reasoning"),
        )

    def cancel (self )->None :
        self ._cancelled =True 

    def clear_messages (self )->None :
        first =self ._messages [0 ]if self ._messages else None 
        system =first if first is not None and first .role =="system"else None 
        self ._messages .clear ()
        if system is not None :
            self ._messages .append (system )
        if self ._budget is not None :
            self ._budget .reset ()

    def switch_profile (self ,name :str )->None :
        profile =self ._config .profiles .get (name )
        if profile is None :
            raise ProfileNotFoundError (f"Profile not found: {name }")
        self ._profile =profile 
        self ._model =profile .model 
        self ._provider =self ._create_provider ()
        self ._budget =None 
        self ._model_info_fetched =False 
