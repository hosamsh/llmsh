from __future__ import annotations 

import asyncio 
from datetime import UTC ,datetime 

from textual import work 
from textual .app import ComposeResult 
from textual .binding import Binding 
from textual .screen import Screen 
from textual .widgets import Static 

from llmsh .app import CommandSuggestionEvent 
from llmsh .models import CapabilityReport ,ChatMessage 
from llmsh .paths import sessions_dir 
from llmsh .providers .base import (
BudgetWarningEvent ,
CancelledEvent ,
CompactionEvent ,
ErrorEvent ,
ReasoningDelta ,
TextDelta ,
TokenUsageEvent ,
)
from llmsh .sessions .store import SessionStore 
from llmsh .ui .flow import InteractiveFlow 
from llmsh .ui .slash import (
handle_slash_command ,
parse_slash_command ,
show_system_message ,
)
from llmsh .ui .widgets import ChatInput ,SubmitMessage ,TranscriptPane 


class HeaderBar (Static ):
    DEFAULT_CSS ="""
    HeaderBar {
        height: 1;
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    """

    def __init__ (self ,profile :str ,model :str )->None :
        super ().__init__ ()
        self ._profile =profile 
        self ._model =model 

    def on_mount (self )->None :
        self .refresh_display (self ._profile ,self ._model )

    def refresh_display (self ,profile :str ,model :str )->None :
        self ._profile =profile 
        self ._model =model 
        self .update (f"llmsh  profile:{profile }  model:{model }")


class FooterBar (Static ):
    DEFAULT_CSS ="""
    FooterBar {
        height: 1;
        background: $primary-darken-2;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__ (self )->None :
        super ().__init__ ("Ctrl+C cancel  /help commands")
        self ._capabilities :str =""
        self ._tokens :str =""
        self ._budget_text :str =""

    def on_mount (self )->None :
        self ._refresh_footer ()

    def update_capabilities (self ,report :CapabilityReport |None )->None :
        if report is None :
            self ._capabilities =""
        else :
            from llmsh .doctor import format_capability_summary 

            self ._capabilities =format_capability_summary (report )
        self ._refresh_footer ()

    def set_testing (self )->None :
        self ._capabilities ="testing..."
        self ._refresh_footer ()

    def show_tokens (self ,total :int |str )->None :
        self ._tokens =f"tokens: {total }"
        self ._refresh_footer ()

    def show_budget (self ,utilization :float )->None :
        self ._budget_text =f"budget: {int (utilization *100 )}%"
        self ._refresh_footer ()

    def clear_budget (self )->None :
        self ._budget_text =""
        self ._refresh_footer ()

    def _render_text (self )->str :
        base ="Ctrl+C cancel  /help commands"
        parts =[base ]
        if self ._capabilities :
            parts .append (self ._capabilities )
        if self ._tokens :
            parts .append (self ._tokens )
        if self ._budget_text :
            parts .append (self ._budget_text )
        return "    ".join (parts )

    def _refresh_footer (self )->None :
        self .update (self ._render_text ())


class MainScreen (Screen ):
    BINDINGS =[Binding ("escape","cancel_request","Cancel")]

    def __init__ (self )->None :
        super ().__init__ ()
        self ._active_flow :InteractiveFlow |None =None 
        self ._summarize_cancelled =False 
        self ._is_busy =False 

    def compose (self )->ComposeResult :
        core =self .app .core 
        profile_name =core .profile .name if core else "none"
        model_name =core .model if core else "none"
        yield HeaderBar (profile =profile_name ,model =model_name )
        yield TranscriptPane (id ="transcript")
        yield ChatInput (placeholder ="Type a message...",id ="input")
        yield FooterBar ()

    def on_mount (self )->None :
        self .query_one (ChatInput ).focus ()
        if self .app .core is None :
            self .app .call_later (self ._start_onboarding )

    async def _start_onboarding (self )->None :
        from llmsh .ui .flows .setup import SetupFlow 

        await self .start_flow (SetupFlow ())

    async def start_flow (self ,flow :InteractiveFlow )->None :
        self ._active_flow =flow 
        await flow .start (self )
        self .query_one (ChatInput ).placeholder =flow .placeholder 

    def end_flow (self )->None :
        self ._active_flow =None 
        self .query_one (ChatInput ).placeholder ="Type a message..."
        self .query_one (ChatInput ).focus ()

    def action_cancel_request (self )->None :
        if self ._active_flow is not None :
            self .app .call_later (self ._cancel_active_flow )
        elif self .app .core is not None and self .app .core .is_streaming :
            self .app .core .cancel ()
        else :
            self ._summarize_cancelled =True 

    async def _cancel_active_flow (self )->None :
        if self ._active_flow is not None :
            await self ._active_flow .cancel (self )
            self .end_flow ()

    def set_streaming (self ,streaming :bool )->None :
        self ._is_busy =streaming 
        input_widget =self .query_one (ChatInput )
        input_widget .disabled =streaming 
        if streaming :
            input_widget .placeholder ="Type /cancel to stop..."
        else :
            input_widget .placeholder ="Type a message..."

    async def on_submit_message (self ,event :SubmitMessage )->None :
        if self ._is_busy :
            parsed =parse_slash_command (event .text )
            if parsed and parsed [0 ]=="cancel":
                self .action_cancel_request ()
            return 

        transcript =self .query_one (TranscriptPane )
        await transcript .add_message (ChatMessage (role ="user",content =event .text ))

        parsed =parse_slash_command (event .text )
        if parsed is not None :
            command ,args =parsed 
            await handle_slash_command (command ,args ,self )
            return 

        if self ._active_flow is not None :
            done =await self ._active_flow .handle_input (event .text ,self )
            if done :
                self .end_flow ()
            else :
                self .query_one (ChatInput ).placeholder =self ._active_flow .placeholder 
            return 

        if self .app .core is None :
            await show_system_message (
            self ,
            "No configuration found. Type /endpoint add to set one up.",
            )
            return 

        self .set_streaming (True )
        self ._stream_response (event .text )

    @work (thread =True ,exclusive =True )
    def _stream_response (self ,text :str )->None :
        core =self .app .core 
        transcript =self .query_one (TranscriptPane )
        state ={"in_reasoning":False ,"streaming_started":False }
        loop =asyncio .new_event_loop ()
        try :
            gen =core .send_message (text )
            while True :
                try :
                    event =loop .run_until_complete (gen .__anext__ ())
                except StopAsyncIteration :
                    break 
                action =self ._dispatch_stream_event (event ,transcript ,core ,state )
                if action =="stop":
                    return 
                if action =="break":
                    break 
        finally :
            loop .close ()
            self .app .call_from_thread (self ._finish_streaming )

    def _dispatch_stream_event (
    self ,event :object ,transcript :TranscriptPane ,
    core :object ,state :dict [str ,bool ],
    )->str |None :
        if isinstance (event ,ReasoningDelta ):
            if not state ["in_reasoning"]:
                state ["in_reasoning"]=True 
                self .app .call_from_thread (transcript .begin_reasoning )
            self .app .call_from_thread (transcript .append_reasoning ,event .text )
        elif isinstance (event ,TextDelta ):
            if not state ["streaming_started"]:
                state ["streaming_started"]=True 
                if state ["in_reasoning"]:
                    state ["in_reasoning"]=False 
                    self .app .call_from_thread (transcript .end_reasoning )
                self .app .call_from_thread (transcript .begin_streaming )
            self .app .call_from_thread (transcript .append_text ,event .text )
        elif isinstance (event ,TokenUsageEvent ):
            self .app .call_from_thread (
            self .query_one (FooterBar ).show_tokens ,
            event .usage .total_tokens or "?",
            )
            if core .budget is not None :
                self .app .call_from_thread (
                self .query_one (FooterBar ).show_budget ,
                core .budget .utilization ,
                )
        elif isinstance (event ,(BudgetWarningEvent ,CompactionEvent )):
            self .app .call_from_thread (self ._post_system_message ,event .message )
        elif isinstance (event ,ErrorEvent ):
            self .app .call_from_thread (transcript .end_streaming )
            self .app .call_from_thread (
            self ._add_error_message ,event .message ,event .error_type 
            )
            return "stop"
        elif isinstance (event ,CommandSuggestionEvent ):
            self .app .call_from_thread (transcript .end_streaming )
            self .app .call_from_thread (
            self ._start_command_confirm ,event .command ,event .reasoning ,
            )
            return "stop"
        elif isinstance (event ,CancelledEvent ):
            return "break"
        return None 

    def _finish_streaming (self )->None :
        self .query_one (TranscriptPane ).end_streaming ()
        self .set_streaming (False )
        self .query_one (ChatInput ).focus ()
        core =self .app .core 
        if core is not None and core .budget is not None :
            self .query_one (FooterBar ).show_budget (core .budget .utilization )
        self ._auto_save ()

    def _start_command_confirm (
    self ,command :str ,reasoning :str |None 
    )->None :
        from llmsh .ui .flows .command_confirm import CommandConfirmFlow 

        self .set_streaming (False )
        self .app .call_later (
        self .start_flow ,CommandConfirmFlow (command ,reasoning )
        )

    def _auto_save (self )->None :
        core =self .app .core 
        if core is None or not core ._messages :
            return 
        if core ._messages [-1 ].role !="assistant":
            return 
        try :
            store =SessionStore (sessions_dir ())
            if core .session is None :
                core .session =store .create (
                profile =core .profile .name ,model =core .model 
                )
            core .session .messages =list (core ._messages )
            core .session .updated_at =datetime .now (UTC )
            store .save (core .session )
        except Exception :
            pass 





    @work (thread =True ,exclusive =True )
    def _run_summarize (self ,path_str :str ,instruction :str )->None :
        """Run summarize in a background thread."""
        from pathlib import Path 

        from llmsh .commands .summarize import _read_directory 
        from llmsh .errors import LlmshError 
        from llmsh .summarize import summarize_file 

        self ._summarize_cancelled =False 
        loop =asyncio .new_event_loop ()
        try :
            path =Path (path_str )
            if path .is_dir ():
                file_text =_read_directory (path )
            else :
                file_text =path .read_text ()

            core =self .app .core 
            provider =core ._provider 
            model =core ._model 

            try :
                info =loop .run_until_complete (provider .get_model_info (model ))
                context_length =info .context_length or 4096 
            except Exception :
                context_length =4096 

            max_output_tokens =core ._profile .max_tokens or 1024 

            def on_progress (phase :str ,current :int ,total :int )->None :
                if phase =="map":
                    msg =f"Summarizing: chunk {current }/{total }..."
                elif phase .startswith ("reduce-"):
                    round_num =phase .split ("-",1 )[1 ]
                    msg =f"Reducing round {round_num }: batch {current }/{total }..."
                else :
                    return 
                self .app .call_from_thread (self ._post_system_message ,msg )

            answer ,_chunks ,truncated =loop .run_until_complete (
            summarize_file (
            provider =provider ,
            model =model ,
            instruction =instruction ,
            file_text =file_text ,
            context_length =context_length ,
            max_output_tokens =max_output_tokens ,
            on_progress =on_progress ,
            cancelled =lambda :self ._summarize_cancelled ,
            )
            )

            result_parts :list [str ]=[]
            if truncated >0 :
                result_parts .append (f"*{truncated } response(s) were truncated*\n")
            result_parts .append (answer )
            self .app .call_from_thread (
            self ._post_system_message ,"".join (result_parts )
            )
        except LlmshError as exc :
            self .app .call_from_thread (self ._post_system_message ,str (exc ))
        except Exception as exc :
            self .app .call_from_thread (
            self ._post_system_message ,f"Error: {exc }"
            )
        finally :
            loop .close ()
            self .app .call_from_thread (self .set_streaming ,False )
            self .app .call_from_thread (self .query_one (ChatInput ).focus )





    @work (thread =True ,exclusive =True )
    def _run_doctor_test (
    self ,
    profile_name :str ,
    is_active_profile :bool ,
    )->None :
        """Run capability tests in a background thread."""
        from llmsh .config import get_endpoint ,resolve_api_key 
        from llmsh .doctor import run_capability_tests 

        core =self .app .core 
        config =core ._config 
        profile =config .profiles [profile_name ]
        endpoint =get_endpoint (config ,profile )

        if profile .name !=core .profile .name :
            from llmsh .providers .openai_compatible import OpenAICompatibleProvider 

            provider =OpenAICompatibleProvider (endpoint ,resolve_api_key (endpoint ))
        else :
            provider =core ._provider 

        loop =asyncio .new_event_loop ()
        try :
            report =loop .run_until_complete (
            run_capability_tests (profile ,endpoint ,provider )
            )
            self .app .call_from_thread (
            self ._show_doctor_results ,report ,profile .name ,is_active_profile 
            )
        except Exception :
            self .app .call_from_thread (self ._show_doctor_error )
        finally :
            loop .close ()
            self .app .call_from_thread (self .set_streaming ,False )
            self .app .call_from_thread (self .query_one (ChatInput ).focus )

    def _show_doctor_results (
    self ,
    report :CapabilityReport ,
    profile_name :str ,
    is_active_profile :bool ,
    )->None :
        """Display capability test results (called on main thread)."""
        from llmsh .doctor import get_cached_report 

        lines =[f'Profile "{profile_name }" capability test:']
        for result in report .results :
            if result .passed is None :
                status ="DISABLED"
            elif result .passed :
                status ="PASS"
            else :
                status ="FAIL"
            timing =(
            f"  ({result .duration_ms }ms)"
            if result .duration_ms is not None 
            else ""
            )
            lines .append (
            f"  {result .name :<15}{status }  {result .message }{timing }"
            )
        self ._post_system_message ("\n".join (lines ))

        footer =self .query_one (FooterBar )
        if is_active_profile :
            footer .update_capabilities (report )
        else :
            active_report =get_cached_report (
            self .app .core .profile .name 
            )
            footer .update_capabilities (active_report )

    def _show_doctor_error (self )->None :
        """Display doctor test error (called on main thread)."""
        self ._post_system_message (
        "Capability test failed \u2014 run `/doctor test` to retry"
        )

    @work (thread =True ,exclusive =True )
    def _run_auto_doctor (self )->None :
        """Run auto-doctor capability tests in a background thread."""
        from llmsh .config import get_endpoint 
        from llmsh .doctor import run_capability_tests 

        core =self .app .core 
        profile =core .profile 
        endpoint =get_endpoint (core ._config ,profile )

        loop =asyncio .new_event_loop ()
        try :
            report =loop .run_until_complete (
            run_capability_tests (profile ,endpoint ,core ._provider )
            )
            self .app .call_from_thread (self ._show_auto_doctor_results ,report )
        except Exception :
            self .app .call_from_thread (self ._show_auto_doctor_error )
        finally :
            loop .close ()
            self .app .call_from_thread (self .set_streaming ,False )
            self .app .call_from_thread (self .query_one (ChatInput ).focus )

    def _show_auto_doctor_results (self ,report :CapabilityReport )->None :
        """Display auto-doctor results (called on main thread)."""
        from llmsh .doctor import format_capability_summary 

        summary =format_capability_summary (report )
        self ._post_system_message (f"Capabilities: {summary }")
        self .query_one (FooterBar ).update_capabilities (report )

        tool_result =next (
        (r for r in report .results if r .name =="tool_calling"),None ,
        )
        if tool_result and tool_result .passed is False :
            self ._post_system_message (
            "Note: Tool calling not supported"
            " \u2014 slash command detection disabled.",
            )

    def _show_auto_doctor_error (self )->None :
        """Display auto-doctor error (called on main thread)."""
        self .query_one (FooterBar ).update_capabilities (None )
        self ._post_system_message (
        "Capability test failed \u2014 run `/doctor test` to retry"
        )

    def _post_system_message (self ,text :str )->None :
        """Post a system message to the transcript from the main thread."""
        transcript =self .query_one (TranscriptPane )
        self .app .call_later (
        transcript .add_message ,
        ChatMessage (role ="system",content =text ),
        )

    def _add_error_message (self ,message :str ,error_type :str ="unknown")->None :
        if error_type =="model_not_found":
            hint =(
            f"{message }\n\n"
            "Use /profile set-model <profile> <model> to change the model, "
            "or /profile use <name> to switch profiles."
            )
        elif error_type =="auth":
            hint =(
            f"{message }\n\n"
            "Use /endpoint list to verify your endpoint settings."
            )
        elif error_type =="connection":
            hint =(
            f"{message }\n\n"
            "Use /doctor to check connectivity, or "
            "/endpoint list to verify your endpoint URL."
            )
        elif error_type =="context_overflow":
            hint =(
            f"{message }\n\n"
            "Use /clear to start fresh, or /save first to keep your history."
            )
        else :
            hint =(
            f"{message }\n\n"
            "Use /doctor to run diagnostics."
            )
        transcript =self .query_one (TranscriptPane )
        self .app .call_later (
        transcript .add_message ,
        ChatMessage (role ="system",content =hint ),
        )
