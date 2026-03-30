from __future__ import annotations 

from pathlib import Path 
from typing import TYPE_CHECKING 

import llmsh .config as _config_mod 
from llmsh .config import get_endpoint ,save_config 
from llmsh .doctor import (
format_capability_summary ,
get_cached_report ,
is_cache_valid ,
run_doctor ,
)
from llmsh .endpoints import list_endpoints ,remove_endpoint 
from llmsh .errors import ConfigError ,EndpointNotFoundError ,ProfileNotFoundError 
from llmsh .models import ChatMessage ,ToolDefinition ,ToolParameter 
from llmsh .paths import sessions_dir 
from llmsh .profiles import remove_profile ,set_model ,use_profile 
from llmsh .sessions .store import SessionStore 

if TYPE_CHECKING :
    from llmsh .ui .screens import MainScreen 


def parse_slash_command (text :str )->tuple [str ,list [str ]]|None :
    text =text .strip ()
    if not text .startswith ("/")or text =="/":
        return None 
    parts =text [1 :].split ()
    return (parts [0 ],parts [1 :])


_COMMAND_NAMES =[
"help","clear","profile","session","save","load",
"copy","doctor","cancel","endpoint","retry",
"summarize",
]


def get_command_names ()->list [str ]:
    """Return the list of registered slash command names."""
    return list (_COMMAND_NAMES )


def build_command_detection_tool ()->ToolDefinition :
    """Build tool definition for slash command detection."""
    command_list =", ".join (f"/{name }"for name in _COMMAND_NAMES )
    return ToolDefinition (
    name ="suggest_slash_command",
    description =(
    "Call this tool when the user's message appears to be a slash command "
    "that is missing the leading '/'. The user may have forgotten to type '/' "
    f"before a command. Available commands: {command_list }. "
    "Only call this if you are reasonably confident the user meant to invoke "
    "a command, not when they are asking a question about commands."
    ),
    parameters =[
    ToolParameter (
    name ="command",
    type ="string",
    description =(
    "The full slash command including the leading '/',"
    " e.g. '/profile use myprofile'"
    ),
    ),
    ToolParameter (
    name ="reasoning",
    type ="string",
    description =(
    "Brief explanation of why you think this is"
    " a forgotten slash command"
    ),
    required =False ,
    ),
    ],
    )


async def handle_slash_command (
command :str ,args :list [str ],screen :MainScreen 
)->None :
    handlers ={
    "help":_handle_help ,
    "clear":_handle_clear ,
    "profile":_handle_profile ,
    "session":_handle_session ,
    "save":_handle_save ,
    "load":_handle_load ,
    "copy":_handle_copy ,
    "doctor":_handle_doctor ,
    "cancel":_handle_cancel ,
    "endpoint":_handle_endpoint ,
    "retry":_handle_retry ,
    "summarize":_handle_summarize ,
    }
    handler =handlers .get (command )
    if handler is None :
        await _show_message (
        screen ,
        f"Unknown command: /{command }. Type /help for available commands.",
        )
    else :
        await handler (args ,screen )


async def show_system_message (screen :MainScreen ,text :str )->None :
    from llmsh .ui .widgets import TranscriptPane 

    transcript =screen .query_one (TranscriptPane )
    await transcript .add_message (ChatMessage (role ="system",content =text ))


async def _show_message (screen :MainScreen ,text :str )->None :
    await show_system_message (screen ,text )


def _clear_transcript (screen :MainScreen )->None :
    from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

    transcript =screen .query_one (TranscriptPane )
    for widget in transcript .query (MessageWidget ):
        widget .remove ()


async def _handle_help (args :list [str ],screen :MainScreen )->None :
    text =(
    "**Commands**\n\n"
    "  **Chat**\n"
    "    /clear              — clear the transcript\n"
    "    /save               — save current session\n"
    "    /load [id]          — list sessions or load one\n"
    "    /session            — session management (list/save/load/delete)\n"
    "    /copy               — copy last assistant message\n"
    "    /retry              — re-send the last message\n\n"
    "  **Configuration**\n"
    "    /endpoint           — manage endpoints (list/add/remove)\n"
    "    /profile            — manage profiles (list/add/use/set-model/remove)\n"
    "    /doctor             — run diagnostics\n\n"
    "  **Analysis**\n"
    "    /summarize <path> <instruction> — analyze a file or directory\n\n"
    "  **Other**\n"
    "    /help               — show this help\n"
    "    /cancel             — cancel active flow or request"
    )
    await show_system_message (screen ,text )


async def _handle_clear (args :list [str ],screen :MainScreen )->None :
    _clear_transcript (screen )
    core =screen .app .core 
    if core is not None :
        core .clear_messages ()
        from llmsh .ui .screens import FooterBar 

        screen .query_one (FooterBar ).clear_budget ()



async def _handle_profile (args :list [str ],screen :MainScreen )->None :
    sub =args [0 ]if args else "list"
    handlers ={
    "list":_profile_list ,
    "add":_profile_add ,
    "use":_profile_use ,
    "set-model":_profile_set_model ,
    "remove":_profile_remove ,
    }
    handler =handlers .get (sub )
    if handler is None :
        await _show_message (
        screen ,
        "Usage: /profile [list|add|use <name>"
        "|set-model <name> <model>|remove <name>]",
        )
    else :
        await handler (args ,screen )


def _sync_config (core :object )->None :
    """Write the in-memory config to disk so shared module can read it."""
    save_config (core ._config )


async def _profile_list (args :list [str ],screen :MainScreen )->None :
    core =screen .app .core 
    if core is None :
        await _show_message (screen ,"No configuration found.")
        return 
    config =core ._config 
    lines =["**Profiles:**"]
    for name ,prof in config .profiles .items ():
        active =" <- active"if name ==config .current_profile else ""
        lines .append (
        f"  {name } — endpoint: {prof .endpoint }, model: {prof .model }{active }"
        )
    await _show_message (screen ,"\n".join (lines ))


async def _profile_add (args :list [str ],screen :MainScreen )->None :
    core =screen .app .core 
    if core is None :
        await _show_message (screen ,"No configuration found.")
        return 
    from llmsh .ui .flows .profile import ProfileAddFlow 

    await screen .start_flow (ProfileAddFlow (core ._config ))


async def _profile_use (args :list [str ],screen :MainScreen )->None :
    name =args [1 ]if len (args )>1 else None 
    if name is None :
        await _show_message (screen ,"Usage: /profile use <name>")
        return 
    core =screen .app .core 
    if core is None :
        await _show_message (screen ,"No configuration found.")
        return 
    try :
        _sync_config (core )
        use_profile (_config_mod .config_path (),name )
        core .switch_profile (name )
        core ._config .current_profile =name 
        _refresh_header (screen )
        from llmsh .ui .screens import FooterBar 

        footer =screen .query_one (FooterBar )
        footer .update_capabilities (None )
        footer .clear_budget ()
        await _show_message (screen ,f"Switched to profile: {name }")
    except ProfileNotFoundError :
        await _show_message (screen ,f"Profile not found: {name }")
        return 
    await _auto_doctor (screen )


async def _profile_set_model (args :list [str ],screen :MainScreen )->None :
    if len (args )<3 :
        await _show_message (screen ,"Usage: /profile set-model <name> <model>")
        return 
    name ,model_name =args [1 ],args [2 ]
    core =screen .app .core 
    if core is None :
        await _show_message (screen ,"No configuration found.")
        return 
    try :
        _sync_config (core )
        set_model (_config_mod .config_path (),name ,model_name )
    except ProfileNotFoundError :
        await _show_message (screen ,f"Profile not found: {name }")
        return 
    core ._config .profiles [name ].model =model_name 
    if name ==core ._config .current_profile :
        core ._model =model_name 
        _refresh_header (screen )
    await _show_message (screen ,f"Model for '{name }' set to: {model_name }")


async def _profile_remove (args :list [str ],screen :MainScreen )->None :
    name =args [1 ]if len (args )>1 else None 
    if name is None :
        await _show_message (screen ,"Usage: /profile remove <name>")
        return 
    core =screen .app .core 
    if core is None :
        await _show_message (screen ,"No configuration found.")
        return 
    try :
        _sync_config (core )
        remove_profile (_config_mod .config_path (),name )
    except ProfileNotFoundError :
        await _show_message (screen ,f"Profile not found: {name }")
        return 
    except ConfigError :
        await _show_message (
        screen ,
        f"Cannot remove active profile '{name }'."
        " Switch to another profile first.",
        )
        return 
    del core ._config .profiles [name ]
    await _show_message (screen ,f"Profile '{name }' removed.")


async def _handle_save (args :list [str ],screen :MainScreen )->None :
    core =screen .app .core 
    store =SessionStore (sessions_dir ())
    session =store .create (profile =core .profile .name ,model =core .model )
    session .messages =list (core ._messages )
    store .save (session )
    await _show_message (screen ,f"Session saved: {session .title } ({session .id [:8 ]})")


async def _handle_load (args :list [str ],screen :MainScreen )->None :
    store =SessionStore (sessions_dir ())

    if not args :
        sessions =store .list ()
        if not sessions :
            await _show_message (screen ,"No sessions found.")
            return 
        lines =["**Recent sessions** (use `/load <id>` to restore):\n"]
        for meta in sessions :
            updated =meta .updated_at .strftime ("%Y-%m-%d %H:%M")
            lines .append (f"- `{meta .id [:8 ]}`  {meta .title }  *{meta .model }*  {updated }")
        await _show_message (screen ,"\n".join (lines ))
        return 

    session_id =args [0 ]
    try :
        session =store .load (session_id )
    except Exception :
        await _show_message (screen ,f"Session not found: {session_id }")
        return 

    core =screen .app .core 
    core ._messages .clear ()
    _clear_transcript (screen )

    from llmsh .ui .widgets import TranscriptPane 

    transcript =screen .query_one (TranscriptPane )
    for msg in session .messages :
        core ._messages .append (msg )
        await transcript .add_message (msg )

    await _show_message (screen ,f"Loaded session: {session .title }")


async def _handle_session (args :list [str ],screen :MainScreen )->None :
    sub =args [0 ]if args else "list"

    if sub =="list":
        await _handle_load ([],screen )
    elif sub =="save":
        await _handle_save (args [1 :],screen )
    elif sub =="load":
        await _handle_load (args [1 :],screen )
    elif sub =="delete":
        if len (args )<2 :
            await _show_message (screen ,"Usage: /session delete <id>")
            return 
        session_id =args [1 ]
        store =SessionStore (sessions_dir ())
        try :
            store .delete (session_id )
            await _show_message (screen ,f"Session deleted: {session_id }")
        except Exception :
            await _show_message (screen ,f"Session not found: {session_id }")
    else :
        await _show_message (
        screen ,"Usage: /session [list|save|load <id>|delete <id>]"
        )


async def _handle_copy (args :list [str ],screen :MainScreen )->None :
    await _show_message (screen ,"Clipboard support coming soon.")


async def _handle_doctor (args :list [str ],screen :MainScreen )->None :
    core =screen .app .core 

    if args and args [0 ]=="test":
        await _handle_doctor_test (args [1 :],screen )
        return 

    endpoint =get_endpoint (core ._config ,core .profile )
    report =await run_doctor (core .profile ,endpoint ,core ._provider )
    lines =["Doctor results:"]
    for check in report .checks :
        status ="PASS"if check .passed else "FAIL"
        lines .append (f"  [{status }] {check .name }: {check .message }")
    await _show_message (screen ,"\n".join (lines ))


async def _handle_doctor_test (args :list [str ],screen :MainScreen )->None :
    from llmsh .ui .screens import FooterBar 

    core =screen .app .core 
    config =core ._config 

    if args :
        profile_name =args [0 ]
        if profile_name not in config .profiles :
            await _show_message (screen ,f"Profile not found: {profile_name }")
            return 
    else :
        profile_name =core .profile .name 

    is_active =profile_name ==core .profile .name 

    footer =screen .query_one (FooterBar )
    footer .set_testing ()
    await _show_message (
    screen ,f'Running capability tests for profile "{profile_name }"...'
    )
    screen .set_streaming (True )
    screen ._run_doctor_test (profile_name ,is_active )


async def _handle_endpoint (args :list [str ],screen :MainScreen )->None :
    sub =args [0 ]if args else "list"
    handlers ={
    "list":_endpoint_list ,
    "add":_endpoint_add ,
    "remove":_endpoint_remove ,
    }
    handler =handlers .get (sub )
    if handler is None :
        await _show_message (screen ,"Usage: /endpoint [list|add|remove <name>]")
    else :
        await handler (args ,screen )


async def _endpoint_list (args :list [str ],screen :MainScreen )->None :
    core =screen .app .core 
    if core is None :
        await _show_message (screen ,"No configuration found. Run setup first.")
        return 
    _sync_config (core )
    entries =list_endpoints (_config_mod .config_path ())
    if not entries :
        await _show_message (screen ,"No endpoints configured.")
        return 
    lines =["**Endpoints:**"]
    for entry in entries :
        lines .append (
        f"  {entry ['name']} — {entry ['base_url']} (auth: {entry ['auth_mode']})"
        )
    await _show_message (screen ,"\n".join (lines ))


async def _endpoint_add (args :list [str ],screen :MainScreen )->None :
    core =screen .app .core 
    if core is None :
        await _show_message (screen ,"No configuration found. Run setup first.")
        return 
    from llmsh .ui .flows .endpoint import EndpointAddFlow 

    await screen .start_flow (EndpointAddFlow (core ._config ))


async def _endpoint_remove (args :list [str ],screen :MainScreen )->None :
    name =args [1 ]if len (args )>1 else None 
    if name is None :
        await _show_message (screen ,"Usage: /endpoint remove <name>")
        return 
    core =screen .app .core 
    if core is None :
        await _show_message (screen ,"No configuration found.")
        return 
    try :
        _sync_config (core )
        referencing =remove_endpoint (_config_mod .config_path (),name )
    except EndpointNotFoundError :
        await _show_message (screen ,f"Endpoint not found: {name }")
        return 
    del core ._config .endpoints [name ]
    msg =f"Endpoint '{name }' removed."
    if referencing :
        msg +=f" Warning: profiles still referencing it: {', '.join (referencing )}"
    await _show_message (screen ,msg )


async def _handle_summarize (args :list [str ],screen :MainScreen )->None :
    if len (args )<2 :
        await show_system_message (
        screen ,"Usage: /summarize <file_or_dir> <instruction>"
        )
        return 

    path =Path (args [0 ])
    if not path .exists ():
        await show_system_message (screen ,f"Not found: {path }")
        return 

    instruction =" ".join (args [1 :])
    screen .set_streaming (True )
    screen ._run_summarize (str (path ),instruction )


async def _handle_retry (args :list [str ],screen :MainScreen )->None :
    core =screen .app .core 
    if core is None :
        await show_system_message (screen ,"No configuration found.")
        return 

    from llmsh .ui .widgets import MessageWidget ,TranscriptPane 

    transcript =screen .query_one (TranscriptPane )
    last_user_text =None 
    for w in reversed (list (transcript .query (MessageWidget ))):
        if w ._role =="user"and not w ._content .startswith ("/"):
            last_user_text =w ._content 
            break 

    if last_user_text is None :
        await show_system_message (screen ,"Nothing to retry.")
        return 

    await show_system_message (screen ,"Retrying...")
    screen .set_streaming (True )
    screen ._stream_response (last_user_text )


async def _handle_cancel (args :list [str ],screen :MainScreen )->None :
    if screen ._active_flow is not None :
        await screen ._active_flow .cancel (screen )
        screen .end_flow ()
        await show_system_message (screen ,"Cancelled.")
    else :
        core =getattr (screen .app ,"core",None )
        if core is not None and core .is_streaming :
            core .cancel ()
            await show_system_message (screen ,"Cancelled.")
        else :
            await show_system_message (screen ,"Nothing to cancel.")


async def _auto_doctor (screen :MainScreen )->None :
    """Run capability tests after a profile switch."""
    from llmsh .ui .screens import FooterBar 

    core =screen .app .core 
    profile =core .profile 
    footer =screen .query_one (FooterBar )


    if is_cache_valid (profile ):
        report =get_cached_report (profile .name )
        if report is not None :
            summary =format_capability_summary (report )
            await _show_message (screen ,f"Capabilities: {summary }")
            footer .update_capabilities (report )
            return 


    footer .set_testing ()
    screen .set_streaming (True )
    screen ._run_auto_doctor ()


def _refresh_header (screen :MainScreen )->None :
    from llmsh .ui .screens import HeaderBar 

    core =screen .app .core 
    screen .query_one (HeaderBar ).refresh_display (
    profile =core .profile .name ,
    model =core .model ,
    )
