from __future__ import annotations 

import typer 

from llmsh .errors import SessionNotFoundError 
from llmsh .paths import sessions_dir 
from llmsh .sessions .store import SessionStore 

session_app =typer .Typer (help ="Manage saved sessions")


def _store ()->SessionStore :
    return SessionStore (sessions_dir ())


@session_app .command ("list")
def list_sessions ()->None :
    """List saved sessions."""
    store =_store ()
    sessions =store .list ()
    if not sessions :
        typer .echo ("No sessions found.")
        return 
    for meta in sessions :
        updated =meta .updated_at .strftime ("%Y-%m-%d %H:%M")
        typer .echo (f"  {meta .id [:8 ]}  {meta .title }  {meta .model }  {updated }")


@session_app .command ("show")
def show_session (session_id :str =typer .Argument (help ="Session ID or prefix"))->None :
    """Print a session transcript."""
    store =_store ()
    try :
        session =store .load (session_id )
    except SessionNotFoundError :
        typer .echo (f"Session not found: {session_id }",err =True )
        raise typer .Exit (code =1 )
    typer .echo (f"Session: {session .title } ({session .id [:8 ]})\n")
    for msg in session .messages :
        typer .echo (f"[{msg .role }] {msg .content }")


@session_app .command ("delete")
def delete_session (
session_id :str =typer .Argument (help ="Session ID or prefix"),
)->None :
    """Delete a saved session."""
    store =_store ()
    try :
        store .delete (session_id )
    except SessionNotFoundError :
        typer .echo (f"Session not found: {session_id }",err =True )
        raise typer .Exit (code =1 )
    typer .echo (f"Deleted session: {session_id }")
