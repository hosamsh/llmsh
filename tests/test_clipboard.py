"""Tests for clipboard copy functionality."""

import subprocess 

import pytest 

from llmsh .errors import ClipboardError 


def _raise (exc :Exception )->None :
    raise exc 


def test_copy_to_clipboard_succeeds_with_pyperclip (
monkeypatch :pytest .MonkeyPatch ,
)->None :
    """copy_to_clipboard returns normally when pyperclip.copy succeeds."""
    copied :list [str ]=[]

    import pyperclip 

    monkeypatch .setattr (pyperclip ,"copy",lambda text :copied .append (text ))

    from llmsh .clipboard import copy_to_clipboard 

    copy_to_clipboard ("hello world")

    assert copied ==["hello world"]


def test_copy_to_clipboard_raises_when_all_backends_fail (
monkeypatch :pytest .MonkeyPatch ,
)->None :
    """copy_to_clipboard raises ClipboardError when all backends fail."""
    import pyperclip 

    monkeypatch .setattr (pyperclip ,"copy",lambda t :_raise (Exception ("no clipboard")))
    monkeypatch .setattr (subprocess ,"run",lambda *a ,**kw :_raise (FileNotFoundError ()))

    from llmsh .clipboard import copy_to_clipboard 

    with pytest .raises (ClipboardError ):
        copy_to_clipboard ("hello world")


def test_clipboard_error_message_is_informative (
monkeypatch :pytest .MonkeyPatch ,
)->None :
    """ClipboardError message contains 'clipboard' and names at least one backend."""
    import pyperclip 

    monkeypatch .setattr (pyperclip ,"copy",lambda t :_raise (Exception ("no clipboard")))
    monkeypatch .setattr (subprocess ,"run",lambda *a ,**kw :_raise (FileNotFoundError ()))

    from llmsh .clipboard import copy_to_clipboard 

    with pytest .raises (ClipboardError )as exc_info :
        copy_to_clipboard ("hello world")

    message =str (exc_info .value ).lower ()
    assert "clipboard"in message 
    backends =["pyperclip","wl-copy","xclip","xsel"]
    assert any (backend in message for backend in backends )
