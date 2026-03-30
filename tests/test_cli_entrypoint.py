import re 
import subprocess 
import sys 
from unittest .mock import patch 

from typer .testing import CliRunner 

runner =CliRunner ()


def _command_names_from_help ()->set [str ]:
    """Extract command names from the CLI help output."""
    result =subprocess .run (
    [sys .executable ,"-m","llmsh.cli","--help"],
    capture_output =True ,
    text =True ,
    )

    return set (re .findall (r"│\s+(\w+)\s{2,}",result .stdout ))


def test_llmsh_help_exits_zero ()->None :
    result =subprocess .run (
    [sys .executable ,"-m","llmsh.cli","--help"],
    capture_output =True ,
    )
    assert result .returncode ==0 


def test_ask_subcommand_in_help ()->None :
    commands =_command_names_from_help ()
    assert "ask"in commands 


def test_removed_commands_not_in_help ()->None :
    """Commands 'do' and 'config' were removed and must not appear in help."""
    commands =_command_names_from_help ()
    for removed in ("do","config"):
        assert removed not in commands ,f"'{removed }' should not appear in help"


def test_new_commands_in_help ()->None :
    """Commands added by tasks 077-080 must appear in help."""
    commands =_command_names_from_help ()
    for cmd in ("doctor","endpoint","profile","session"):
        assert cmd in commands ,f"'{cmd }' should appear in help"


def test_no_args_invokes_chat ()->None :
    from llmsh .cli import app 

    with patch ("llmsh.commands.chat.LlmshApp")as mock_app_cls :
        mock_app_cls .return_value .run .return_value =None 
        result =runner .invoke (app ,[])

    assert result .exit_code ==0 
    mock_app_cls .assert_called_once ()
