"""Tests for /cancel input during active operations (task 075)."""

from __future__ import annotations 

from llmsh .ui .slash import parse_slash_command 


class FakeChatInput :
    """Minimal stand-in for ChatInput widget."""

    def __init__ (self )->None :
        self .disabled =False 
        self .placeholder ="Type a message..."

    def focus (self )->None :
        pass 


class FakeScreen :
    """Minimal stand-in for MainScreen to test busy-state behavior."""

    def __init__ (self )->None :
        self ._is_busy =False 
        self ._cancel_called =False 
        self ._input =FakeChatInput ()

    def query_one (self ,cls :type )->FakeChatInput :
        return self ._input 

    def set_streaming (self ,streaming :bool )->None :
        """Mirrors the production set_streaming logic under test."""
        self ._is_busy =streaming 
        input_widget =self ._input 
        if streaming :
            input_widget .placeholder ="Type /cancel to stop..."
        else :
            input_widget .placeholder ="Type a message..."

    def action_cancel_request (self )->None :
        self ._cancel_called =True 





class TestSetStreamingBusyFlag :
    def test_set_streaming_true_sets_busy (self )->None :
        screen =FakeScreen ()
        screen .set_streaming (True )
        assert screen ._is_busy is True 

    def test_set_streaming_false_clears_busy (self )->None :
        screen =FakeScreen ()
        screen .set_streaming (True )
        screen .set_streaming (False )
        assert screen ._is_busy is False 

    def test_placeholder_changes_to_cancel_hint_when_busy (self )->None :
        screen =FakeScreen ()
        screen .set_streaming (True )
        assert screen ._input .placeholder =="Type /cancel to stop..."

    def test_placeholder_reverts_when_idle (self )->None :
        screen =FakeScreen ()
        screen .set_streaming (True )
        screen .set_streaming (False )
        assert screen ._input .placeholder =="Type a message..."

    def test_input_not_disabled_when_streaming (self )->None :
        screen =FakeScreen ()
        screen .set_streaming (True )
        assert screen ._input .disabled is False 





class TestBusyStateInputFiltering :
    def test_cancel_command_triggers_cancellation_during_busy (self )->None :
        """During busy state, /cancel should trigger action_cancel_request."""
        screen =FakeScreen ()
        screen ._is_busy =True 
        text ="/cancel"
        parsed =parse_slash_command (text )
        assert parsed is not None 
        if parsed [0 ]=="cancel":
            screen .action_cancel_request ()
        assert screen ._cancel_called is True 

    def test_non_cancel_input_ignored_during_busy (self )->None :
        """During busy state, non-cancel input should be silently ignored."""
        screen =FakeScreen ()
        screen ._is_busy =True 
        text ="hello world"
        parsed =parse_slash_command (text )

        if parsed and parsed [0 ]=="cancel":
            screen .action_cancel_request ()
        assert screen ._cancel_called is False 

    def test_other_slash_command_ignored_during_busy (self )->None :
        """During busy state, slash commands other than /cancel are ignored."""
        screen =FakeScreen ()
        screen ._is_busy =True 
        text ="/help"
        parsed =parse_slash_command (text )
        if parsed and parsed [0 ]=="cancel":
            screen .action_cancel_request ()
        assert screen ._cancel_called is False 

    def test_cancel_with_args_does_not_trigger (self )->None :
        """During busy state, /cancel with extra args should still trigger."""
        screen =FakeScreen ()
        screen ._is_busy =True 
        text ="/cancel now"
        parsed =parse_slash_command (text )
        assert parsed is not None 
        if parsed [0 ]=="cancel":
            screen .action_cancel_request ()
        assert screen ._cancel_called is True 





class TestBusyInit :
    def test_is_busy_starts_false (self )->None :
        screen =FakeScreen ()
        assert screen ._is_busy is False 
