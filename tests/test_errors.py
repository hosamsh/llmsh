import pytest 

from llmsh import errors 


def test_llmsh_error_is_base_exception ()->None :
    assert issubclass (errors .LlmshError ,Exception )


def test_config_error_inherits_llmsh_error ()->None :
    assert issubclass (errors .ConfigError ,errors .LlmshError )


def test_profile_not_found_error_inherits_llmsh_error ()->None :
    assert issubclass (errors .ProfileNotFoundError ,errors .LlmshError )


def test_endpoint_not_reachable_error_inherits_llmsh_error ()->None :
    assert issubclass (errors .EndpointNotReachableError ,errors .LlmshError )


def test_authentication_error_inherits_llmsh_error ()->None :
    assert issubclass (errors .AuthenticationError ,errors .LlmshError )


def test_model_not_found_error_inherits_llmsh_error ()->None :
    assert issubclass (errors .ModelNotFoundError ,errors .LlmshError )


def test_stream_interrupted_error_inherits_llmsh_error ()->None :
    assert issubclass (errors .StreamInterruptedError ,errors .LlmshError )


def test_clipboard_error_inherits_llmsh_error ()->None :
    assert issubclass (errors .ClipboardError ,errors .LlmshError )


def test_session_not_found_error_inherits_llmsh_error ()->None :
    assert issubclass (errors .SessionNotFoundError ,errors .LlmshError )


def test_errors_can_be_raised_and_caught_as_llmsh_error ()->None :
    with pytest .raises (errors .LlmshError ):
        raise errors .ConfigError ("bad config")


def test_errors_carry_message ()->None :
    exc =errors .ProfileNotFoundError ("my-profile")
    assert "my-profile"in str (exc )
