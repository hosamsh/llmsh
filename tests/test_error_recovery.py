"""Tests for friendly error recovery with guided fix (task 041)."""

from __future__ import annotations 

import json 

import httpx 
import pytest 
import respx 

from llmsh .models import ChatMessage ,EndpointConfig 
from llmsh .providers .base import ErrorEvent ,ProviderRequest 
from llmsh .providers .openai_compatible import (
OpenAICompatibleProvider ,
_classify_http_error ,
_parse_error_body ,
)





API_KEY ="test-key-123"
BASE_URL ="http://localhost:8006/v1"


def _endpoint ()->EndpointConfig :
    return EndpointConfig (
    name ="test",
    base_url =BASE_URL ,
    auth_mode ="api_key",
    provider_type ="openai_compatible",
    )


def _request ()->ProviderRequest :
    return ProviderRequest (
    messages =[ChatMessage (role ="user",content ="hello")],
    model ="test-model",
    )


def _openai_error_body (message :str ,code :str ="MODEL_NOT_FOUND")->dict :
    return {"error":{"code":code ,"message":message ,"details":{}}}







class TestParseErrorBody :
    def test_extracts_openai_style_message (self ):
        body =json .dumps (_openai_error_body ("Model not found: model-b"))
        result =_parse_error_body (body ,404 )
        assert result =="Model not found: model-b"

    def test_extracts_string_error_field (self ):
        body =json .dumps ({"error":"unauthorized"})
        result =_parse_error_body (body ,401 )
        assert result =="unauthorized"

    def test_non_json_body_with_short_text (self ):
        result =_parse_error_body ("Internal Server Error",500 )
        assert "500"in result 
        assert "Internal Server Error"in result 

    def test_non_json_empty_body_falls_back_gracefully (self ):
        result =_parse_error_body ("",503 )
        assert "503"in result 

    def test_long_body_uses_status_code_fallback (self ):
        long_body ="x"*201 
        result =_parse_error_body (long_body ,400 )
        assert "400"in result 
        assert long_body not in result 

    def test_invalid_json_falls_back (self ):
        result =_parse_error_body ("{not valid json}",400 )
        assert "400"in result 


class TestClassifyHttpError :
    def test_401_returns_auth (self ):
        assert _classify_http_error (401 ,"")=="auth"

    def test_403_returns_auth (self ):
        assert _classify_http_error (403 ,"")=="auth"

    def test_404_with_model_in_body_returns_model_not_found (self ):
        body =json .dumps (_openai_error_body ("Model not found: x"))
        assert _classify_http_error (404 ,body )=="model_not_found"

    def test_404_with_not_found_text_returns_model_not_found (self ):
        assert _classify_http_error (404 ,"model not found")=="model_not_found"

    def test_404_without_model_returns_not_found (self ):
        assert _classify_http_error (404 ,"resource missing")=="not_found"

    def test_500_returns_server_error (self ):
        assert _classify_http_error (500 ,"")=="server_error"

    def test_503_returns_server_error (self ):
        assert _classify_http_error (503 ,"")=="server_error"

    def test_other_returns_unknown (self ):
        assert _classify_http_error (429 ,"")=="unknown"







class TestProviderErrorTypes :
    @pytest .mark .anyio 
    @respx .mock 
    async def test_401_yields_auth_error_type (self ):
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (401 ,json ={"error":"unauthorized"})
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request ()):
            events .append (event )

        assert len (events )==1 
        assert isinstance (events [0 ],ErrorEvent )
        assert events [0 ].error_type =="auth"

    @pytest .mark .anyio 
    @respx .mock 
    async def test_403_yields_auth_error_type (self ):
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (403 ,json ={"error":"forbidden"})
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request ()):
            events .append (event )

        assert len (events )==1 
        assert isinstance (events [0 ],ErrorEvent )
        assert events [0 ].error_type =="auth"

    @pytest .mark .anyio 
    @respx .mock 
    async def test_404_model_not_found_yields_correct_type_and_message (self ):
        body =_openai_error_body ("Model not found: model-b",code ="MODEL_NOT_FOUND")
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (404 ,json =body )
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request ()):
            events .append (event )

        assert len (events )==1 
        err =events [0 ]
        assert isinstance (err ,ErrorEvent )
        assert err .error_type =="model_not_found"

        assert err .message =="Model not found: model-b"

    @pytest .mark .anyio 
    @respx .mock 
    async def test_500_yields_server_error_type (self ):
        respx .post (f"{BASE_URL }/chat/completions").mock (
        return_value =httpx .Response (500 ,text ="Internal Server Error")
        )
        provider =OpenAICompatibleProvider (_endpoint (),api_key =API_KEY )
        events =[]
        async for event in provider .stream_chat (_request ()):
            events .append (event )

        assert len (events )==1 
        assert isinstance (events [0 ],ErrorEvent )
        assert events [0 ].error_type =="server_error"







class TestAppConnectionError :
    @pytest .mark .anyio 
    async def test_connection_exception_yields_connection_error_type (self ):
        from typing import AsyncIterator 

        from llmsh .app import AppCore 
        from llmsh .config import AppConfig 
        from llmsh .models import ModelCapabilities ,ProfileConfig 
        from llmsh .providers .base import BaseProvider 

        class RaisingProvider (BaseProvider ):
            async def stream_chat (self ,request )->AsyncIterator :
                raise ConnectionRefusedError ("refused")
                yield 

        endpoint =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        provider_type ="openai_compatible",
        )
        profile =ProfileConfig (
        name ="default",
        endpoint ="local",
        model ="test-model",
        capabilities =ModelCapabilities (),
        )
        config =AppConfig (
        current_profile ="default",
        endpoints ={"local":endpoint },
        profiles ={"default":profile },
        )
        core =AppCore (config =config ,provider =RaisingProvider ())

        events =[]
        async for event in core .send_message ("hi"):
            events .append (event )

        assert len (events )==1 
        assert isinstance (events [0 ],ErrorEvent )
        assert events [0 ].error_type =="connection"







class TestAddErrorMessageHints :
    """Test that _add_error_message builds the right hint text for each error type."""

    def _make_app (self ):
        from llmsh .app import AppCore 
        from llmsh .config import AppConfig 
        from llmsh .models import ModelCapabilities ,ProfileConfig 
        from llmsh .ui .main import LlmshApp 
        from tests .conftest import StubProvider 

        endpoint =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        provider_type ="openai_compatible",
        )
        profile =ProfileConfig (
        name ="default",
        endpoint ="local",
        model ="test-model",
        capabilities =ModelCapabilities (),
        )
        config =AppConfig (
        current_profile ="default",
        endpoints ={"local":endpoint },
        profiles ={"default":profile },
        )
        core =AppCore (config =config ,provider =StubProvider ())
        return LlmshApp (core )

    def _get_message_widgets (self ,app ):
        from llmsh .ui .widgets import MessageWidget 

        return list (app .screen .query (MessageWidget ))

    @pytest .mark .anyio 
    async def test_model_not_found_shows_profile_hint (self ):
        app =self ._make_app ()
        async with app .run_test ()as pilot :
            app .screen ._add_error_message ("Model not found: x","model_not_found")
            await pilot .pause (delay =0.3 )
            widgets =self ._get_message_widgets (app )
            combined ="\n".join (w ._content for w in widgets )
            assert "/model"not in combined 
            assert "/profile"in combined 

    @pytest .mark .anyio 
    async def test_auth_shows_endpoint_hint (self ):
        app =self ._make_app ()
        async with app .run_test ()as pilot :
            app .screen ._add_error_message ("Authentication failed","auth")
            await pilot .pause (delay =0.3 )
            widgets =self ._get_message_widgets (app )
            combined ="\n".join (w ._content for w in widgets )
            assert "/endpoint"in combined 

    @pytest .mark .anyio 
    async def test_connection_shows_doctor_and_endpoint_hints (self ):
        app =self ._make_app ()
        async with app .run_test ()as pilot :
            app .screen ._add_error_message ("Connection refused","connection")
            await pilot .pause (delay =0.3 )
            widgets =self ._get_message_widgets (app )
            combined ="\n".join (w ._content for w in widgets )
            assert "/doctor"in combined 
            assert "/endpoint"in combined 

    @pytest .mark .anyio 
    async def test_unknown_shows_doctor_hint (self ):
        app =self ._make_app ()
        async with app .run_test ()as pilot :
            app .screen ._add_error_message ("Something broke","unknown")
            await pilot .pause (delay =0.3 )
            widgets =self ._get_message_widgets (app )
            combined ="\n".join (w ._content for w in widgets )
            assert "/doctor"in combined 

    @pytest .mark .anyio 
    async def test_error_message_is_system_role (self ):
        app =self ._make_app ()
        async with app .run_test ()as pilot :
            app .screen ._add_error_message ("Oops","unknown")
            await pilot .pause (delay =0.3 )
            widgets =self ._get_message_widgets (app )
            assert len (widgets )>0 
            last =widgets [-1 ]
            assert last ._role =="system"







class TestContextOverflowDetection :
    def _make_core (self ,provider ):
        from llmsh .app import AppCore 
        from llmsh .config import AppConfig 
        from llmsh .models import ModelCapabilities ,ProfileConfig 

        endpoint =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        provider_type ="openai_compatible",
        )
        profile =ProfileConfig (
        name ="default",
        endpoint ="local",
        model ="test-model",
        capabilities =ModelCapabilities (),
        )
        config =AppConfig (
        current_profile ="default",
        endpoints ={"local":endpoint },
        profiles ={"default":profile },
        )
        return AppCore (config =config ,provider =provider )

    @pytest .mark .anyio 
    async def test_long_conversation_with_connection_drop_is_context_overflow (self ):
        """Long conversation + connection drop => context_overflow."""
        from typing import AsyncIterator 

        from llmsh .models import ChatMessage 
        from llmsh .providers .base import BaseProvider 

        class FailingProvider (BaseProvider ):
            async def stream_chat (self ,request )->AsyncIterator :
                msg ="peer closed connection without sending complete message body"
                raise Exception (msg )
                yield 

        core =self ._make_core (FailingProvider ())

        long_content ="x"*2000 
        core ._messages .append (ChatMessage (role ="user",content =long_content ))
        core ._messages .append (ChatMessage (role ="assistant",content =long_content ))

        events =[]
        async for event in core .send_message ("next message"):
            events .append (event )

        assert len (events )==1 
        err =events [0 ]
        assert isinstance (err ,ErrorEvent )
        assert err .error_type =="context_overflow"

    @pytest .mark .anyio 
    async def test_short_conversation_with_connection_drop_is_connection_error (self ):
        """Short conversations with the same drop error are just connection errors."""
        from typing import AsyncIterator 

        from llmsh .providers .base import BaseProvider 

        class FailingProvider (BaseProvider ):
            async def stream_chat (self ,request )->AsyncIterator :
                msg ="peer closed connection without sending complete message body"
                raise Exception (msg )
                yield 

        core =self ._make_core (FailingProvider ())


        events =[]
        async for event in core .send_message ("hi"):
            events .append (event )

        assert len (events )==1 
        err =events [0 ]
        assert isinstance (err ,ErrorEvent )
        assert err .error_type =="connection"

    def test_http_400_with_context_message_is_context_overflow (self ):
        """HTTP 400 with 'context' or 'exceeds' in body is context_overflow."""
        body1 ='{"error":"request exceeds context size"}'
        assert _classify_http_error (400 ,body1 )=="context_overflow"
        body2 ='{"error":"context length exceeded"}'
        assert _classify_http_error (400 ,body2 )=="context_overflow"

    def test_http_400_with_token_limit_message_is_context_overflow (self ):
        """HTTP 400 with 'token' keyword is also context_overflow."""
        body ='{"error":"too many tokens in request"}'
        assert _classify_http_error (400 ,body )=="context_overflow"

    def test_http_400_with_too_long_message_is_context_overflow (self ):
        """HTTP 400 with 'too long' in body is context_overflow."""
        assert _classify_http_error (400 ,"prompt is too long")=="context_overflow"

    def test_http_400_without_context_keywords_is_unknown (self ):
        """HTTP 400 without context keywords is still unknown."""
        assert _classify_http_error (400 ,"bad request format")=="unknown"

    @pytest .mark .anyio 
    async def test_context_overflow_hint_mentions_clear_and_save (self ):
        """Context overflow error message contains /clear and /save hints."""
        from llmsh .app import AppCore 
        from llmsh .config import AppConfig 
        from llmsh .models import ModelCapabilities ,ProfileConfig 
        from llmsh .ui .main import LlmshApp 
        from tests .conftest import StubProvider 

        endpoint =EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        provider_type ="openai_compatible",
        )
        profile =ProfileConfig (
        name ="default",
        endpoint ="local",
        model ="test-model",
        capabilities =ModelCapabilities (),
        )
        config =AppConfig (
        current_profile ="default",
        endpoints ={"local":endpoint },
        profiles ={"default":profile },
        )
        core =AppCore (config =config ,provider =StubProvider ())
        app =LlmshApp (core )

        async with app .run_test ()as pilot :
            app .screen ._add_error_message (
            "The conversation may be too long for this model's context window.",
            "context_overflow",
            )
            await pilot .pause (delay =0.3 )
            from llmsh .ui .widgets import MessageWidget 
            widgets =list (app .screen .query (MessageWidget ))
            combined ="\n".join (w ._content for w in widgets )
            assert "/clear"in combined 
            assert "/save"in combined 
