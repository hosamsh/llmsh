"""Tests for data models in llmsh.models."""

import json 
from datetime import UTC ,datetime 

import pytest 
from pydantic import ValidationError 

from llmsh .models import (
ChatMessage ,
EndpointConfig ,
ModelCapabilities ,
ProfileConfig ,
SessionRecord ,
UsageInfo ,
)


class TestEndpointConfig :
    def test_required_fields (self ):
        config =EndpointConfig (
        name ="my-endpoint",
        base_url ="https://api.example.com",
        auth_mode ="api_key",
        provider_type ="openai_compatible",
        )
        assert config .name =="my-endpoint"
        assert config .base_url =="https://api.example.com"
        assert config .auth_mode =="api_key"
        assert config .provider_type =="openai_compatible"

    def test_defaults (self ):
        config =EndpointConfig (
        name ="x",
        base_url ="https://api.example.com",
        auth_mode ="none",
        provider_type ="openai_compatible",
        )
        assert config .verify_ssl is True 
        assert config .timeout_seconds ==60.0 
        assert config .api_key_env is None 
        assert config .api_key_value is None 

    def test_invalid_auth_mode_raises_validation_error (self ):
        with pytest .raises (ValidationError ):
            EndpointConfig (
            name ="x",
            base_url ="https://api.example.com",
            auth_mode ="invalid_mode",
            provider_type ="openai_compatible",
            )

    def test_invalid_provider_type_raises_validation_error (self ):
        with pytest .raises (ValidationError ):
            EndpointConfig (
            name ="x",
            base_url ="https://api.example.com",
            auth_mode ="none",
            provider_type ="unsupported",
            )


class TestModelCapabilities :
    def test_defaults (self ):
        caps =ModelCapabilities ()
        assert caps .streaming is True 
        assert caps .token_usage is True 
        assert caps .multimodal_input is False 
        assert caps .tool_calling is True 
        assert caps .structured_output is False 

    def test_override_defaults (self ):
        caps =ModelCapabilities (streaming =False ,tool_calling =True )
        assert caps .streaming is False 
        assert caps .tool_calling is True 


class TestProfileConfig :
    def test_required_fields_with_default_capabilities (self ):
        profile =ProfileConfig (
        name ="my-profile",
        endpoint ="my-endpoint",
        model ="gpt-4",
        capabilities =ModelCapabilities (),
        )
        assert profile .name =="my-profile"
        assert profile .endpoint =="my-endpoint"
        assert profile .model =="gpt-4"
        assert isinstance (profile .capabilities ,ModelCapabilities )

    def test_references_model_capabilities (self ):
        caps =ModelCapabilities (tool_calling =True )
        profile =ProfileConfig (
        name ="p",
        endpoint ="e",
        model ="m",
        capabilities =caps ,
        )
        assert profile .capabilities .tool_calling is True 

    def test_max_tokens_defaults_to_none (self ):
        profile =ProfileConfig (
        name ="p",
        endpoint ="e",
        model ="m",
        capabilities =ModelCapabilities (),
        )
        assert profile .max_tokens is None 

    def test_max_tokens_accepts_integer (self ):
        profile =ProfileConfig (
        name ="p",
        endpoint ="e",
        model ="m",
        capabilities =ModelCapabilities (),
        max_tokens =2048 ,
        )
        assert profile .max_tokens ==2048 


class TestChatMessage :
    def test_required_fields (self ):
        msg =ChatMessage (role ="user",content ="Hello")
        assert msg .role =="user"
        assert msg .content =="Hello"

    def test_created_at_defaults_to_utc_now (self ):
        before =datetime .now (UTC )
        msg =ChatMessage (role ="assistant",content ="Hi")
        after =datetime .now (UTC )
        assert before <=msg .created_at <=after 

    def test_all_valid_roles (self ):
        for role in ("system","user","assistant"):
            msg =ChatMessage (role =role ,content ="text")
            assert msg .role ==role 

    def test_invalid_role_raises_validation_error (self ):
        with pytest .raises (ValidationError ):
            ChatMessage (role ="admin",content ="text")

    def test_roundtrip_json_preserves_datetime (self ):
        original =ChatMessage (
        role ="user",
        content ="Hello",
        created_at =datetime (2024 ,6 ,15 ,12 ,0 ,0 ,tzinfo =UTC ),
        )
        json_str =original .model_dump_json ()
        restored =ChatMessage .model_validate_json (json_str )
        assert restored ==original 
        assert restored .created_at ==original .created_at 


class TestUsageInfo :
    def test_all_fields_optional (self ):
        usage =UsageInfo ()
        assert usage .input_tokens is None 
        assert usage .output_tokens is None 
        assert usage .total_tokens is None 

    def test_with_values (self ):
        usage =UsageInfo (input_tokens =10 ,output_tokens =20 ,total_tokens =30 )
        assert usage .input_tokens ==10 
        assert usage .output_tokens ==20 
        assert usage .total_tokens ==30 


class TestSessionRecord :
    def _make_session (self )->SessionRecord :
        now =datetime (2024 ,1 ,1 ,tzinfo =UTC )
        return SessionRecord (
        id ="sess-001",
        title ="Test session",
        profile ="default",
        model ="gpt-4",
        created_at =now ,
        updated_at =now ,
        messages =[
        ChatMessage (
        role ="user",
        content ="Hello",
        created_at =now ,
        ),
        ChatMessage (
        role ="assistant",
        content ="Hi there",
        created_at =now ,
        ),
        ],
        usage =[
        UsageInfo (input_tokens =5 ,output_tokens =10 ,total_tokens =15 ),
        ],
        )

    def test_required_fields (self ):
        session =self ._make_session ()
        assert session .id =="sess-001"
        assert session .title =="Test session"
        assert session .profile =="default"
        assert session .model =="gpt-4"
        assert len (session .messages )==2 
        assert len (session .usage )==1 

    def test_messages_are_chat_message_instances (self ):
        session =self ._make_session ()
        for msg in session .messages :
            assert isinstance (msg ,ChatMessage )

    def test_usage_are_usage_info_instances (self ):
        session =self ._make_session ()
        for u in session .usage :
            assert isinstance (u ,UsageInfo )

    def test_roundtrip_json (self ):
        original =self ._make_session ()
        json_str =original .model_dump_json ()
        restored =SessionRecord .model_validate_json (json_str )
        assert restored ==original 

    def test_roundtrip_json_via_dict (self ):
        original =self ._make_session ()
        data =json .loads (original .model_dump_json ())
        restored =SessionRecord .model_validate (data )
        assert restored .id ==original .id 
        assert restored .messages [0 ].content ==original .messages [0 ].content 
        assert restored .messages [0 ].created_at ==original .messages [0 ].created_at 
