"""Tests for config load/save and profile resolution in llmsh.config."""

import pytest 

from llmsh .config import (
AppConfig ,
get_active_profile ,
get_endpoint ,
load_config ,
resolve_api_key ,
save_config ,
)
from llmsh .errors import ConfigError ,ProfileNotFoundError 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 

SAMPLE_TOML ="""\
current_profile = "local"

[endpoints.localhost]
base_url = "http://localhost:1234/v1"
auth_mode = "none"
provider_type = "openai_compatible"
verify_ssl = true
timeout_seconds = 60

[endpoints.remote]
base_url = "https://example.com/v1"
auth_mode = "api_key"
api_key_env = "REMOTE_LLM_API_KEY"
provider_type = "openai_compatible"
verify_ssl = true
timeout_seconds = 60

[profiles.local]
endpoint = "localhost"
model = "my-local-model"

[profiles.local.capabilities]
streaming = true
token_usage = true
multimodal_input = false
tool_calling = false
structured_output = false
"""


@pytest .fixture 
def config_file (tmp_path ):
    path =tmp_path /"config.toml"
    path .write_text (SAMPLE_TOML )
    return path 


class TestLoadConfig :
    def test_load_valid_toml_parses_all_fields (self ,config_file ):
        config =load_config (config_file )

        assert config .current_profile =="local"
        assert "localhost"in config .endpoints 
        assert "remote"in config .endpoints 
        assert "local"in config .profiles 

    def test_endpoint_fields_parsed_correctly (self ,config_file ):
        config =load_config (config_file )

        ep =config .endpoints ["localhost"]
        assert ep .name =="localhost"
        assert ep .base_url =="http://localhost:1234/v1"
        assert ep .auth_mode =="none"
        assert ep .provider_type =="openai_compatible"
        assert ep .verify_ssl is True 
        assert ep .timeout_seconds ==60.0 

    def test_endpoint_with_api_key_env_parsed (self ,config_file ):
        config =load_config (config_file )

        ep =config .endpoints ["remote"]
        assert ep .name =="remote"
        assert ep .auth_mode =="api_key"
        assert ep .api_key_env =="REMOTE_LLM_API_KEY"

    def test_profile_fields_parsed_correctly (self ,config_file ):
        config =load_config (config_file )

        profile =config .profiles ["local"]
        assert profile .name =="local"
        assert profile .endpoint =="localhost"
        assert profile .model =="my-local-model"

    def test_profile_capabilities_parsed_correctly (self ,config_file ):
        config =load_config (config_file )

        caps =config .profiles ["local"].capabilities 
        assert caps .streaming is True 
        assert caps .token_usage is True 
        assert caps .multimodal_input is False 
        assert caps .tool_calling is False 
        assert caps .structured_output is False 

    def test_missing_config_file_raises_config_error (self ,tmp_path ):
        missing =tmp_path /"nonexistent.toml"
        with pytest .raises (ConfigError ):
            load_config (missing )

    def test_config_error_message_mentions_path (self ,tmp_path ):
        missing =tmp_path /"nonexistent.toml"
        with pytest .raises (ConfigError ,match =str (missing )):
            load_config (missing )

    def test_max_tokens_absent_loads_as_none (self ,config_file ):
        config =load_config (config_file )
        assert config .profiles ["local"].max_tokens is None 

    def test_max_tokens_present_loads_correctly (self ,tmp_path ):
        toml_content ="""\
current_profile = "dev"

[endpoints.dev-ep]
base_url = "http://localhost:1234/v1"
auth_mode = "none"
provider_type = "openai_compatible"

[profiles.dev]
endpoint = "dev-ep"
model = "some-model"
max_tokens = 2048
"""
        path =tmp_path /"config.toml"
        path .write_text (toml_content )
        config =load_config (path )
        assert config .profiles ["dev"].max_tokens ==2048 

    def test_capabilities_default_when_not_specified (self ,tmp_path ):
        toml_content ="""\
current_profile = "dev"

[endpoints.dev-ep]
base_url = "http://localhost:1234/v1"
auth_mode = "none"
provider_type = "openai_compatible"

[profiles.dev]
endpoint = "dev-ep"
model = "some-model"
"""
        path =tmp_path /"config.toml"
        path .write_text (toml_content )
        config =load_config (path )

        caps =config .profiles ["dev"].capabilities 
        assert isinstance (caps ,ModelCapabilities )
        assert caps .streaming is True 


class TestSaveConfig :
    def test_save_creates_file (self ,tmp_path ):
        path =tmp_path /"sub"/"config.toml"
        config =AppConfig (
        current_profile ="local",
        endpoints ={
        "localhost":EndpointConfig (
        name ="localhost",
        base_url ="http://localhost:1234/v1",
        auth_mode ="none",
        provider_type ="openai_compatible",
        )
        },
        profiles ={
        "local":ProfileConfig (
        name ="local",
        endpoint ="localhost",
        model ="my-model",
        capabilities =ModelCapabilities (),
        )
        },
        )
        save_config (config ,path )

        assert path .exists ()

    def test_roundtrip_produces_equivalent_config (self ,config_file ,tmp_path ):
        original =load_config (config_file )
        out_path =tmp_path /"output.toml"
        save_config (original ,out_path )
        restored =load_config (out_path )

        assert restored .current_profile ==original .current_profile 
        assert set (restored .endpoints .keys ())==set (original .endpoints .keys ())
        assert set (restored .profiles .keys ())==set (original .profiles .keys ())

        ep_orig =original .endpoints ["localhost"]
        ep_restored =restored .endpoints ["localhost"]
        assert ep_restored .base_url ==ep_orig .base_url 
        assert ep_restored .auth_mode ==ep_orig .auth_mode 
        assert ep_restored .name ==ep_orig .name 

        prof_orig =original .profiles ["local"]
        prof_restored =restored .profiles ["local"]
        assert prof_restored .endpoint ==prof_orig .endpoint 
        assert prof_restored .model ==prof_orig .model 
        assert prof_restored .capabilities .streaming ==prof_orig .capabilities .streaming 

    def test_roundtrip_preserves_max_tokens (self ,tmp_path ):
        config =AppConfig (
        current_profile ="p",
        endpoints ={
        "ep":EndpointConfig (
        name ="ep",
        base_url ="http://localhost/v1",
        auth_mode ="none",
        provider_type ="openai_compatible",
        )
        },
        profiles ={
        "p":ProfileConfig (
        name ="p",
        endpoint ="ep",
        model ="m",
        capabilities =ModelCapabilities (),
        max_tokens =2048 ,
        )
        },
        )
        path =tmp_path /"config.toml"
        save_config (config ,path )
        restored =load_config (path )
        assert restored .profiles ["p"].max_tokens ==2048 

    def test_save_omits_max_tokens_when_none (self ,tmp_path ):
        config =AppConfig (
        current_profile ="p",
        endpoints ={
        "ep":EndpointConfig (
        name ="ep",
        base_url ="http://localhost/v1",
        auth_mode ="none",
        provider_type ="openai_compatible",
        )
        },
        profiles ={
        "p":ProfileConfig (
        name ="p",
        endpoint ="ep",
        model ="m",
        capabilities =ModelCapabilities (),
        )
        },
        )
        path =tmp_path /"config.toml"
        save_config (config ,path )
        content =path .read_text ()
        assert "max_tokens"not in content 

    def test_save_creates_parent_directories (self ,tmp_path ):
        path =tmp_path /"a"/"b"/"config.toml"
        config =AppConfig (
        current_profile ="p",
        endpoints ={
        "ep":EndpointConfig (
        name ="ep",
        base_url ="http://localhost/v1",
        auth_mode ="none",
        provider_type ="openai_compatible",
        )
        },
        profiles ={
        "p":ProfileConfig (
        name ="p",
        endpoint ="ep",
        model ="m",
        capabilities =ModelCapabilities (),
        )
        },
        )
        save_config (config ,path )
        assert path .exists ()


class TestGetActiveProfile :
    def test_returns_correct_profile (self ,config_file ):
        config =load_config (config_file )
        profile =get_active_profile (config )

        assert isinstance (profile ,ProfileConfig )
        assert profile .name =="local"
        assert profile .model =="my-local-model"

    def test_missing_profile_raises_profile_not_found_error (self ,tmp_path ):
        config =AppConfig (
        current_profile ="nonexistent",
        endpoints ={},
        profiles ={},
        )
        with pytest .raises (ProfileNotFoundError ):
            get_active_profile (config )

    def test_profile_not_found_error_message_mentions_name (self ,tmp_path ):
        config =AppConfig (
        current_profile ="ghost",
        endpoints ={},
        profiles ={},
        )
        with pytest .raises (ProfileNotFoundError ,match ="ghost"):
            get_active_profile (config )


class TestGetEndpoint :
    def test_returns_correct_endpoint (self ,config_file ):
        config =load_config (config_file )
        profile =get_active_profile (config )
        endpoint =get_endpoint (config ,profile )

        assert isinstance (endpoint ,EndpointConfig )
        assert endpoint .name =="localhost"
        assert endpoint .base_url =="http://localhost:1234/v1"

    def test_missing_endpoint_raises_config_error (self ):
        config =AppConfig (
        current_profile ="p",
        endpoints ={},
        profiles ={
        "p":ProfileConfig (
        name ="p",
        endpoint ="missing-ep",
        model ="m",
        capabilities =ModelCapabilities (),
        )
        },
        )
        profile =config .profiles ["p"]
        with pytest .raises (ConfigError ):
            get_endpoint (config ,profile )


class TestResolveApiKey :
    def test_returns_key_when_env_var_set (self ,monkeypatch ):
        monkeypatch .setenv ("MY_API_KEY","secret-value")
        endpoint =EndpointConfig (
        name ="ep",
        base_url ="https://api.example.com/v1",
        auth_mode ="api_key",
        api_key_env ="MY_API_KEY",
        provider_type ="openai_compatible",
        )
        result =resolve_api_key (endpoint )
        assert result =="secret-value"

    def test_returns_none_when_env_var_not_set (self ,monkeypatch ):
        monkeypatch .delenv ("UNSET_API_KEY",raising =False )
        endpoint =EndpointConfig (
        name ="ep",
        base_url ="https://api.example.com/v1",
        auth_mode ="api_key",
        api_key_env ="UNSET_API_KEY",
        provider_type ="openai_compatible",
        )
        result =resolve_api_key (endpoint )
        assert result is None 

    def test_returns_none_when_no_api_key_env (self ):
        endpoint =EndpointConfig (
        name ="ep",
        base_url ="https://api.example.com/v1",
        auth_mode ="none",
        api_key_env =None ,
        provider_type ="openai_compatible",
        )
        result =resolve_api_key (endpoint )
        assert result is None 
