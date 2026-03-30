"""Tests for the shared endpoints module."""

from __future__ import annotations 

import pytest 

from llmsh .errors import ConfigError ,EndpointNotFoundError 


def _write_config (path ,profiles =None ,endpoints =None ,current_profile ="default"):
    """Write a minimal TOML config file for testing."""
    import tomli_w 

    data ={"current_profile":current_profile }
    if endpoints is None :
        endpoints ={
        "local":{
        "base_url":"http://localhost:8006/v1",
        "auth_mode":"none",
        },
        }
    data ["endpoints"]=endpoints 
    if profiles is None :
        profiles ={
        "default":{
        "endpoint":"local",
        "model":"test-model",
        "capabilities":{},
        },
        }
    data ["profiles"]=profiles 
    path .write_bytes (tomli_w .dumps (data ).encode ())


class TestListEndpoints :
    def test_returns_all_endpoints (self ,tmp_path ):
        from llmsh .endpoints import list_endpoints 

        config_file =tmp_path /"config.toml"
        _write_config (
        config_file ,
        endpoints ={
        "local":{
        "base_url":"http://localhost:8006/v1",
        "auth_mode":"none",
        },
        "remote":{
        "base_url":"https://api.example.com/v1",
        "auth_mode":"api_key",
        "api_key_env":"EXAMPLE_KEY",
        },
        },
        )

        result =list_endpoints (config_file )

        assert len (result )==2 
        names ={e ["name"]for e in result }
        assert names =={"local","remote"}

    def test_includes_url_and_auth_mode (self ,tmp_path ):
        from llmsh .endpoints import list_endpoints 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        result =list_endpoints (config_file )

        entry =result [0 ]
        assert entry ["name"]=="local"
        assert entry ["base_url"]=="http://localhost:8006/v1"
        assert entry ["auth_mode"]=="none"

    def test_raises_config_error_when_file_missing (self ,tmp_path ):
        from llmsh .endpoints import list_endpoints 

        missing =tmp_path /"nonexistent.toml"

        with pytest .raises (ConfigError ):
            list_endpoints (missing )


class TestAddEndpoint :
    def test_adds_new_endpoint_to_config (self ,tmp_path ):
        from llmsh .endpoints import add_endpoint ,list_endpoints 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        add_endpoint (config_file ,"remote","https://api.example.com/v1")

        result =list_endpoints (config_file )
        names ={e ["name"]for e in result }
        assert "remote"in names 

    def test_persists_to_disk (self ,tmp_path ):
        from llmsh .config import load_config 
        from llmsh .endpoints import add_endpoint 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        add_endpoint (config_file ,"remote","https://api.example.com/v1")

        config =load_config (config_file )
        assert "remote"in config .endpoints 
        assert config .endpoints ["remote"].base_url =="https://api.example.com/v1"
        assert config .endpoints ["remote"].auth_mode =="none"

    def test_add_with_auth_mode_and_api_key_env (self ,tmp_path ):
        from llmsh .config import load_config 
        from llmsh .endpoints import add_endpoint 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        add_endpoint (
        config_file ,"remote","https://api.example.com/v1",
        auth_mode ="api_key",api_key_env ="MY_KEY",
        )

        config =load_config (config_file )
        ep =config .endpoints ["remote"]
        assert ep .auth_mode =="api_key"
        assert ep .api_key_env =="MY_KEY"


class TestRemoveEndpoint :
    def test_removes_endpoint_from_config (self ,tmp_path ):
        from llmsh .config import load_config 
        from llmsh .endpoints import remove_endpoint 

        config_file =tmp_path /"config.toml"
        _write_config (
        config_file ,
        endpoints ={
        "local":{
        "base_url":"http://localhost:8006/v1",
        "auth_mode":"none",
        },
        "remote":{
        "base_url":"https://api.example.com/v1",
        "auth_mode":"none",
        },
        },
        )

        remove_endpoint (config_file ,"remote")

        config =load_config (config_file )
        assert "remote"not in config .endpoints 

    def test_raises_when_endpoint_not_found (self ,tmp_path ):
        from llmsh .endpoints import remove_endpoint 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        with pytest .raises (EndpointNotFoundError ):
            remove_endpoint (config_file ,"nonexistent")

    def test_returns_referencing_profiles (self ,tmp_path ):
        from llmsh .endpoints import remove_endpoint 

        config_file =tmp_path /"config.toml"
        _write_config (
        config_file ,
        endpoints ={
        "local":{
        "base_url":"http://localhost:8006/v1",
        "auth_mode":"none",
        },
        },
        profiles ={
        "default":{
        "endpoint":"local",
        "model":"test-model",
        "capabilities":{},
        },
        },
        )

        referencing =remove_endpoint (config_file ,"local")

        assert referencing ==["default"]
