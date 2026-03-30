"""Tests for the endpoint CLI command group."""

from __future__ import annotations 

import tomli_w 
from typer .testing import CliRunner 

from llmsh .cli import app 

runner =CliRunner ()


def _write_config (path ,profiles =None ,endpoints =None ,current_profile ="default"):
    """Write a minimal TOML config file for testing."""
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


class TestEndpointList :
    def test_lists_endpoints (self ,tmp_path ,monkeypatch ):
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
        monkeypatch .setattr (
        "llmsh.commands.endpoint.config_path",lambda :config_file ,
        )

        result =runner .invoke (app ,["endpoint","list"])

        assert result .exit_code ==0 
        assert "local"in result .stdout 
        assert "remote"in result .stdout 

    def test_shows_url_and_auth (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr (
        "llmsh.commands.endpoint.config_path",lambda :config_file ,
        )

        result =runner .invoke (app ,["endpoint","list"])

        assert result .exit_code ==0 
        assert "http://localhost:8006/v1"in result .stdout 
        assert "none"in result .stdout 


class TestEndpointAdd :
    def test_adds_endpoint (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr (
        "llmsh.commands.endpoint.config_path",lambda :config_file ,
        )

        result =runner .invoke (
        app ,
        ["endpoint","add","remote","--url","https://api.example.com/v1"],
        )

        assert result .exit_code ==0 
        assert "remote"in result .stdout .lower ()

        from llmsh .config import load_config 

        config =load_config (config_file )
        assert "remote"in config .endpoints 

    def test_adds_endpoint_with_auth (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr (
        "llmsh.commands.endpoint.config_path",lambda :config_file ,
        )

        result =runner .invoke (
        app ,
        [
        "endpoint","add","remote",
        "--url","https://api.example.com/v1",
        "--auth-mode","api_key",
        "--api-key-env","MY_KEY",
        ],
        )

        assert result .exit_code ==0 

        from llmsh .config import load_config 

        config =load_config (config_file )
        ep =config .endpoints ["remote"]
        assert ep .auth_mode =="api_key"
        assert ep .api_key_env =="MY_KEY"


class TestEndpointRemove :
    def test_removes_endpoint (self ,tmp_path ,monkeypatch ):
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
        monkeypatch .setattr (
        "llmsh.commands.endpoint.config_path",lambda :config_file ,
        )

        result =runner .invoke (app ,["endpoint","remove","remote"])

        assert result .exit_code ==0 

        from llmsh .config import load_config 

        config =load_config (config_file )
        assert "remote"not in config .endpoints 

    def test_nonexistent_endpoint_exits_nonzero (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr (
        "llmsh.commands.endpoint.config_path",lambda :config_file ,
        )

        result =runner .invoke (app ,["endpoint","remove","nonexistent"])

        assert result .exit_code !=0 

    def test_warns_about_referencing_profiles (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr (
        "llmsh.commands.endpoint.config_path",lambda :config_file ,
        )

        result =runner .invoke (app ,["endpoint","remove","local"])

        assert result .exit_code ==0 
        assert "warning"in result .stdout .lower ()or "default"in result .stdout .lower ()
