"""Tests for the profile CLI command group."""

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


class TestProfileList :
    def test_lists_profiles (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (
        config_file ,
        profiles ={
        "default":{"endpoint":"local","model":"m1","capabilities":{}},
        "work":{"endpoint":"local","model":"m2","capabilities":{}},
        },
        )
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (app ,["profile","list"])

        assert result .exit_code ==0 
        assert "default"in result .stdout 
        assert "work"in result .stdout 

    def test_marks_active_profile (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file ,current_profile ="default")
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (app ,["profile","list"])

        assert result .exit_code ==0 
        assert "active"in result .stdout .lower ()


class TestProfileAdd :
    def test_adds_profile (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (
        app ,
        ["profile","add","work","--endpoint","local","--model","gpt-4"],
        )

        assert result .exit_code ==0 
        assert "work"in result .stdout .lower ()

        from llmsh .config import load_config 

        config =load_config (config_file )
        assert "work"in config .profiles 


class TestProfileUse :
    def test_switches_profile (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (
        config_file ,
        profiles ={
        "default":{"endpoint":"local","model":"m1","capabilities":{}},
        "work":{"endpoint":"local","model":"m2","capabilities":{}},
        },
        )
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (app ,["profile","use","work"])

        assert result .exit_code ==0 

        from llmsh .config import load_config 

        config =load_config (config_file )
        assert config .current_profile =="work"

    def test_nonexistent_profile_exits_nonzero (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (app ,["profile","use","nonexistent"])

        assert result .exit_code !=0 


class TestProfileSetModel :
    def test_changes_model (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (
        app ,["profile","set-model","default","new-model"],
        )

        assert result .exit_code ==0 

        from llmsh .config import load_config 

        config =load_config (config_file )
        assert config .profiles ["default"].model =="new-model"

    def test_nonexistent_profile_exits_nonzero (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (
        app ,["profile","set-model","nonexistent","new-model"],
        )

        assert result .exit_code !=0 


class TestProfileRemove :
    def test_removes_profile (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (
        config_file ,
        profiles ={
        "default":{"endpoint":"local","model":"m1","capabilities":{}},
        "work":{"endpoint":"local","model":"m2","capabilities":{}},
        },
        )
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (app ,["profile","remove","work"])

        assert result .exit_code ==0 

        from llmsh .config import load_config 

        config =load_config (config_file )
        assert "work"not in config .profiles 

    def test_nonexistent_profile_exits_nonzero (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file )
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (app ,["profile","remove","nonexistent"])

        assert result .exit_code !=0 

    def test_active_profile_exits_nonzero (self ,tmp_path ,monkeypatch ):
        config_file =tmp_path /"config.toml"
        _write_config (config_file ,current_profile ="default")
        monkeypatch .setattr ("llmsh.commands.profile.config_path",lambda :config_file )

        result =runner .invoke (app ,["profile","remove","default"])

        assert result .exit_code !=0 
