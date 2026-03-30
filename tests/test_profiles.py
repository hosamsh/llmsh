"""Tests for the shared profiles module."""

from __future__ import annotations 

import pytest 

from llmsh .errors import ConfigError ,ProfileNotFoundError 


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


class TestListProfiles :
    def test_returns_all_profiles (self ,tmp_path ):
        from llmsh .profiles import list_profiles 

        config_file =tmp_path /"config.toml"
        _write_config (
        config_file ,
        profiles ={
        "default":{"endpoint":"local","model":"m1","capabilities":{}},
        "work":{"endpoint":"local","model":"m2","capabilities":{}},
        },
        )

        result =list_profiles (config_file )

        assert len (result )==2 
        names ={p ["name"]for p in result }
        assert names =={"default","work"}

    def test_marks_active_profile (self ,tmp_path ):
        from llmsh .profiles import list_profiles 

        config_file =tmp_path /"config.toml"
        _write_config (config_file ,current_profile ="default")

        result =list_profiles (config_file )

        active =[p for p in result if p ["active"]]
        assert len (active )==1 
        assert active [0 ]["name"]=="default"

    def test_includes_endpoint_and_model (self ,tmp_path ):
        from llmsh .profiles import list_profiles 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        result =list_profiles (config_file )

        entry =result [0 ]
        assert entry ["endpoint"]=="local"
        assert entry ["model"]=="test-model"

    def test_raises_config_error_when_file_missing (self ,tmp_path ):
        from llmsh .profiles import list_profiles 

        missing =tmp_path /"nonexistent.toml"

        with pytest .raises (ConfigError ):
            list_profiles (missing )


class TestAddProfile :
    def test_adds_new_profile_to_config (self ,tmp_path ):
        from llmsh .profiles import add_profile ,list_profiles 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        add_profile (config_file ,"work","local","gpt-4")

        result =list_profiles (config_file )
        names ={p ["name"]for p in result }
        assert "work"in names 

    def test_persists_to_disk (self ,tmp_path ):
        from llmsh .config import load_config 
        from llmsh .profiles import add_profile 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        add_profile (config_file ,"work","local","gpt-4")

        config =load_config (config_file )
        assert "work"in config .profiles 
        assert config .profiles ["work"].model =="gpt-4"
        assert config .profiles ["work"].endpoint =="local"


class TestUseProfile :
    def test_switches_current_profile (self ,tmp_path ):
        from llmsh .config import load_config 
        from llmsh .profiles import use_profile 

        config_file =tmp_path /"config.toml"
        _write_config (
        config_file ,
        profiles ={
        "default":{"endpoint":"local","model":"m1","capabilities":{}},
        "work":{"endpoint":"local","model":"m2","capabilities":{}},
        },
        )

        use_profile (config_file ,"work")

        config =load_config (config_file )
        assert config .current_profile =="work"

    def test_raises_when_profile_not_found (self ,tmp_path ):
        from llmsh .profiles import use_profile 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        with pytest .raises (ProfileNotFoundError ):
            use_profile (config_file ,"nonexistent")


class TestSetModel :
    def test_changes_model_for_named_profile (self ,tmp_path ):
        from llmsh .config import load_config 
        from llmsh .profiles import set_model 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        set_model (config_file ,"default","new-model")

        config =load_config (config_file )
        assert config .profiles ["default"].model =="new-model"

    def test_raises_when_profile_not_found (self ,tmp_path ):
        from llmsh .profiles import set_model 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        with pytest .raises (ProfileNotFoundError ):
            set_model (config_file ,"nonexistent","new-model")


class TestRemoveProfile :
    def test_removes_profile_from_config (self ,tmp_path ):
        from llmsh .config import load_config 
        from llmsh .profiles import remove_profile 

        config_file =tmp_path /"config.toml"
        _write_config (
        config_file ,
        profiles ={
        "default":{"endpoint":"local","model":"m1","capabilities":{}},
        "work":{"endpoint":"local","model":"m2","capabilities":{}},
        },
        )

        remove_profile (config_file ,"work")

        config =load_config (config_file )
        assert "work"not in config .profiles 

    def test_raises_when_profile_not_found (self ,tmp_path ):
        from llmsh .profiles import remove_profile 

        config_file =tmp_path /"config.toml"
        _write_config (config_file )

        with pytest .raises (ProfileNotFoundError ):
            remove_profile (config_file ,"nonexistent")

    def test_raises_when_removing_active_profile (self ,tmp_path ):
        from llmsh .profiles import remove_profile 

        config_file =tmp_path /"config.toml"
        _write_config (config_file ,current_profile ="default")

        with pytest .raises (ConfigError ):
            remove_profile (config_file ,"default")
