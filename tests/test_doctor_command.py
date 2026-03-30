"""Tests for the doctor CLI command."""

from __future__ import annotations 

from typer .testing import CliRunner 

from llmsh .cli import app 
from llmsh .providers .base import DoctorCheck ,DoctorReport 

runner =CliRunner ()


class TestDoctorAllPass :
    def test_prints_results_and_exits_0 (self ,monkeypatch ):
        report =DoctorReport (checks =[
        DoctorCheck (
        name ="config_valid",passed =True ,
        message ="Configuration is valid",
        ),
        DoctorCheck (
        name ="endpoint_reachable",passed =True ,
        message ="Endpoint OK",
        ),
        ])

        async def fake_run_doctor (profile ,endpoint ,provider ):
            return report 

        monkeypatch .setattr ("llmsh.commands.doctor.run_doctor",fake_run_doctor )
        monkeypatch .setattr (
        "llmsh.commands.doctor._make_core",
        lambda profile :type ("FakeCore",(),{
        "profile":type ("P",(),{"name":"default"})(),
        "_config":type ("C",(),{"endpoints":{}})(),
        "_provider":None ,
        })(),
        )
        monkeypatch .setattr (
        "llmsh.commands.doctor.get_endpoint",
        lambda config ,profile :None ,
        )

        result =runner .invoke (app ,["doctor"])

        assert result .exit_code ==0 
        assert "[PASS]"in result .stdout 
        assert "config_valid"in result .stdout 
        assert "endpoint_reachable"in result .stdout 


class TestDoctorWithFailure :
    def test_prints_fail_and_exits_nonzero (self ,monkeypatch ):
        report =DoctorReport (checks =[
        DoctorCheck (
        name ="config_valid",passed =True ,message ="OK",
        ),
        DoctorCheck (
        name ="endpoint_reachable",passed =False ,
        message ="Connection refused",
        ),
        ])

        async def fake_run_doctor (profile ,endpoint ,provider ):
            return report 

        monkeypatch .setattr ("llmsh.commands.doctor.run_doctor",fake_run_doctor )
        monkeypatch .setattr (
        "llmsh.commands.doctor._make_core",
        lambda profile :type ("FakeCore",(),{
        "profile":type ("P",(),{"name":"default"})(),
        "_config":type ("C",(),{"endpoints":{}})(),
        "_provider":None ,
        })(),
        )
        monkeypatch .setattr (
        "llmsh.commands.doctor.get_endpoint",
        lambda config ,profile :None ,
        )

        result =runner .invoke (app ,["doctor"])

        assert result .exit_code ==1 
        assert "[FAIL]"in result .stdout 
        assert "Connection refused"in result .stdout 


class TestDoctorProfileFlag :
    def test_profile_passed_to_core (self ,monkeypatch ):
        received ={}

        def fake_make_core (profile ):
            received ["profile"]=profile 
            return type ("FakeCore",(),{
            "profile":type ("P",(),{"name":profile or "default"})(),
            "_config":type ("C",(),{"endpoints":{}})(),
            "_provider":None ,
            })()

        report =DoctorReport (checks =[
        DoctorCheck (name ="config_valid",passed =True ,message ="OK"),
        ])

        async def fake_run_doctor (profile ,endpoint ,provider ):
            return report 

        monkeypatch .setattr ("llmsh.commands.doctor.run_doctor",fake_run_doctor )
        monkeypatch .setattr ("llmsh.commands.doctor._make_core",fake_make_core )
        monkeypatch .setattr (
        "llmsh.commands.doctor.get_endpoint",
        lambda config ,profile :None ,
        )

        runner .invoke (app ,["doctor","--profile","custom"])

        assert received ["profile"]=="custom"
