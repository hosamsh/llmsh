"""Tests for the doctor diagnostic workflow."""

from __future__ import annotations 

from typing import AsyncIterator 

import pytest 

from llmsh .app import AppCore 
from llmsh .config import AppConfig 
from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 
from llmsh .providers .base import (
BaseProvider ,
DoctorCheck ,
DoctorReport ,
ProviderEvent ,
ProviderRequest ,
ProviderResult ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
)






class CooperativeProvider (BaseProvider ):
    """A provider that passes all checks."""

    def __init__ (self ,models :list [str ]|None =None )->None :
        self ._models =models or ["test-model"]

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        yield ResponseStarted ()
        yield TextDelta (text ="ok")
        yield ResponseCompleted (content ="ok")

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        return ProviderResult (content ="ok")

    async def list_models (self )->list [str ]:
        return self ._models 

    async def doctor (self )->DoctorReport :
        return DoctorReport (
        checks =[
        DoctorCheck (
        name ="reachability",
        passed =True ,
        message ="Endpoint reachable",
        )
        ]
        )


class UnreachableProvider (BaseProvider ):
    """A provider that raises on doctor() to simulate unreachable endpoint."""

    async def stream_chat (
    self ,request :ProviderRequest 
    )->AsyncIterator [ProviderEvent ]:
        raise ConnectionError ("unreachable")
        yield 

    async def chat (self ,request :ProviderRequest )->ProviderResult :
        raise ConnectionError ("unreachable")

    async def list_models (self )->list [str ]:
        raise ConnectionError ("unreachable")

    async def doctor (self )->DoctorReport :
        raise ConnectionError ("Cannot connect to endpoint")


class WrongModelsProvider (CooperativeProvider ):
    """A cooperative provider but configured model is missing from list_models."""

    async def list_models (self )->list [str ]:
        return ["other-model","another-model"]







def _make_config (model :str ="test-model")->AppConfig :
    endpoint =EndpointConfig (
    name ="local",
    base_url ="http://localhost:8006/v1",
    auth_mode ="none",
    provider_type ="openai_compatible",
    )
    profile =ProfileConfig (
    name ="default",
    endpoint ="local",
    model =model ,
    capabilities =ModelCapabilities (),
    )
    return AppConfig (
    current_profile ="default",
    endpoints ={"local":endpoint },
    profiles ={"default":profile },
    )


def _make_core (provider :BaseProvider ,model :str ="test-model")->AppCore :
    return AppCore (config =_make_config (model =model ),provider =provider )







class TestRunDoctorAllPass :
    @pytest .mark .anyio 
    async def test_all_checks_pass_with_cooperative_provider (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config ()
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =CooperativeProvider (models =["test-model"])

        report =await run_doctor (profile ,endpoint ,provider )

        assert isinstance (report ,DoctorReport )
        assert len (report .checks )>0 
        assert all (c .passed for c in report .checks )

    @pytest .mark .anyio 
    async def test_report_contains_config_valid_check (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config ()
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =CooperativeProvider ()

        report =await run_doctor (profile ,endpoint ,provider )

        names =[c .name for c in report .checks ]
        assert "config_valid"in names 

    @pytest .mark .anyio 
    async def test_report_contains_endpoint_reachable_check (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config ()
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =CooperativeProvider ()

        report =await run_doctor (profile ,endpoint ,provider )

        names =[c .name for c in report .checks ]
        assert "endpoint_reachable"in names 

    @pytest .mark .anyio 
    async def test_report_contains_models_listable_check (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config ()
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =CooperativeProvider ()

        report =await run_doctor (profile ,endpoint ,provider )

        names =[c .name for c in report .checks ]
        assert "models_listable"in names 

    @pytest .mark .anyio 
    async def test_report_contains_model_present_check (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config ()
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =CooperativeProvider ()

        report =await run_doctor (profile ,endpoint ,provider )

        names =[c .name for c in report .checks ]
        assert "model_present"in names 

    @pytest .mark .anyio 
    async def test_each_check_has_name_passed_message (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config ()
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =CooperativeProvider ()

        report =await run_doctor (profile ,endpoint ,provider )

        for check in report .checks :
            assert isinstance (check .name ,str )and check .name 
            assert isinstance (check .passed ,bool )
            assert isinstance (check .message ,str )and check .message 


class TestRunDoctorEndpointUnreachable :
    @pytest .mark .anyio 
    async def test_endpoint_reachable_fails_when_provider_raises (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config ()
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =UnreachableProvider ()

        report =await run_doctor (profile ,endpoint ,provider )

        reachable =next (
        (c for c in report .checks if c .name =="endpoint_reachable"),None 
        )
        assert reachable is not None 
        assert reachable .passed is False 

    @pytest .mark .anyio 
    async def test_subsequent_checks_still_run_after_reachability_fails (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config ()
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =UnreachableProvider ()

        report =await run_doctor (profile ,endpoint ,provider )

        names =[c .name for c in report .checks ]

        assert "models_listable"in names 
        assert "model_present"in names 

    @pytest .mark .anyio 
    async def test_config_valid_still_passes_when_endpoint_unreachable (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config ()
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =UnreachableProvider ()

        report =await run_doctor (profile ,endpoint ,provider )

        config_check =next (
        (c for c in report .checks if c .name =="config_valid"),None 
        )
        assert config_check is not None 
        assert config_check .passed is True 


class TestRunDoctorModelNotPresent :
    @pytest .mark .anyio 
    async def test_model_present_fails_when_model_not_in_list (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config (model ="missing-model")
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =WrongModelsProvider ()

        report =await run_doctor (profile ,endpoint ,provider )

        model_check =next (
        (c for c in report .checks if c .name =="model_present"),None 
        )
        assert model_check is not None 
        assert model_check .passed is False 

    @pytest .mark .anyio 
    async def test_model_present_pass_message_contains_model_name (self ):
        from llmsh .doctor import run_doctor 

        config =_make_config (model ="test-model")
        profile =config .profiles ["default"]
        endpoint =config .endpoints ["local"]
        provider =CooperativeProvider (models =["test-model"])

        report =await run_doctor (profile ,endpoint ,provider )

        model_check =next (
        (c for c in report .checks if c .name =="model_present"),None 
        )
        assert model_check is not None 
        assert "test-model"in model_check .message 


