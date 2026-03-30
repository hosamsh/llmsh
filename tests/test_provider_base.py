"""Tests for provider base contract and event model."""

import pytest 

from llmsh .models import ChatMessage ,UsageInfo 
from llmsh .providers .base import (
BaseProvider ,
CancelledEvent ,
DoctorCheck ,
DoctorReport ,
ErrorEvent ,
ProviderEvent ,
ProviderRequest ,
ProviderResult ,
ResponseCompleted ,
ResponseStarted ,
TextDelta ,
TokenUsageEvent ,
)






class TestResponseStarted :
    def test_can_be_instantiated (self ):
        event =ResponseStarted ()
        assert isinstance (event ,ResponseStarted )

    def test_is_provider_event (self ):


        import typing 
        assert ResponseStarted in typing .get_args (ProviderEvent )


class TestTextDelta :
    def test_has_text_field (self ):
        event =TextDelta (text ="hello")
        assert event .text =="hello"

    def test_is_provider_event (self ):
        import typing 
        assert TextDelta in typing .get_args (ProviderEvent )


class TestResponseCompleted :
    def test_has_content_field (self ):
        event =ResponseCompleted (content ="full response")
        assert event .content =="full response"

    def test_is_provider_event (self ):
        import typing 
        assert ResponseCompleted in typing .get_args (ProviderEvent )


class TestTokenUsageEvent :
    def test_has_usage_field (self ):
        usage =UsageInfo (input_tokens =10 ,output_tokens =20 ,total_tokens =30 )
        event =TokenUsageEvent (usage =usage )
        assert event .usage .input_tokens ==10 
        assert event .usage .output_tokens ==20 
        assert event .usage .total_tokens ==30 

    def test_is_provider_event (self ):
        import typing 
        assert TokenUsageEvent in typing .get_args (ProviderEvent )


class TestErrorEvent :
    def test_has_message_field (self ):
        event =ErrorEvent (message ="something went wrong")
        assert event .message =="something went wrong"

    def test_is_provider_event (self ):
        import typing 
        assert ErrorEvent in typing .get_args (ProviderEvent )


class TestCancelledEvent :
    def test_can_be_instantiated (self ):
        event =CancelledEvent ()
        assert isinstance (event ,CancelledEvent )

    def test_is_provider_event (self ):
        import typing 
        assert CancelledEvent in typing .get_args (ProviderEvent )







class TestProviderRequest :
    def test_required_fields (self ):
        messages =[ChatMessage (role ="user",content ="hello")]
        request =ProviderRequest (messages =messages ,model ="gpt-4")
        assert request .messages ==messages 
        assert request .model =="gpt-4"

    def test_optional_params_default_none (self ):
        request =ProviderRequest (
        messages =[ChatMessage (role ="user",content ="hi")],
        model ="gpt-4",
        )
        assert request .temperature is None 
        assert request .max_tokens is None 







class TestProviderResult :
    def test_required_fields (self ):
        result =ProviderResult (content ="response text")
        assert result .content =="response text"

    def test_usage_optional (self ):
        result =ProviderResult (content ="response text")
        assert result .usage is None 

    def test_with_usage (self ):
        usage =UsageInfo (input_tokens =5 ,output_tokens =10 ,total_tokens =15 )
        result =ProviderResult (content ="text",usage =usage )
        assert result .usage is not None 
        assert result .usage .input_tokens ==5 







class TestDoctorCheck :
    def test_fields (self ):
        check =DoctorCheck (name ="api-key",passed =True ,message ="API key found")
        assert check .name =="api-key"
        assert check .passed is True 
        assert check .message =="API key found"


class TestDoctorReport :
    def test_checks_field (self ):
        checks =[
        DoctorCheck (name ="connectivity",passed =True ,message ="OK"),
        DoctorCheck (name ="auth",passed =False ,message ="Missing API key"),
        ]
        report =DoctorReport (checks =checks )
        assert len (report .checks )==2 
        assert report .checks [0 ].name =="connectivity"
        assert report .checks [1 ].passed is False 







class TestBaseProviderUnimplemented :
    """An incomplete subclass that only inherits raises NotImplementedError."""

    def setup_method (self ):
        class IncompleteProvider (BaseProvider ):
            pass 

        self .provider =IncompleteProvider ()

    @pytest .mark .anyio 
    async def test_chat_raises_not_implemented (self ):
        request =ProviderRequest (
        messages =[ChatMessage (role ="user",content ="hi")],
        model ="gpt-4",
        )
        with pytest .raises (NotImplementedError ):
            await self .provider .chat (request )

    @pytest .mark .anyio 
    async def test_stream_chat_raises_not_implemented (self ):
        request =ProviderRequest (
        messages =[ChatMessage (role ="user",content ="hi")],
        model ="gpt-4",
        )
        with pytest .raises (NotImplementedError ):

            async for _ in self .provider .stream_chat (request ):
                pass 

    @pytest .mark .anyio 
    async def test_list_models_raises_not_implemented (self ):
        with pytest .raises (NotImplementedError ):
            await self .provider .list_models ()

    @pytest .mark .anyio 
    async def test_doctor_raises_not_implemented (self ):
        with pytest .raises (NotImplementedError ):
            await self .provider .doctor ()


class TestBaseProviderConcreteSubclass :
    """A fully implemented subclass can be called without error."""

    def setup_method (self ):
        class ConcreteProvider (BaseProvider ):
            async def chat (self ,request :ProviderRequest )->ProviderResult :
                return ProviderResult (content ="response")

            async def stream_chat (self ,request :ProviderRequest ):
                yield TextDelta (text ="hello")
                yield ResponseCompleted (content ="hello")

            async def list_models (self )->list [str ]:
                return ["model-a","model-b"]

            async def doctor (self )->DoctorReport :
                return DoctorReport (
                checks =[DoctorCheck (name ="ok",passed =True ,message ="all good")]
                )

        self .provider =ConcreteProvider ()

    @pytest .mark .anyio 
    async def test_chat_returns_provider_result (self ):
        request =ProviderRequest (
        messages =[ChatMessage (role ="user",content ="hi")],
        model ="model-a",
        )
        result =await self .provider .chat (request )
        assert isinstance (result ,ProviderResult )
        assert result .content =="response"

    @pytest .mark .anyio 
    async def test_stream_chat_yields_events (self ):
        request =ProviderRequest (
        messages =[ChatMessage (role ="user",content ="hi")],
        model ="model-a",
        )
        events =[]
        async for event in self .provider .stream_chat (request ):
            events .append (event )
        assert len (events )==2 
        assert isinstance (events [0 ],TextDelta )
        assert isinstance (events [1 ],ResponseCompleted )

    @pytest .mark .anyio 
    async def test_list_models_returns_list (self ):
        models =await self .provider .list_models ()
        assert models ==["model-a","model-b"]

    @pytest .mark .anyio 
    async def test_doctor_returns_report (self ):
        report =await self .provider .doctor ()
        assert isinstance (report ,DoctorReport )
        assert len (report .checks )==1 
        assert report .checks [0 ].passed is True 
