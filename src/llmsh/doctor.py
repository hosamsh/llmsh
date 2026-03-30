from __future__ import annotations 

import time 
from datetime import UTC ,datetime 

from llmsh .models import (
CapabilityReport ,
CapabilityTestResult ,
ChatMessage ,
EndpointConfig ,
ProfileConfig ,
ToolDefinition ,
ToolParameter ,
)
from llmsh .providers .base import (
BaseProvider ,
DoctorCheck ,
DoctorReport ,
ProviderRequest ,
ReasoningDelta ,
ResponseCompleted ,
TextDelta ,
ToolCallEvent ,
)


_capability_cache :dict [str ,CapabilityReport ]={}


def _check (name :str ,passed :bool ,message :str )->DoctorCheck :
    return DoctorCheck (name =name ,passed =passed ,message =message )


async def run_doctor (
profile :ProfileConfig ,
endpoint :EndpointConfig ,
provider :BaseProvider ,
)->DoctorReport :
    checks =[_check ("config_valid",True ,"Configuration is valid")]

    try :
        report =await provider .doctor ()
        first =report .checks [0 ]if report .checks else None 
        checks .append (_check (
        "endpoint_reachable",
        first .passed if first else False ,
        first .message if first else "No checks returned",
        ))
    except Exception as exc :
        checks .append (_check ("endpoint_reachable",False ,str (exc )))

    models :list [str ]=[]
    try :
        models =await provider .list_models ()
        checks .append (_check ("models_listable",True ,f"Found {len (models )} models"))
    except Exception as exc :
        checks .append (_check ("models_listable",False ,str (exc )))

    if models :
        present =profile .model in models 
        msg =(
        f"Model '{profile .model }' found"
        if present 
        else f"Model '{profile .model }' not in available models: {models }"
        )
        checks .append (_check ("model_present",present ,msg ))
    else :
        checks .append (
        _check ("model_present",False ,"Cannot verify - model list unavailable")
        )

    return DoctorReport (checks =checks )







def _classify_speed (duration_ms :int )->str :
    if duration_ms <2000 :
        return "fast"
    elif duration_ms <5000 :
        return "moderate"
    else :
        return "slow"


def get_cached_report (profile_name :str )->CapabilityReport |None :
    return _capability_cache .get (profile_name )


def is_cache_valid (profile :ProfileConfig )->bool :
    report =_capability_cache .get (profile .name )
    if report is None :
        return False 
    return report .model ==profile .model and report .endpoint ==profile .endpoint 


def format_capability_summary (report :CapabilityReport )->str :
    """Format a one-line capability summary."""
    parts :list [str ]=[]
    for r in report .results :
        if r .name =="qa_response":
            parts .append (f"qa:{'ok'if r .passed else 'fail'}")
        elif r .name =="tool_calling":
            if r .passed is None :
                parts .append ("tools:disabled")
            else :
                parts .append (f"tools:{'yes'if r .passed else 'no'}")
        elif r .name =="tool_inference":
            if r .passed is not None :
                parts .append (f"inference:{'yes'if r .passed else 'no'}")
        elif r .name =="reasoning":
            parts .append (f"reasoning:{'yes'if r .passed else 'no'}")
        elif r .name =="speed":
            parts .append (f"speed:{r .message }")
    return "  ".join (parts )


async def run_capability_tests (
profile :ProfileConfig ,
endpoint :EndpointConfig ,
provider :BaseProvider ,
)->CapabilityReport :
    results :list [CapabilityTestResult ]=[]


    qa_result ,qa_duration =await _test_qa (provider ,profile .model )
    results .append (qa_result )


    tool_result :CapabilityTestResult |None =None 
    if profile .capabilities .tool_calling :
        tool_result =await _test_tool_explicit (provider ,profile .model )
        results .append (tool_result )
    else :
        results .append (CapabilityTestResult (
        name ="tool_calling",passed =None ,message ="Disabled",
        ))


    if tool_result and tool_result .passed :
        results .append (await _test_tool_implicit (provider ,profile .model ))
    else :
        reason =(
        "Skipped (explicit tool test failed)"
        if tool_result 
        else "Disabled"
        )
        results .append (CapabilityTestResult (
        name ="tool_inference",passed =None ,message =reason ,
        ))


    results .append (await _test_reasoning (provider ,profile .model ))


    if qa_duration is not None :
        speed =_classify_speed (qa_duration )
        results .append (CapabilityTestResult (
        name ="speed",passed =True ,message =f"{speed } ({qa_duration }ms)",
        duration_ms =qa_duration ,
        ))

    report =CapabilityReport (
    profile_name =profile .name ,
    model =profile .model ,
    endpoint =profile .endpoint ,
    results =results ,
    tested_at =datetime .now (UTC ),
    )
    _capability_cache [profile .name ]=report 
    return report 


async def _test_qa (
provider :BaseProvider ,model :str ,
)->tuple [CapabilityTestResult ,int |None ]:
    prompt ="What is the capital of France? Reply with just the city name."
    request =ProviderRequest (
    messages =[ChatMessage (role ="user",content =prompt )],
    model =model ,
    )
    start =time .monotonic ()
    try :
        text =await _collect_text (provider ,request )
        duration_ms =int ((time .monotonic ()-start )*1000 )
        passed ="paris"in text .lower ()
        if passed :
            message =f"Response: {text [:80 ]}"
        else :
            message =f"Expected 'paris', got: {text [:80 ]}"
        return (
        CapabilityTestResult (
        name ="qa_response",passed =passed ,message =message ,
        duration_ms =duration_ms ,
        ),
        duration_ms ,
        )
    except Exception as exc :
        duration_ms =int ((time .monotonic ()-start )*1000 )
        return (
        CapabilityTestResult (
        name ="qa_response",passed =False ,message =str (exc ),
        duration_ms =duration_ms ,
        ),
        duration_ms ,
        )


async def _test_tool_explicit (
provider :BaseProvider ,model :str ,
)->CapabilityTestResult :
    tool =ToolDefinition (
    name ="get_current_time",
    description ="Get the current time",
    parameters =[],
    )
    request =ProviderRequest (
    messages =[ChatMessage (
    role ="user",
    content ="Please call the get_current_time tool to check the time.",
    )],
    model =model ,
    tools =[tool ],
    )
    start =time .monotonic ()
    try :
        tool_calls =await _collect_tool_calls (provider ,request )
        duration_ms =int ((time .monotonic ()-start )*1000 )
        called =any (tc .tool_call .name =="get_current_time"for tc in tool_calls )
        if called :
            return CapabilityTestResult (
            name ="tool_calling",passed =True ,
            message =f"Tool called ({duration_ms }ms)",duration_ms =duration_ms ,
            )
        return CapabilityTestResult (
        name ="tool_calling",passed =False ,
        message ="Model did not call the tool",duration_ms =duration_ms ,
        )
    except Exception as exc :
        duration_ms =int ((time .monotonic ()-start )*1000 )
        return CapabilityTestResult (
        name ="tool_calling",passed =False ,message =str (exc ),
        duration_ms =duration_ms ,
        )


async def _test_tool_implicit (
provider :BaseProvider ,model :str ,
)->CapabilityTestResult :
    tool =ToolDefinition (
    name ="check_weather",
    description ="Check weather for a city",
    parameters =[ToolParameter (
    name ="city",type ="string",description ="City name",
    )],
    )
    request =ProviderRequest (
    messages =[ChatMessage (
    role ="user",
    content ="I'm packing for a trip to Tokyo tomorrow, what should I bring?",
    )],
    model =model ,
    tools =[tool ],
    )
    start =time .monotonic ()
    try :
        tool_calls =await _collect_tool_calls (provider ,request )
        duration_ms =int ((time .monotonic ()-start )*1000 )
        match =any (
        tc .tool_call .name =="check_weather"
        and "tokyo"in str (tc .tool_call .arguments ).lower ()
        for tc in tool_calls 
        )
        if match :
            return CapabilityTestResult (
            name ="tool_inference",passed =True ,
            message =f"Tool inferred ({duration_ms }ms)",duration_ms =duration_ms ,
            )
        return CapabilityTestResult (
        name ="tool_inference",passed =False ,
        message ="Model did not call the tool",duration_ms =duration_ms ,
        )
    except Exception as exc :
        duration_ms =int ((time .monotonic ()-start )*1000 )
        return CapabilityTestResult (
        name ="tool_inference",passed =False ,message =str (exc ),
        duration_ms =duration_ms ,
        )


async def _test_reasoning (
provider :BaseProvider ,model :str ,
)->CapabilityTestResult :
    request =ProviderRequest (
    messages =[ChatMessage (role ="user",content ="What is 15 * 37?")],
    model =model ,
    )
    start =time .monotonic ()
    try :
        found_reasoning =False 
        async for event in provider .stream_chat (request ):
            if isinstance (event ,ReasoningDelta ):
                found_reasoning =True 
        duration_ms =int ((time .monotonic ()-start )*1000 )
        if found_reasoning :
            return CapabilityTestResult (
            name ="reasoning",passed =True ,
            message =f"Reasoning detected ({duration_ms }ms)",
            duration_ms =duration_ms ,
            )
        return CapabilityTestResult (
        name ="reasoning",passed =False ,
        message ="No reasoning output",duration_ms =duration_ms ,
        )
    except Exception as exc :
        duration_ms =int ((time .monotonic ()-start )*1000 )
        return CapabilityTestResult (
        name ="reasoning",passed =False ,message =str (exc ),
        duration_ms =duration_ms ,
        )


async def _collect_text (provider :BaseProvider ,request :ProviderRequest )->str :
    chunks :list [str ]=[]
    async for event in provider .stream_chat (request ):
        if isinstance (event ,TextDelta ):
            chunks .append (event .text )
        elif isinstance (event ,ResponseCompleted ):
            if event .content :
                return event .content 
    return "".join (chunks )


async def _collect_tool_calls (
provider :BaseProvider ,request :ProviderRequest ,
)->list [ToolCallEvent ]:
    calls :list [ToolCallEvent ]=[]
    async for event in provider .stream_chat (request ):
        if isinstance (event ,ToolCallEvent ):
            calls .append (event )
    return calls 
