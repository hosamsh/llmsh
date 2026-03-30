from datetime import UTC ,datetime 
from typing import Any ,Literal 

from pydantic import BaseModel ,Field 


class ModelInfo (BaseModel ):
    id :str 
    context_length :int |None =None 


class EndpointConfig (BaseModel ):
    name :str 
    base_url :str 
    auth_mode :Literal ["api_key","none"]
    api_key_env :str |None =None 
    api_key_value :str |None =None 
    provider_type :Literal ["openai_compatible"]="openai_compatible"
    verify_ssl :bool =True 
    timeout_seconds :float =60.0 


class ModelCapabilities (BaseModel ):
    streaming :bool =True 
    token_usage :bool =True 
    multimodal_input :bool =False 
    tool_calling :bool =True 
    structured_output :bool =False 


class ProfileConfig (BaseModel ):
    name :str 
    endpoint :str 
    model :str 
    capabilities :ModelCapabilities 
    max_tokens :int |None =None 


class ToolParameter (BaseModel ):
    name :str 
    type :str 
    description :str 
    required :bool =True 


class ToolDefinition (BaseModel ):
    name :str 
    description :str 
    parameters :list [ToolParameter ]


class ToolCall (BaseModel ):
    id :str 
    name :str 
    arguments :dict [str ,Any ]


class ChatMessage (BaseModel ):
    role :Literal ["system","user","assistant"]
    content :str 
    created_at :datetime =Field (default_factory =lambda :datetime .now (UTC ))


class UsageInfo (BaseModel ):
    input_tokens :int |None =None 
    output_tokens :int |None =None 
    total_tokens :int |None =None 


class SessionRecord (BaseModel ):
    id :str 
    title :str 
    profile :str 
    model :str 
    created_at :datetime 
    updated_at :datetime 
    messages :list [ChatMessage ]
    usage :list [UsageInfo ]


class CapabilityTestResult (BaseModel ):
    name :str 
    passed :bool |None 
    message :str 
    duration_ms :int |None =None 


class CapabilityReport (BaseModel ):
    profile_name :str 
    model :str =""
    endpoint :str =""
    results :list [CapabilityTestResult ]
    tested_at :datetime 
