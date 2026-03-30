class LlmshError (Exception ):
    pass 


class ConfigError (LlmshError ):
    pass 


class ProfileNotFoundError (LlmshError ):
    pass 


class EndpointNotFoundError (LlmshError ):
    pass 


class EndpointNotReachableError (LlmshError ):
    pass 


class AuthenticationError (LlmshError ):
    pass 


class ModelNotFoundError (LlmshError ):
    pass 


class StreamInterruptedError (LlmshError ):
    pass 


class ClipboardError (LlmshError ):
    pass 


class SessionNotFoundError (LlmshError ):
    pass 


class ProviderError (LlmshError ):
    def __init__ (self ,message :str ,error_type :str ="unknown")->None :
        super ().__init__ (message )
        self .error_type =error_type 
