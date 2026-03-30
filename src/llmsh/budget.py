class ContextBudget :
    """Tracks the token budget for a conversation."""

    def __init__ (self ,context_length :int ,reserved_output :int )->None :
        self ._context_length =context_length 
        self ._reserved_output =reserved_output 
        self ._used =0 

    @property 
    def context_length (self )->int :
        return self ._context_length 

    @property 
    def reserved_output (self )->int :
        return self ._reserved_output 

    @property 
    def used (self )->int :
        return self ._used 

    @property 
    def remaining (self )->int :
        return max (0 ,self ._context_length -self ._reserved_output -self ._used )

    @property 
    def utilization (self )->float :
        input_budget =self ._context_length -self ._reserved_output 
        if input_budget <=0 :
            return 0.0 
        return self ._used /input_budget 

    def update_usage (self ,input_tokens :int )->None :
        self ._used =input_tokens 

    def would_fit (self ,estimated_tokens :int )->bool :
        return estimated_tokens <=self .remaining 

    def estimate_tokens (self ,text :str )->int :
        return len (text )//3 

    def reset (self )->None :
        self ._used =0 
