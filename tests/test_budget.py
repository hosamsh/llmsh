from llmsh .budget import ContextBudget 


def test_initial_state ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    assert b .context_length ==8000 
    assert b .reserved_output ==2000 
    assert b .used ==0 
    assert b .remaining ==6000 


def test_update_usage_sets_used ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    b .update_usage (1500 )
    assert b .used ==1500 


def test_remaining_decreases_after_update ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    b .update_usage (1000 )
    assert b .remaining ==5000 


def test_utilization_calculation ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    b .update_usage (3000 )

    assert b .utilization ==0.5 


def test_would_fit_true_when_room ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    b .update_usage (1000 )

    assert b .would_fit (4999 )is True 
    assert b .would_fit (5000 )is True 


def test_would_fit_false_when_no_room ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    b .update_usage (1000 )

    assert b .would_fit (5001 )is False 


def test_estimate_tokens ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    text ="a"*99 
    assert b .estimate_tokens (text )==33 


def test_estimate_tokens_empty_string ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    assert b .estimate_tokens ("")==0 


def test_reset_clears_usage ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    b .update_usage (3000 )
    assert b .used ==3000 
    b .reset ()
    assert b .used ==0 
    assert b .remaining ==6000 


def test_over_budget_remaining_is_zero ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    b .update_usage (7000 )

    assert b .remaining ==0 


def test_over_budget_utilization_above_one ()->None :
    b =ContextBudget (context_length =8000 ,reserved_output =2000 )
    b .update_usage (7000 )

    assert b .utilization >1.0 


def test_zero_input_budget_remaining_is_zero ()->None :
    b =ContextBudget (context_length =4000 ,reserved_output =4000 )
    assert b .remaining ==0 


def test_zero_input_budget_utilization_is_zero ()->None :
    b =ContextBudget (context_length =4000 ,reserved_output =4000 )

    assert b .utilization ==0.0 
