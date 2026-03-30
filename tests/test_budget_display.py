"""Tests for budget display in FooterBar."""

from llmsh .ui .screens import FooterBar 


class TestShowBudget :
    def test_show_budget_stores_text_and_refreshes (self ):
        bar =FooterBar ()
        bar .show_budget (0.45 )
        assert bar ._budget_text =="budget: 45%"

    def test_show_budget_zero_percent (self ):
        bar =FooterBar ()
        bar .show_budget (0.0 )
        assert bar ._budget_text =="budget: 0%"

    def test_show_budget_rounds_down (self ):
        bar =FooterBar ()
        bar .show_budget (0.999 )
        assert bar ._budget_text =="budget: 99%"

    def test_show_budget_hundred_percent (self ):
        bar =FooterBar ()
        bar .show_budget (1.0 )
        assert bar ._budget_text =="budget: 100%"


class TestClearBudget :
    def test_clear_budget_removes_text (self ):
        bar =FooterBar ()
        bar .show_budget (0.45 )
        bar .clear_budget ()
        assert bar ._budget_text ==""


class TestRefreshFooterWithBudget :
    def test_refresh_includes_budget_when_set (self ):
        bar =FooterBar ()
        bar ._tokens ="tokens: 500"
        bar .show_budget (0.25 )
        text =bar ._render_text ()
        assert "budget: 25%"in text 
        assert "tokens: 500"in text 
        assert "Ctrl+C cancel"in text 

    def test_refresh_excludes_budget_when_never_set (self ):
        bar =FooterBar ()
        text =bar ._render_text ()
        assert "budget"not in text 

    def test_refresh_excludes_budget_after_clear (self ):
        bar =FooterBar ()
        bar .show_budget (0.5 )
        bar .clear_budget ()
        text =bar ._render_text ()
        assert "budget"not in text 

    def test_budget_appears_after_tokens (self ):
        bar =FooterBar ()
        bar ._tokens ="tokens: 100"
        bar .show_budget (0.1 )
        text =bar ._render_text ()
        tokens_pos =text .index ("tokens: 100")
        budget_pos =text .index ("budget: 10%")
        assert budget_pos >tokens_pos 
