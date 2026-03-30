from pathlib import Path 

from llmsh import paths 


def test_config_path_is_absolute ()->None :
    assert paths .config_path ().is_absolute ()


def test_config_path_contains_llmsh ()->None :
    assert "llmsh"in str (paths .config_path ())


def test_sessions_dir_is_child_of_data_dir ()->None :
    assert paths .sessions_dir ().parent ==paths .data_dir ()


def test_data_dir_is_absolute ()->None :
    assert paths .data_dir ().is_absolute ()


def test_config_path_returns_path_type ()->None :
    assert isinstance (paths .config_path (),Path )


def test_data_dir_returns_path_type ()->None :
    assert isinstance (paths .data_dir (),Path )


def test_sessions_dir_returns_path_type ()->None :
    assert isinstance (paths .sessions_dir (),Path )
