from llmsh import constants 


def test_app_name_is_defined ()->None :
    assert constants .APP_NAME =="llmsh"


def test_config_filename_is_defined ()->None :
    assert isinstance (constants .CONFIG_FILENAME ,str )
    assert len (constants .CONFIG_FILENAME )>0 


def test_sessions_dir_name_is_defined ()->None :
    assert isinstance (constants .SESSIONS_DIR_NAME ,str )
    assert len (constants .SESSIONS_DIR_NAME )>0 
