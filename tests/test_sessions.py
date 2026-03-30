"""Tests for session storage (sessions/store.py)."""

from datetime import UTC ,datetime ,timedelta 
from pathlib import Path 

import pytest 

from llmsh .errors import SessionNotFoundError 
from llmsh .models import ChatMessage ,SessionRecord ,UsageInfo 
from llmsh .sessions .store import SessionMeta ,SessionStore 


class TestCreate :
    def test_returns_session_record_with_uuid_id (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        assert isinstance (session ,SessionRecord )
        assert len (session .id )>0 

    def test_id_is_unique_across_calls (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        s1 =store .create (profile ="default",model ="gpt-4")
        s2 =store .create (profile ="default",model ="gpt-4")
        assert s1 .id !=s2 .id 

    def test_profile_and_model_are_set (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="myprofile",model ="llama3")
        assert session .profile =="myprofile"
        assert session .model =="llama3"

    def test_messages_is_empty (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        assert session .messages ==[]

    def test_usage_is_empty (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        assert session .usage ==[]

    def test_timestamps_are_set (self ,tmp_path :Path ):
        before =datetime .now (UTC )
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        after =datetime .now (UTC )
        assert before <=session .created_at <=after 
        assert before <=session .updated_at <=after 

    def test_title_is_placeholder (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        assert isinstance (session .title ,str )
        assert len (session .title )>0 


class TestSaveAndLoad :
    def test_round_trip_preserves_all_fields (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        session .messages .append (
        ChatMessage (role ="user",content ="Hello world")
        )
        session .usage .append (
        UsageInfo (input_tokens =5 ,output_tokens =10 ,total_tokens =15 )
        )
        store .save (session )

        loaded =store .load (session .id )
        assert loaded .id ==session .id 
        assert loaded .profile ==session .profile 
        assert loaded .model ==session .model 
        assert loaded .title ==session .title 
        assert len (loaded .messages )==1 
        assert loaded .messages [0 ].content =="Hello world"
        assert loaded .messages [0 ].role =="user"
        assert len (loaded .usage )==1 
        assert loaded .usage [0 ].input_tokens ==5 

    def test_save_creates_sessions_dir_if_missing (self ,tmp_path :Path ):
        sessions_path =tmp_path /"nested"/"sessions"
        store =SessionStore (sessions_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )
        assert sessions_path .exists ()

    def test_save_writes_json_file (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )
        json_file =tmp_path /f"{session .id }.json"
        assert json_file .exists ()

    def test_load_deserializes_datetime_fields (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )

        loaded =store .load (session .id )
        assert loaded .created_at .tzinfo is not None 
        assert loaded .updated_at .tzinfo is not None 


class TestSaveTwice :
    def test_save_twice_does_not_duplicate_index_entry (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )
        store .save (session )
        entries =store .list ()
        assert len (entries )==1 

    def test_save_twice_updated_at_advances (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )

        first_updated_at =store .load (session .id ).updated_at 


        session .updated_at =session .updated_at +timedelta (seconds =1 )
        store .save (session )

        second_updated_at =store .load (session .id ).updated_at 
        assert second_updated_at >first_updated_at 


class TestList :
    def test_list_empty_directory_returns_empty_list (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        assert store .list ()==[]

    def test_list_after_saving_two_sessions_returns_two_entries (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        s1 =store .create (profile ="default",model ="gpt-4")
        store .save (s1 )
        s2 =store .create (profile ="other",model ="llama3")
        store .save (s2 )

        entries =store .list ()
        assert len (entries )==2 

    def test_list_returns_session_meta_instances (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )

        entries =store .list ()
        assert all (isinstance (e ,SessionMeta )for e in entries )

    def test_list_meta_has_correct_fields (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="myprofile",model ="mymodel")
        store .save (session )

        entries =store .list ()
        meta =entries [0 ]
        assert meta .id ==session .id 
        assert meta .profile =="myprofile"
        assert meta .model =="mymodel"
        assert isinstance (meta .updated_at ,datetime )

    def test_list_does_not_load_full_session_files (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )

        json_file =tmp_path /f"{session .id }.json"
        json_file .unlink ()

        entries =store .list ()
        assert len (entries )==1 


class TestDelete :
    def test_delete_removes_json_file (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )
        json_file =tmp_path /f"{session .id }.json"
        assert json_file .exists ()

        store .delete (session .id )
        assert not json_file .exists ()

    def test_delete_removes_index_entry (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )
        assert len (store .list ())==1 

        store .delete (session .id )
        assert store .list ()==[]

    def test_delete_only_removes_target_session (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        s1 =store .create (profile ="default",model ="gpt-4")
        store .save (s1 )
        s2 =store .create (profile ="default",model ="gpt-4")
        store .save (s2 )

        store .delete (s1 .id )
        remaining =store .list ()
        assert len (remaining )==1 
        assert remaining [0 ].id ==s2 .id 


class TestLoad :
    def test_load_missing_id_raises_session_not_found_error (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        with pytest .raises (SessionNotFoundError ):
            store .load ("nonexistent-id")


class TestTitleGeneration :
    def test_short_message_is_used_as_title_unchanged (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        title =store ._generate_title ("Short message")
        assert title =="Short message"

    def test_long_message_is_truncated_at_word_boundary (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        long_msg ="word "*20 
        title =store ._generate_title (long_msg )
        assert len (title )<=53 
        assert title .endswith ("...")

    def test_truncation_does_not_cut_mid_word (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        long_msg ="word "*20 
        title =store ._generate_title (long_msg )

        base =title [:-3 ]
        assert not base .endswith (" ")

        last_word =base .split ()[-1 ]
        assert last_word in long_msg .split ()

    def test_title_generated_from_first_user_message_on_save (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        long_msg ="first user message "*5 
        session .messages .append (ChatMessage (role ="user",content =long_msg ))
        store .save (session )

        loaded =store .load (session .id )
        assert loaded .title !="New session"
        assert len (loaded .title )<=53 

    def test_no_user_message_keeps_placeholder_title (self ,tmp_path :Path ):
        store =SessionStore (tmp_path )
        session =store .create (profile ="default",model ="gpt-4")
        store .save (session )

        loaded =store .load (session .id )
        assert loaded .title =="New session"
