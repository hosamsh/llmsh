"""Tests for the summarize command."""

import json 

import pytest 
from typer .testing import CliRunner 

from llmsh .cli import app 
from llmsh .commands .summarize import _read_directory 
from llmsh .errors import LlmshError 

runner =CliRunner ()


class TestSummarizeValidFile :
    def test_calls_engine_and_outputs_answer (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"test.txt"
        f .write_text ("file content here")

        async def fake_summarize (**kwargs ):
            return ("the answer",1 ,0 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(
        None ,
        "test-model",
        "default",
        "file content here",
        8192 ,
        1024 ,
        ),
        )

        result =runner .invoke (app ,["summarize",str (f ),"find bugs"])

        assert result .exit_code ==0 
        assert "the answer"in result .stdout 


class TestSummarizeNonexistentFile :
    def test_error_on_stderr_exit_1 (self ):
        result =runner .invoke (app ,["summarize","/no/such/file.txt","find bugs"])

        assert result .exit_code ==1 
        assert "not found"in result .stderr .lower ()


class TestSummarizeJsonOutput :
    def test_json_has_expected_keys (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"data.log"
        f .write_text ("log data")

        async def fake_summarize (**kwargs ):
            return ("json answer",3 ,0 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(
        None ,
        "test-model",
        "myprofile",
        "log data",
        8192 ,
        1024 ,
        ),
        )

        result =runner .invoke (
        app ,["summarize","--json",str (f ),"summarize logs"]
        )

        assert result .exit_code ==0 
        data =json .loads (result .stdout )
        assert data ["answer"]=="json answer"
        assert data ["model"]=="test-model"
        assert data ["profile"]=="myprofile"
        assert data ["file"]==str (f )
        assert data ["chunks_processed"]==3 
        assert data ["truncated_calls"]==0 

    def test_json_includes_truncated_calls (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"data.log"
        f .write_text ("log data")

        async def fake_summarize (**kwargs ):
            return ("json answer",3 ,2 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(
        None ,
        "test-model",
        "myprofile",
        "log data",
        8192 ,
        1024 ,
        ),
        )

        result =runner .invoke (
        app ,["summarize","--json",str (f ),"summarize logs"]
        )

        assert result .exit_code ==0 
        data =json .loads (result .stdout )
        assert data ["truncated_calls"]==2 


class TestSummarizeProfileFlag :
    def test_profile_passed_to_resolver (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"test.txt"
        f .write_text ("content")
        received ={}

        def fake_resolve (file ,profile ):
            received ["profile"]=profile 
            return (None ,"m","p","content",4096 ,1024 )

        async def fake_summarize (**kwargs ):
            return ("ok",1 ,0 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",fake_resolve 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )

        runner .invoke (app ,["summarize","--profile","custom",str (f ),"analyze"])

        assert received ["profile"]=="custom"


class TestSummarizeContextLengthFallback :
    def test_falls_back_to_4096_with_stderr_warning (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"test.txt"
        f .write_text ("content")



        from llmsh .config import AppConfig 
        from llmsh .models import EndpointConfig ,ModelCapabilities ,ProfileConfig 

        config =AppConfig (
        current_profile ="default",
        endpoints ={
        "local":EndpointConfig (
        name ="local",
        base_url ="http://localhost:8006/v1",
        auth_mode ="none",
        )
        },
        profiles ={
        "default":ProfileConfig (
        name ="default",
        endpoint ="local",
        model ="test-model",
        capabilities =ModelCapabilities (),
        )
        },
        )
        monkeypatch .setattr ("llmsh.commands.summarize.load_config",lambda :config )


        from tests .conftest import BudgetStubProvider 

        provider =BudgetStubProvider (
        model_info_error =Exception ("model info failed")
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._make_provider",lambda ep :provider 
        )

        async def fake_summarize (**kwargs ):
            received_ctx [0 ]=kwargs ["context_length"]
            return ("answer",1 ,0 )

        received_ctx =[None ]
        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )

        result =runner .invoke (app ,["summarize",str (f ),"analyze"])

        assert result .exit_code ==0 
        assert received_ctx [0 ]==4096 
        assert "4096"in result .stderr or "warning"in result .stderr .lower ()


class TestSummarizeTruncationWarning :
    def test_warns_on_truncation (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"test.txt"
        f .write_text ("content")

        async def fake_summarize (**kwargs ):
            return ("answer",3 ,2 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(None ,"m","p","content",4096 ,1024 ),
        )

        result =runner .invoke (app ,["summarize",str (f ),"analyze"])

        assert result .exit_code ==0 
        assert "2 response(s) were truncated"in result .stderr 

    def test_no_warning_when_no_truncation (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"test.txt"
        f .write_text ("content")

        async def fake_summarize (**kwargs ):
            return ("answer",3 ,0 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(None ,"m","p","content",4096 ,1024 ),
        )

        result =runner .invoke (app ,["summarize",str (f ),"analyze"])

        assert result .exit_code ==0 
        assert "truncated"not in result .stderr 


class TestSummarizeProviderError :
    def test_provider_error_exits_1_with_stderr (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"test.txt"
        f .write_text ("content")

        async def fake_summarize (**kwargs ):
            raise RuntimeError ("provider connection refused")

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(None ,"m","p","content",4096 ,1024 ),
        )

        result =runner .invoke (app ,["summarize",str (f ),"analyze"])

        assert result .exit_code ==1 
        assert "provider connection refused"in result .stderr .lower ()


class TestSummarizeFileReadError :
    def test_unreadable_file_exits_1_with_stderr (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"test.txt"
        f .write_text ("content")
        f .chmod (0o000 )

        result =runner .invoke (app ,["summarize",str (f ),"analyze"])

        assert result .exit_code ==1 
        assert "cannot read"in result .stderr .lower ()

        f .chmod (0o644 )


class TestSummarizePlanToStderr :
    def test_plan_displayed_on_stderr (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"test.txt"
        f .write_text ("content")

        async def fake_summarize (**kwargs ):
            from llmsh .summarize import SummarizePlan 

            cb =kwargs .get ("on_plan")
            if cb :
                cb (SummarizePlan (
                total_chunks =10 ,
                estimated_reduce_rounds =2 ,
                estimated_total_calls =15 ,
                ))
            return ("done",10 ,0 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(None ,"m","p","content",4096 ,1024 ),
        )

        result =runner .invoke (app ,["summarize",str (f ),"analyze"])

        assert result .exit_code ==0 
        assert "Plan: 10 chunks"in result .stderr 
        assert "~2 reduce rounds"in result .stderr 
        assert "~15 total LLM calls"in result .stderr 


class TestSummarizeProgressToStderr :
    def test_map_and_reduce_progress_on_stderr (self ,monkeypatch ,tmp_path ):
        f =tmp_path /"test.txt"
        f .write_text ("content")

        async def fake_summarize (**kwargs ):
            cb =kwargs .get ("on_progress")
            if cb :
                cb ("map",1 ,3 )
                cb ("map",2 ,3 )
                cb ("map",3 ,3 )
                cb ("reduce-1",1 ,2 )
                cb ("reduce-1",2 ,2 )
                cb ("reduce-2",1 ,1 )
            return ("done",3 ,0 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(None ,"m","p","content",4096 ,1024 ),
        )

        result =runner .invoke (app ,["summarize",str (f ),"analyze"])

        assert result .exit_code ==0 
        assert "Map: chunk 1/3"in result .stderr 
        assert "Map: chunk 2/3"in result .stderr 
        assert "Map: chunk 3/3"in result .stderr 
        assert "Reduce round 1: batch 1/2"in result .stderr 
        assert "Reduce round 1: batch 2/2"in result .stderr 
        assert "Reduce round 2: batch 1/1"in result .stderr 

        assert result .stdout .strip ()=="done"


class TestReadDirectory :
    def test_reads_all_text_files_with_headers (self ,tmp_path ):
        (tmp_path /"a.txt").write_text ("alpha")
        (tmp_path /"b.txt").write_text ("beta")

        result =_read_directory (tmp_path )

        assert "--- FILE: a.txt ---"in result 
        assert "alpha"in result 
        assert "--- FILE: b.txt ---"in result 
        assert "beta"in result 

    def test_reads_nested_files_with_relative_paths (self ,tmp_path ):
        sub =tmp_path /"sub"
        sub .mkdir ()
        (sub /"nested.txt").write_text ("nested content")

        result =_read_directory (tmp_path )

        assert "--- FILE: sub/nested.txt ---"in result 
        assert "nested content"in result 

    def test_skips_hidden_directories (self ,tmp_path ):
        hidden =tmp_path /".git"
        hidden .mkdir ()
        (hidden /"config").write_text ("secret")
        (tmp_path /"visible.txt").write_text ("visible")

        result =_read_directory (tmp_path )

        assert "visible"in result 
        assert "secret"not in result 
        assert ".git"not in result 

    def test_skips_binary_files (self ,tmp_path ):
        (tmp_path /"text.txt").write_text ("readable")
        (tmp_path /"binary.bin").write_bytes (b"\x00\x01\x80\xff")

        result =_read_directory (tmp_path )

        assert "readable"in result 
        assert "binary.bin"not in result 

    def test_empty_directory_raises_error (self ,tmp_path ):
        with pytest .raises (LlmshError ,match ="No readable text files"):
            _read_directory (tmp_path )

    def test_files_separated_by_blank_line (self ,tmp_path ):
        (tmp_path /"a.txt").write_text ("alpha")
        (tmp_path /"b.txt").write_text ("beta")

        result =_read_directory (tmp_path )

        assert "--- FILE: a.txt ---\nalpha\n\n--- FILE: b.txt ---\nbeta"in result 


class TestSummarizeDirectory :
    def test_directory_path_works (self ,monkeypatch ,tmp_path ):
        (tmp_path /"file.txt").write_text ("dir content")

        received ={}

        async def fake_summarize (**kwargs ):
            received ["file_text"]=kwargs ["file_text"]
            return ("dir answer",1 ,0 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(
        None ,
        "test-model",
        "default",
        _read_directory (tmp_path ),
        8192 ,
        1024 ,
        ),
        )

        result =runner .invoke (app ,["summarize",str (tmp_path ),"analyze"])

        assert result .exit_code ==0 
        assert "dir answer"in result .stdout 

    def test_empty_directory_exits_1 (self ,tmp_path ):
        result =runner .invoke (app ,["summarize",str (tmp_path ),"analyze"])

        assert result .exit_code ==1 
        assert "no readable text files"in result .stderr .lower ()

    def test_json_output_shows_directory_path (self ,monkeypatch ,tmp_path ):
        (tmp_path /"file.txt").write_text ("content")

        async def fake_summarize (**kwargs ):
            return ("answer",1 ,0 )

        monkeypatch .setattr (
        "llmsh.commands.summarize._run_summarize",fake_summarize 
        )
        monkeypatch .setattr (
        "llmsh.commands.summarize._resolve_setup",
        lambda file ,profile :(
        None ,
        "test-model",
        "default",
        "content",
        8192 ,
        1024 ,
        ),
        )

        result =runner .invoke (
        app ,["summarize","--json",str (tmp_path ),"analyze"]
        )

        assert result .exit_code ==0 
        data =json .loads (result .stdout )
        assert data ["file"]==str (tmp_path )
