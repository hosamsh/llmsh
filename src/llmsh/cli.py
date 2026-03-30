import typer 

from llmsh .commands .ask import ask 
from llmsh .commands .chat import chat 
from llmsh .commands .doctor import doctor 
from llmsh .commands .endpoint import endpoint_app 
from llmsh .commands .profile import profile_app 
from llmsh .commands .session import session_app 
from llmsh .commands .summarize import summarize 

app =typer .Typer (
help ="llmsh — a terminal CLI for LLMs",
add_completion =False ,
invoke_without_command =True ,
)

ask_app =typer .Typer (
help ="Send a single prompt and print the response",
invoke_without_command =True ,
)
ask_app .callback ()(ask )


@app .callback (invoke_without_command =True )
def main (ctx :typer .Context )->None :
    if ctx .invoked_subcommand is None :
        chat (profile =None )


app .add_typer (ask_app ,name ="ask")

summarize_app =typer .Typer (
help ="Analyze a file with instruction-guided chunked processing",
invoke_without_command =True ,
)
summarize_app .callback ()(summarize )
app .add_typer (summarize_app ,name ="summarize")

doctor_app =typer .Typer (
help ="Run diagnostics on the current configuration",
invoke_without_command =True ,
)
doctor_app .callback ()(doctor )
app .add_typer (doctor_app ,name ="doctor")

app .add_typer (endpoint_app ,name ="endpoint")
app .add_typer (profile_app ,name ="profile")
app .add_typer (session_app ,name ="session")


if __name__ =="__main__":
    app ()
