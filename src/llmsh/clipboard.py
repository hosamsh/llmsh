import subprocess 

from llmsh .errors import ClipboardError 


def copy_to_clipboard (text :str )->None :
    try :
        import pyperclip 

        pyperclip .copy (text )
        return 
    except Exception :
        pass 

    try :
        subprocess .run (["wl-copy",text ],check =True ,capture_output =True )
        return 
    except (FileNotFoundError ,subprocess .CalledProcessError ):
        pass 

    try :
        subprocess .run (
        ["xclip","-selection","clipboard"],
        input =text .encode (),
        check =True ,
        capture_output =True ,
        )
        return 
    except (FileNotFoundError ,subprocess .CalledProcessError ):
        pass 

    try :
        subprocess .run (
        ["xsel","--clipboard","--input"],
        input =text .encode (),
        check =True ,
        capture_output =True ,
        )
        return 
    except (FileNotFoundError ,subprocess .CalledProcessError ):
        pass 

    raise ClipboardError (
    "Could not copy to clipboard. Tried: pyperclip, wl-copy, xclip, xsel. "
    "Install one of these tools to enable clipboard support."
    )
