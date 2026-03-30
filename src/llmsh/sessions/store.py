import json 
import uuid 
from datetime import UTC ,datetime 
from pathlib import Path 

from pydantic import BaseModel 

from llmsh .errors import SessionNotFoundError 
from llmsh .models import SessionRecord 


class SessionMeta (BaseModel ):
    id :str 
    title :str 
    profile :str 
    model :str 
    updated_at :datetime 


class SessionStore :
    def __init__ (self ,sessions_dir :Path )->None :
        self ._dir =sessions_dir 

    def create (self ,profile :str ,model :str )->SessionRecord :
        now =datetime .now (UTC )
        return SessionRecord (
        id =str (uuid .uuid4 ()),
        title ="New session",
        profile =profile ,
        model =model ,
        created_at =now ,
        updated_at =now ,
        messages =[],
        usage =[],
        )

    def save (self ,session :SessionRecord )->None :
        self ._dir .mkdir (parents =True ,exist_ok =True )

        first_user =next (
        (m .content for m in session .messages if m .role =="user"),None 
        )
        if first_user :
            session .title =self ._generate_title (first_user )

        (self ._dir /f"{session .id }.json").write_text (
        session .model_dump_json (),encoding ="utf-8"
        )

        self ._update_index (session )

    def load (self ,session_id :str )->SessionRecord :
        path =self ._dir /f"{session_id }.json"
        if not path .exists ():
            path =self ._resolve_prefix (session_id )
        return SessionRecord .model_validate_json (path .read_text (encoding ="utf-8"))

    def _resolve_prefix (self ,prefix :str )->Path :
        matches =[
        p for p in self ._dir .glob ("*.json")
        if p .stem !="index"and p .stem .startswith (prefix )
        ]
        if len (matches )==1 :
            return matches [0 ]
        raise SessionNotFoundError (prefix )

    def list (self )->list [SessionMeta ]:
        index_path =self ._dir /"index.json"
        if not index_path .exists ():
            return []
        raw =json .loads (index_path .read_text (encoding ="utf-8"))
        return [SessionMeta .model_validate (entry )for entry in raw ]

    def delete (self ,session_id :str )->None :
        path =self ._dir /f"{session_id }.json"
        if not path .exists ():
            path =self ._resolve_prefix (session_id )
        path .unlink ()

        index_path =self ._dir /"index.json"
        if not index_path .exists ():
            return 
        entries =json .loads (index_path .read_text (encoding ="utf-8"))
        entries =[e for e in entries if not e ["id"].startswith (session_id )]
        index_path .write_text (json .dumps (entries ),encoding ="utf-8")

    def _generate_title (self ,first_message :str )->str :
        if len (first_message )<=50 :
            return first_message 
        cut =first_message [:50 ]
        space =cut .rfind (" ")
        return (cut [:space ]if space >0 else cut )+"..."

    def _update_index (self ,session :SessionRecord )->None :
        index_path =self ._dir /"index.json"
        entries =(
        json .loads (index_path .read_text (encoding ="utf-8"))
        if index_path .exists ()else []
        )
        meta_dict =json .loads (
        SessionMeta (
        id =session .id ,title =session .title ,profile =session .profile ,
        model =session .model ,updated_at =session .updated_at ,
        ).model_dump_json ()
        )
        for i ,entry in enumerate (entries ):
            if entry ["id"]==session .id :
                entries [i ]=meta_dict 
                break 
        else :
            entries .append (meta_dict )
        index_path .write_text (json .dumps (entries ),encoding ="utf-8")
