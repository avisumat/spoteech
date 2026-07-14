import config
from typing import TypedDict,Annotated,Literal,List
from enum import Enum
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
import datetime
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from spotify_client import search_spotify_api
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field,field_validator

#the below schema is what LLM is told to output and follow this exact schema
#has no relation to DJState whatsoever, this is purely for LLM to look at ....
class MusicIntent(BaseModel):
    q: str = Field(
        description="The optimized Spotify search query."
        "If the user asks a question completely unrelated to music, audio, or podcasts (e.g., coding help, math, general chat), you MUST output 'OUT_OF_DOMAIN' for the q field. Do NOT use this just because a music request is vague. ONLY use this if music/audio is not the topic."
    )
    
    intent: Literal["individual", "queue", "album", "artist", "OUT_OF_DOMAIN"] = Field(
        description="CRITICAL: If the user asks for a 'playlist', 'songs', a 'vibe', or a 'mix', choose 'queue'. ONLY choose 'individual' if they ask for ONE specific track. If the user's request has NOTHING to do with music or audio (e.g., coding help, math, general chat), you MUST choose 'OUT_OF_DOMAIN'."
    )
    
    # This stops the LLM from outputting a stringified list.
    search_types: List[str] = Field(
        description="An array of strings. ONLY use a combination of: 'track', 'playlist', 'album', 'artist'."
    )
    
    ui_mood: Literal["energetic", "chill", "melancholy", "focus", "neutral"] = Field(
        description="the visual UI theme"
        "STRICT RULE: You must pick exactly one of the 5 allowed values: energetic, chill, melancholy, focus, neutral. NEVER invent your own word. NEVER use the word 'romantic'." \
        "you must have to map any other ui_mood to exactly one of the allowed values e.g map 'romantic' or 'lo-fi'  to 'chill' "
    )
    
    ai_thought: str = Field(
        description="Extremely concise reasoning. Maximum 10 words."
    )
    @field_validator("ui_mood",mode="before")
    @classmethod
    def validate_mood(cls, v:str)->str:
        allowed_moods =["energetic","chill","melancholy","focus","neutral"]
        cleaned_v = v.lower().strip()

        if cleaned_v not in allowed_moods:
            print(f"LLM hallucinated mood '{v}', hence defaulting to neutral...")
            return "neutral"
        
        return cleaned_v
    

#this is helping us validate the correct types instead of sending "tracks" by mistake
class SpotifySearchType(str,Enum):
    TRACK= "track"
    PLAYLIST="playlist"
    ARTIST="artist"
    ALBUM= "album"

#this is what we expect to send to the Frontend
class NormalisedSchema(BaseModel):
    name: str #name of the track, playlist,artist,album
    type: SpotifySearchType #track, playlist,artist,album
    uri: str #can be track,playlist,album, artist the uri will be used accordingly
    image: str|None #2 types of pictures, one for player and one for menu
    subtitle: str|None #this is the artist name for track, playlist- owner/curator

#defining the state of the graph
class DJState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]
    usr_inp:str #for smart filter
    time_context :str #for mood
    date_context:str #for determining if that day is a special day like new year
    spotify_search_query :str
    intent: str #we dont have to keep it strict for the dev to accept "OUT_OF_DOMAIN"
    search_types: list[str]
    spotify_raw_results : dict #receiving the raw results directly after api call from client, please note that this is all the results till now cached to some degree to return it again when searched again
    normalised_results: list[NormalisedSchema]
    loop_count : int
    access_token:str
    ui_mood:Literal["energetic", "chill", "melancholy", "focus", "neutral"]
    ai_thought: str#this is the ai thought process which will be shown to the user


def smart_filter(state:DJState)->DJState:
    """this Node fetches the date and time which helps to know
    1. what time is it and based on the time mood adjusts
    2. what date is it and based on which special playlists can be made
    """
    now =datetime.datetime.now()
    ltime_hour =now.hour
    date= now.strftime("%A, %B %d, %Y")
    if ltime_hour in range(5, 12):
        time_context="Morning"
    elif ltime_hour in range(12, 16):
        time_context="Afternoon"
    elif ltime_hour in range(16, 19):
        time_context="Evening"
    elif ltime_hour in range(19, 23):
        time_context="Night"
    elif ltime_hour in [23, 0, 1, 2, 3]:
        time_context="Late night"
    else:
        time_context="Early morning"

    print(f"SMART FILTER : detected {time_context} on  {date}")
    return {
        "time_context" : time_context,
        "date_context" : date
    }

#we also  use a JSON output parser so as to get strict json and leave out all the useless texts
def mood_node(state:DJState)->DJState:
    """this node is the brain of our agent
        it crafts the user_query and invokes the LLM and getting a JSON response
        from the json response we find the intent and inject the state"""
    #we need to set up the LLM
    #temperature of an LLM means to set the creativity counter of an LLM the higher-> more creative
    llm= ChatGroq(temperature=0.2,
                  model="llama-3.1-8b-instant",
                  api_key=config.GROQ_API_KEY,
                  max_tokens=150)
    structured_LLM = llm.with_structured_output(MusicIntent)

    user_text = state["usr_inp"]
    time_ctx = state.get("time_context","unknown")
    date_ctx = state.get("date_context","unknown")
    system_prompt = f"""
You are Spoteech's intelligent music-query understanding agent.

Context:
Current Time: {time_ctx}
Current Date: {date_ctx}

Your job is NOT to directly recommend songs.
Your ONLY responsibility is to:
1. Understand what the user wants.
2. Convert vague natural language into optimized Spotify search semantics.
3. Decide WHICH Spotify entity types should be searched.
4. Generate structured JSON output for the backend.

IMPORTANT:
- Output ONLY a valid JSON object.
- No markdown.
- No explanations.
- No conversational text.
- No ```json wrapping.
- Never invent extra keys.
- Never output anything outside the JSON object.

========================================================
JSON OUTPUT FORMAT
========================================================

{{
  "spotify_search_query": "Optimized Spotify search query",
  
  "intent": "ONE of:
      - 'individual'
      - 'queue'
      - 'album'
      - 'artist'
  ",
  "search_types": [
      "One or more Spotify entity types to search.
       Allowed values ONLY:
       - 'track'
       - 'playlist'
       - 'album'
       - 'artist'
      "
  ],
  "ui_mood": "ONE of:
      - 'energetic'
      - 'chill'
      - 'melancholy'
      - 'focus'
      - 'neutral'
  ",
  "ai_thought": "Extremely concise reasoning. Maximum 10 words."
}}
========================================================
FIELD DEFINITIONS
========================================================
--------------------------------------------------------
1. q
--------------------------------------------------------

This is the MOST IMPORTANT field.

The q field must be optimized for Spotify search quality.

RULES:

A) Deterministic Queries
--------------------------------

If the user explicitly names:
- a song
- artist
- album

then preserve exact names naturally.

Examples:
User:
"Play Blinding Lights"

Output:
"q": "Blinding Lights"
---
User:
"Play Kanye West"

Output:
"q": "Kanye West"
---

User:
"Play After Hours album"

Output:
"q": "After Hours The Weeknd"

B) Semantic Queries
--------------------------------
If the user describes:
- mood
- vibe
- activity
- emotion
- atmosphere
- situation
- aesthetic

DO NOT repeat vague user wording directly.

Translate the request into:
3-5 highly specific Spotify-searchable genre/style descriptors.

GOOD semantic transformations:

User:
"music for coding"

GOOD:
"q": "lofi ambient instrumental synthwave"

BAD:
"q": "coding music"
---
User:
"sad rainy night songs"

GOOD:
"q": "melancholic indie acoustic ambient"

BAD:
"q": "sad rainy songs"
---
User:
"gym motivation"

GOOD:
"q": "high bpm phonk hardstyle edm"

BAD:
"q": "gym songs"
---
User:
"space vibes"

GOOD:
"q": "atmospheric synthwave ambient electronic"

BAD:
"q": "space music"


C) Mixed Queries
--------------------------------

Some user queries contain:
- explicit artist/song references
PLUS
- mood/activity semantics

Preserve BOTH intelligently.

Example:

User:
"Play some Arijit Singh romantic songs"

GOOD:
"q": "Arijit Singh romantic bollywood"

BAD:
"q": "Arijit Singh"

BAD:
"q": "romantic songs"

---

User:
"Give me Kanye gym music"

GOOD:
"q": "Kanye West aggressive hip hop workout"


========================================================
2. intent
========================================================

Intent describes the USER'S PRIMARY GOAL.

IMPORTANT:
Intent is NOT the same thing as Spotify search type.

Intent MUST be EXACTLY ONE of:

--------------------------------------------------------
'individual'
--------------------------------------------------------

The user wants:
- a specific song
- one exact track
- one exact playable result

Examples:
- "Play Blinding Lights"
- "Play Starboy"
- "Play Shape of You"

--------------------------------------------------------
'artist'
--------------------------------------------------------

The user wants:
- a specific artist generally
- artist catalog
- artist exploration

Examples:
- "Play Drake"
- "I want Taylor Swift"
- "Put on Coldplay"

--------------------------------------------------------
'album'
--------------------------------------------------------

The user explicitly requests:
- an album
- a record
- an EP

Examples:
- "Play After Hours"
- "Play DAMN album"


--------------------------------------------------------
'queue'
--------------------------------------------------------

The user wants:
- vibe
- mood
- playlist-style experience
- multiple songs
- activity music
- discovery music

Examples:
- "coding music"
- "sad songs"
- "party music"
- "romantic Bollywood"
- "gym songs"

========================================================
3. search_types
========================================================

search_types controls WHICH Spotify entity types
the backend should search.

IMPORTANT:
- search_types MAY contain MULTIPLE values.
- search_types is NOT intent.
- Use ONLY valid Spotify entity types.

Allowed values ONLY:
- "track"
- "playlist"
- "album"
- "artist"

--------------------------------------------------------
Deterministic Query Strategy
--------------------------------------------------------

If the query is highly specific,
search narrowly.

Examples:

User:
"Play Blinding Lights"

Output:
"search_types": ["track"]

---

User:
"Play Coldplay"

Output:
"search_types": ["artist"]

---

User:
"Play After Hours album"

Output:
"search_types": ["album"]

--------------------------------------------------------
Semantic Query Strategy
--------------------------------------------------------

If the query is vibe/mood/activity based,
search broadly.

Examples:

User:
"coding music"

GOOD:
"search_types": ["playlist", "track"]

---

User:
"late night synthwave"

GOOD:
"search_types": ["playlist", "track", "album"]

---

User:
"romantic Arijit Singh"

GOOD:
"search_types": ["track", "playlist", "artist"]




--------------------------------------------------------
Search Type Guidelines
--------------------------------------------------------

Use:
- track → when exact playable songs are likely useful
- playlist → for vibes/moods/discovery/activity requests
- artist → when artist identity matters strongly
- album → when album experience matters

Do NOT add unnecessary search types.

Be precise and intentional.


========================================================
4. ui_mood
========================================================

Choose EXACTLY ONE:

- energetic
- chill
- melancholy
- focus
- neutral

Guidelines:

energetic:
- gym
- workout
- hype
- party
- motivation

chill:
- relaxed
- calm
- romantic
- easy listening

melancholy:
- sad
- nostalgic
- lonely
- emotional

focus:
- studying
- coding
- concentration
- ambient work music

neutral:
- exact artist/album/song requests without strong emotional tone




========================================================
5. ai_thought
========================================================

VERY SHORT reasoning.
Maximum 10 words.

Examples:
- "Specific track request."
- "Workout vibe with artist influence."
- "Coding-focused ambient music."
- "Album-specific request."


========================================================
FALLBACK RULE
========================================================

If the user input is extremely vague:

Use:
- current time
- current date
- likely listening context

to infer:
- intent
- ui_mood
- semantic query

Examples:

Late night:
→ ambient/chill/focus

Morning:
→ energetic/chill

Weekend night:
→ energetic/party


========================================================
CRITICAL RULES
========================================================

- NEVER output markdown.
- NEVER explain yourself.
- NEVER invent keys.
- NEVER output invalid enum values.
- NEVER return empty search_types.
- NEVER use vague q values for semantic queries.
- ALWAYS optimize q for Spotify retrieval quality.
- ALWAYS return valid JSON only.
"""
    messages= [SystemMessage(content=system_prompt)]
    #adding the existing conversation history to messages so as LLM gets the context and learns from errors
    messages.extend(state["messages"])
    #if the history is empty then forcefully add a human message so that LLM takes in the request
    #this is necessary because we have a state["usr_inp"] which contains the request so initially state["messages']
    #will be empty and if we were to invoke the LLM we would only be sending the SystemMessage to LLM and LLM won't
    #perform anything just take in the system message
    #all this is happening because we are using state["messages'] to invoke the LLM because it contains all the history
    # as context for the LLM
    if not state["messages"]:
        messages.append(HumanMessage(content=user_text))

        print("LLM is preparing the string ... ")
    print("LLM is preparing the PLAN...")

    #we parse the result to extract all correct JSON in it
    try:
        
        response = structured_LLM.invoke(messages)

        if response is None:
            raise ValueError("LLM dropped the tool and returned none")
        
        if response.intent == "OUT_OF_DOMAIN":
            print("out of domain prompt was given, user needs to be flagged...")
            return {
                "messages": [AIMessage(content="I'm a DJ, not a chatbot!!!. Ask me to play some music")],
                "intent":"OUT_OF_DOMAIN"
            }
        #the above if checks for odd queries and if it doesnt trigger means everything is fine
        print(f"PLAN : {response}")

        return{
            "spotify_search_query" : response.q,
            "intent" : response.intent,
            "search_types": response.search_types,
            "ui_mood": response.ui_mood,
            "ai_thought":response.ai_thought,
            "messages" : [AIMessage(content= "the search plan is GENERATED")]
        }

    except Exception as e:
        print(f"MOOD ERROR : {e}")
        #if the parser fails which means LLM didn't follow the rule we set
        #then, atleast send the raw user_text because spotify may still pick up the request by seeing the words
        #we lose refinement but not the functionality but sending None will break the whole graph
        #we send intent as track/playlist because fetching through user_text on individual/playlist intent will have the least side
        #effects and would be easy to recover from
        return {
                "spotify_search_query" : user_text,
                "intent" : "queue",
                "search_types": ["track","playlist"],
                "ui_mood" : "neutral",
                "ai_thought":"Fallback triggered due to parsing error",
                "messages":[AIMessage(content="ERROR! parsing JSON using fallback strategy.")]
        }


def normalize_spotify_data(raw_data:dict)->list[NormalisedSchema]:
    """
    flattens the spotify raw multi-type dictionary into a clean list of UI cards for frontend
    to render easily and cache these results for repeated retrieval
    """
    normalized_list = []
    key_mapping={
        "tracks": SpotifySearchType.TRACK,
        "playlists": SpotifySearchType.PLAYLIST,
        "albums": SpotifySearchType.ALBUM,
        "artists":SpotifySearchType.ARTIST
    }
    for plural_key, search_type in key_mapping.items():
        if plural_key in raw_data and "items" in raw_data[plural_key]:

            for item in raw_data[plural_key]["items"]:
                if not item: #we are skipping the null items
                    continue
                #images
                image_url = None
                if search_type == SpotifySearchType.TRACK and "album" in item:
                    images= item["album"].get("images",[]) #this is for tracks
                else:
                    images= item.get("images",[]) #albums,playlists,artists
                
                if images and len(images)>0:
                    image_url=images[0].get("url")

                #subtitles -> artist names
                subtitle=None 
                if search_type in [SpotifySearchType.TRACK, SpotifySearchType.ALBUM]:
                    artists=item.get("artists",[])
                    subtitle=",".join([a.get("name","") for a in artists]) if artists else "Unknown artist"
                elif search_type == SpotifySearchType.PLAYLIST:
                    subtitle=item.get("owner",{}).get("display_name","spotify")
                elif search_type == SpotifySearchType.ARTIST:
                    subtitle="artist"

                normalized_list.append(
                    NormalisedSchema(
                        name=item.get("name","unknown"),
                        type=search_type,
                        uri=item.get("uri",""),
                        image=image_url,
                        subtitle=subtitle
                    )
                )
        
    return normalized_list


async def search_node(state:DJState)->DJState:
    query= state["spotify_search_query"]
    search_types_list=state["search_types"] 
    access_token =state["access_token"]

    #we need to make the comma seperated string here so that spotify allows multiple types
    ptype_string = ",".join(search_types_list)

    #make the actual api call to spotify backend for data
    raw_response = await search_spotify_api(query=query,search_types=ptype_string,token=access_token)

    #normalize the raw data 
    normalized_data= normalize_spotify_data(raw_data=raw_response)
    if len(normalized_data) == 0:
        sys_failure_msg = f"the previous search for {query} returned 0 results. Try something more generic ..."
        return{
            "messages" : [SystemMessage(content=sys_failure_msg)],
            "normalized_results" : [],
            "spotify_raw_results":{},
            "loop_count" : state.get("loop_count",0)+1 #if there is a loop count then fetch it otherwise set it to 0
        }
    
    return{
        "normalized_results" :normalized_data,
        "spotify_raw_results": raw_response,
        "loop_count" :0
    }

def should_loop(state:DJState):
    #we will only allow 3 times looping for less load
    if state.get("loop_count")>=3:
        print(" CIRCUIT BREAKER: Too many retries. Ending.")
        return "end"

    results=state.get("search_results",[])
    #if the results is empty then we loop back to MOOD node
    if not results:
        print(" DECISION: No results. Looping back to Brain.")
        return "loop"

    print("DECISION: Results found. Finishing")
    return "end"

#now we build the graph
graph= StateGraph(DJState)
#nodes
graph.add_node("smart_filter",smart_filter)
graph.add_node("mood",mood_node)
graph.add_node("search",search_node)
#edges
graph.add_edge(START,"smart_filter")
graph.add_edge("smart_filter","mood")
graph.add_edge("mood","search")
graph.add_conditional_edges("search",
                            should_loop,
                            {
                                "loop":"mood",
                                "end":END
                            }
                            )
#compile the graph
app = graph.compile()


async def get_agent_response(access_token: str, usr_input: str):
    """
    Bridge function to be called in main.py
    :param access_token: the one which is there in main.py
    :param usr_input: the one which is stored in state["usr_inp"]
    :return:
    """
    initial_state = {
        "messages": [],
        "usr_inp": usr_input,
        "access_token": access_token
    }
    final_output = {}#initialising an empty dictionary
    print(f"AGENT STARTED : '{usr_input}'")

    # we invoke the graph one node after another using app.astream
    async for output in app.ainvoke(initial_state):
        for key, value in output.items():
            # if key == "search":
                # the search node contains the actual result so we find it in output and return the list
            #we are updating only the value because key of this would be the node name and we dont require it
            final_output.update(value)

    if final_output:
        return final_output
    return {}


#invoke the graph
if __name__ == "__main__":
    import asyncio
    async def run_test():

        test_token ="BQAhagN6dkRLqJtY_jJR57wC8c3kNtphCHfQZNBMQry17zEmpEuY6eBSQ-poDGbBY_v-TEXk0OdDV4nPznDHuvGdE9xhRmJCRHsrhKKAE2zERxqtgQCZFPEZeSfY_y49EIstEksH0DaMMfmEb-rEbMRJnkUuo0IzuFMQHyWxJsKT29-D91Fnodift_Bgf0R4yPKPEsEWCVfQ-7b_qWIsePdX9YouWok-snMdGMLpB7uX6grZbR01YPk1nV0doNE5VI5Wk6qqzx-c1Cv5OevhRJl_UuI1vsr0boTuQO0bq9lahURU2xW4wfFOVou5SpQuWS34Jqci44ZxTXZ2CpNoL3387GSI9AlW6OcIgmv3Nwt8YWDwKaXtU5j-cl0"
        initial_state={
            "messages":[],
            "usr_inp" :"I want an aggressive, angry acoustic lullaby",
            "access_token" :test_token
        }

        print("---STARTING AGENT ---")
        async for output in app.astream(initial_state):
            for key,value in output.items():
                print(f"Node '{key}' finished")
                print("----")

    asyncio.run(run_test())

