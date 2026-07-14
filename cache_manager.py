from upstash_redis.asyncio import Redis
from dotenv import load_dotenv
import os
load_dotenv()

class CacheManager:
    
    def __init__(self):
        #exists for connection of redis, qdrant, and load of fastembed model
        
        #initialize redis
        redis= Redis.from_url(os.getenv("REDIS_URL"),
                              decode_responses=True)
        
        #initialize qdrant
        
        

    def _hash_text(text):
        #private method for normalizing and hashing strings
        pass

    def _generate_vector(self,text):
        """
        private method : HIDDEN TO ABSTRACT THE EMBEDDING CREATION PROCESS
        """
        pass
    
    #PATHWAY B : the one where we match buttons and directly check redis
    def get_exact_match(self, key:str):
        """
        takes a pre-formatted string(like user:123:top_tracks), htis redis,
        and returns JSON or None
        """
        pass

    def save_exact_match(key:str,data:dict, ttl_timer : int):
        """
        saves the data into redis and adds a expiration timer to them 
        """
        pass

    def get_semantic_match(user_prompt:str):
        """
        first make the embedding of user_prompt and then check vector db
        also attach a unique id
        """
        pass

    def save_semantic_match(user_prompt:str, normalized_data:list):
        """
        handles the double write back to 2 different databases; it just 
        hands off the data and moves on
        this will run in the background, rather than right away
        we keep a delay so that user doesnt have to face any issue
        """