import requests
from numpy import ndarray
from targon import __api_endpoint__
from json import JSONEncoder, dumps
from targon.validator.config import env_config

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class VectorController:
    _instance = None
    _initialized_with = (
        None,  # store the api_key
    )

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorController, cls).__new__(cls)

            assert env_config.get("SYBIL_API_KEY", None) is not None, "SYBIL_API_KEY not set in environment variables. Please contact Manifold in the bittensor discord."
            api_key = env_config.get("SYBIL_API_KEY")

            cls._initialized_with = (api_key)
            cls._instance._initalize(api_key)

            
        return cls._instance
    
    def _initalize(self, api_key):
        self.api_key = api_key

    def __init__(self):
        pass

    def submit(self, url, title, text, query, embeddings):
        try:
            params = {
                "url": url,
                "title": title,
                "text": text,
                "query": query,
                "embeddings": dumps({"embeddings": embeddings}, cls=NumpyArrayEncoder),
            }

            headers = {
                'X-API-KEY': self.api_key,
            }

            response = requests.post(__api_endpoint__+"/v1/db/submit", json=params, headers=headers)

            if response.status_code != 200:
                print(response)
                print(response.content)

                return False
            
            return True
        
        except Exception as e:
            print(e)
            return False
