import os

from serpapi import GoogleSearch
from pydantic import BaseModel

class QueryParams(BaseModel):
    q: str
    location: str = "US"
    hl: str = "Google UI Language",
    gl: str = "Google Country",
    google_domain: str = "google.com",
    api_key: str


        
def query(params):
    assert params['api_key'], "SERP_API_KEY not found in .env"


    search = GoogleSearch(params)
    results = search.get_json()

    # get organic_results from results
    organic_results = results['organic_results']
    
    #create response object
    response_object = [{"description": result['snippet'], "url": result['link']} for result in organic_results]

    return response_object