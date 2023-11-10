import ray
import requests
from bs4 import BeautifulSoup
from json import JSONEncoder, dumps
import numpy as np
from db import VectorDBClient
from llm import MODEL_REGISTRY
import requests
import html2text


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


@ray.remote
class WebCrawler:
    """Web crawler class as a Ray actor."""

    def __init__(
        self,
        llm_model: str = "bert-base-uncased",
        batch_size: int = 32,
    ):
        # Initialize language model
        self.llm_model = MODEL_REGISTRY[llm_model]()

        # Initialize Milvus connection
        # self.db_client = VectorDBClient(embed_size=self.llm_model.embed_size, batch_size=batch_size)


    def submit(self, url, text, summary, embeddings):
        params = {
            "url": url,
            "text": text,
            "summary": summary,
            "embeddings": dumps({"embeddings": embeddings}, cls=NumpyArrayEncoder),
        }

        headers = {
            'X-API-KEY': ""
        }

        response = requests.post("http://localhost:8000/v1/vector/submit", json=params, headers=headers)

        print(response)
        print(response.content)

    def crawl(self, url, depth, max_depth):
        if depth > max_depth:
            return

        # try:
            # Fetch the webpage
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            h = html2text.HTML2Text()
            h.ignore_links = True
            text = h.handle(response.text)

            # get summary

            # Generate BERT embeddings for the text
            embeddings = self.llm_model.text_to_embedding(text)

            # Insert data into Milvus
            self.submit(url, text, "summary here", embeddings)
            # self.db_client.insert(url, text, embeddings)
            new_links = []
            links = soup.find_all("a")
            for link in links:
                child_url = link.get("href")
                if child_url and child_url.startswith("http") or child_url and child_url.startswith("https"):
                    new_links.append(child_url)
            return new_links
        # except Exception as e:
        #     print(f"Error crawling {url}: {str(e)}")
