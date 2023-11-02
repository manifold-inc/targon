import ray
import requests

from typing import Optional
from bs4 import BeautifulSoup
from pydantic import BaseModel
from targon.db import VectorDBClient
from targon.encoder import ENCODER_REGISTRY

class CrawlJob(BaseModel):
    url: list[str]
    max_depth: Optional[int] = 2

@ray.remote
class WebCrawler:
    """Web crawler class as a Ray actor."""

    def __init__(
        self,
        encoder: str = "bert-base-uncased",
        batch_size: int = 32,
    ):
        # Initialize language model
        self.encoder = ENCODER_REGISTRY[encoder]()

        # Initialize Milvus connection
        self.db_client = VectorDBClient(embed_size=self.encoder.embed_size, batch_size=batch_size)

    def crawl(self, url, depth, max_depth):
        if depth > max_depth:
            return

        try:
            # Fetch the webpage
            response = requests.get(url)
            if response.status_code == 200:
                # Parse the HTML content
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract text from the webpage
                text = soup.get_text()

                # Generate BERT embeddings for the text
                embeddings = self.encoder.text_to_embedding(text)

                # Insert data into Milvus
                self.db_client.insert(url, text, embeddings)

                # Find and crawl child links
                links = soup.find_all("a")
                for link in links:
                    child_url = link.get("href")
                    if child_url and child_url.startswith("http"):
                        self.crawl.remote(child_url, depth + 1, max_depth)
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")


def run_crawl(crawler: WebCrawler, urls: list[str], max_depth: int):
    ray.get([crawler.crawl.remote(url, 0, max_depth) for url in urls])
