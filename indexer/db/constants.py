# TODO: most variable here should be set in .env file
MILVUS_HOST = "186.233.187.130"
MILVUS_PORT = "19530"
USER = "admin"
PASSWORD = "admin"
URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
COLLECTION_NAME = "targon_prod"
INDEX_PARAM = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
DB_COLS = {
    "URL": "url",
    "FULL_TEXT": "text",
    "SUMMARY": "summary",
    "EMBED": "embeddings",
}
