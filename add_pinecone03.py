import logging
import os
import sys
import glob

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_vectorstore():
    # Pinecone APIキーの取得
    api_key = os.environ["PINECONE_API_KEY"]
    index_name = "docs-quickstart-index"

    # Pineconeインスタンスの作成
    pinecone_instance = Pinecone(api_key=api_key)

    # インデックスが存在しない場合は作成
    if index_name not in pinecone_instance.list_indexes().names():
        pinecone_instance.create_index(
            name=index_name,
            dimension=1536,  # 適切な次元数に設定してください
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # インデックスの利用準備
    index = pinecone_instance.Index(name=index_name)
    return index

if __name__ == "__main__":
    file_pattern = 'data/*.txt'
    file_paths = glob.glob(file_pattern)

    vectorstore = initialize_vectorstore()

    for file_path in file_paths:
        loader = TextLoader(file_path, encoding='utf-8')
        raw_docs = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        docs = text_splitter.split_documents(raw_docs)
        logger.info("split %d documents", len(docs))

        vectorstore.upsert(vectors=[(str(i), doc, 1.0) for i, doc in enumerate(docs)])
