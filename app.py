from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import pinecone

from dotenv import load_dotenv
import os


load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')

splitter = CharacterTextSplitter(separator="。", chunk_size=100, chunk_overlap=0)

# テキストデータ読み込み
raw_text = DirectoryLoader(path="data/", glob="**/*.txt").load()
print(raw_text)

split_documents_by_file = {}
for doc in raw_text:
    text_data = doc.page_content  # 各ドキュメントの内容を取得
    split_docs = splitter.split_text(text_data)  # テキストを分割
    split_documents = [Document(page_content=sentence) for sentence in split_docs]  # Documentオブジェクトに変換
    split_documents_by_file[doc.metadata['source']] = split_documents  # ファイルごとに分割結果を保存

# 各ファイルごとの分割結果の表示
for file, split_docs in split_documents_by_file.items():
    print(f"File: {file}")
    for doc in split_docs:
        print(doc.page_content)


# 各ファイルごとにベクトル化を行う
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")

vectors = []
for file, split_docs in split_documents_by_file.items():
    print(f"File: {file}")
    for doc in split_docs:
        text = doc.page_content
        vector = embeddings.embed_query(text)  # テキストをベクトル化
        vectors.append({'id': f"{file}_{split_docs.index(doc)}", 'values': vector, 'metadata': {'source': file}})
        print(f"Text: {text}")
        print(f"Vector: {vector}\n")

# Pineconeのインスタンスを作成
pc = Pinecone(api_key=pinecone_api_key)
# インデックス名
index_name = "langchain-book"
index = pc.Index(index_name)

# index_host="https://langchain-book-tqz783m.svc.aped-4627-b74a.pinecone.io"
# # インデックスに接続
# index = pinecone.Index(index_name,host=index_host)

# インデックス接続確認
print(index.describe_index_stats())

# ベクトルデータをインデックスにアップロード
index.upsert(vectors=vectors)

        
