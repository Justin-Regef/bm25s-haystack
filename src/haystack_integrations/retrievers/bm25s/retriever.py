# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from haystack import component, Document
from haystack_integrations.document_stores.bm25s import BM25S_DocumentStore
import bm25s


@component
class BM25S_Retriever:
    """
    A component for retrieving documents from an ExampleDocumentStore.
    """

    def __init__(self, document_store: BM25S_DocumentStore, top_k: int = 10):
        """
        Create an ExampleRetriever component. Usually you pass some basic configuration
        parameters to the constructor.

        :param document_store: A Document Store object used to retrieve documents
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).

        :raises ValueError: If the specified top_k is not > 0.
        """
        self.document_store = document_store
        self.top_k = top_k
        

    def run(self, query: str):
        """
        Run the Retriever on the given input data.

        :param data: The input data for the retriever. In this case, a list of queries.
        :return: The retrieved documents.
        """
        query_tokens = bm25s.tokenize(query, stemmer=self.document_store.stemmer)
        bm25s_retrieved_docs = self.document_store.bm25s.retrieve(query_tokens, k=self.top_k, return_as='documents')
        if self.document_store.load_corpus:
            return [Document(content=doc['text']) for doc in bm25s_retrieved_docs[0]]
        return [Document(id=doc) for doc in bm25s_retrieved_docs[0]]
