# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional

from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DuplicateDocumentError, MissingDocumentError
from haystack.document_stores.types import DuplicatePolicy
import bm25s
import Stemmer

logger = logging.getLogger(__name__)


class BM25S_DocumentStore:
    """
    Except for the __init__(), signatures of any other method in this class must not change.
    """

    def __init__(self, corpus_dirpath: str, load_corpus: bool=True, mmap: bool=True, stemmer_lang: str="english"):
        """
        Initializes the store. The __init__ constructor is not part of the Store Protocol
        and the signature can be customized to your needs. For example, parameters needed
        to set up a database client would be passed to this method.
        """
        self.corpus_dirpath = corpus_dirpath
        self.load_corpus = load_corpus
        self.mmap = mmap
        self.stemmer_lang = stemmer_lang
        self.bm25s = bm25s.BM25.load(corpus_dirpath, load_corpus=load_corpus, mmap=mmap)
        self.stemmer = Stemmer.Stemmer(stemmer_lang)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self.bm25s.scores["num_docs"]

    def filter_documents(self, _: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        Filters are defined as nested dictionaries that can be of two types:
        - Comparison
        - Logic

        Comparison dictionaries must contain the keys:

        - `field`
        - `operator`
        - `value`

        Logic dictionaries must contain the keys:

        - `operator`
        - `conditions`

        The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

        The `operator` value in Comparison dictionaries must be one of:

        - `==`
        - `!=`
        - `>`
        - `>=`
        - `<`
        - `<=`
        - `in`
        - `not in`

        The `operator` values in Logic dictionaries must be one of:

        - `NOT`
        - `OR`
        - `AND`


        A simple filter:
        ```python
        filters = {"field": "meta.type", "operator": "==", "value": "article"}
        ```

        A more complex filter:
        ```python
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": 1420066800},
                {"field": "meta.date", "operator": "<", "value": 1609455600},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        }

        :param filters: the filters to apply to the document list.
        :return: a list of Documents that match the given filters.
        """
        raise NotImplementedError # FIXME

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param policy: documents with the same ID count as duplicates. When duplicates are met,
            the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateDocumentError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: None
        """
        raise Exception('bm25s is immutable')

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param object_ids: the object_ids to delete
        """
        raise Exception('bm25s is immutable')

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary. You can customise here what goes into the
        final serialized format.
        """
        data = default_to_dict(
            self,
            corpus_dirpath=self.corpus_dirpath,
            load_corpus=self.load_corpus,
            mmap = self.mmap,
            stemmer_lang = self.stemmer_lang,
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25S_DocumentStore":
        """
        Deserializes the store from a dictionary, if you customised anything in `to_dict`,
        you can changed it back here.
        """
        return default_from_dict(cls, data)
