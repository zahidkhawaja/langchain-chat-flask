"""Ingest examples into Weaviate."""
import os
from pathlib import Path

import weaviate

WEAVIATE_URL = "https://hwc-testing.semi.network"
client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
)

client.schema.delete_class("Rephrase")
client.schema.delete_class("QA")
client.schema.get()
schema = {
    "classes": [
        {
            "class": "Rephrase",
            "description": "Rephrase Examples",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text",
                }
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "question",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "answer",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "chat_history",
                },
            ],
        },
    ]
}

client.schema.create(schema)

documents = [
    {
        "question": "how do i load those?",
        "chat_history": "Human: What types of memory exist?\nAssistant: \n\nThere are a few different types of memory: Buffer, Summary, and Conversational Memory.",
        "answer": "How do I load Buffer, Summary, and Conversational Memory"
    },
    {
        "question": "how do i install this package?",
        "chat_history": "",
        "answer": "How do I install langchain?"
    }
]
from langchain.prompts.example_selector.semantic_similarity import sorted_values
for d in documents:
    d["content"] = " ".join(sorted_values(d))
with client.batch as batch:
    for text in documents:
        batch.add_data_object(
            text,
            "Rephrase",
        )

client.schema.get()
schema = {
    "classes": [
        {
            "class": "QA",
            "description": "Rephrase Examples",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text",
                }
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "question",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "answer",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "summaries",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "sources",
                },
            ],
        },
    ]
}

client.schema.create(schema)

documents = [
        {
            "question": "how do i install langchain?",
            "answer": "```pip install langchain```",
            "summaries": ">Example:\nContent:\n---------\nYou can pip install langchain package by running 'pip install langchain'\n----------\nSource: foo.html",
            "sources": "foo.html"

        },
        {
            "question": "how do i import an openai LLM?",
            "answer": "```from langchain.llm import OpenAI```",
            "summaries": ">Example:\nContent:\n---------\nyou can import the open ai wrapper (OpenAI) from the langchain.llm module\n----------\nSource: bar.html",
            "sources": "bar.html"
        }
    ]
from langchain.prompts.example_selector.semantic_similarity import sorted_values
for d in documents:
    d["content"] = " ".join(sorted_values(d))
with client.batch as batch:
    for text in documents:
        batch.add_data_object(
            text,
            "QA",
        )
