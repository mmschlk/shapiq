"""This module defines the HuggingFaceClient class.

Provides functionality for:
- Interacting with HuggingFace datasets.
- Creates a local copy of the dataset for processing.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Self

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from .connection_exceptions import MissingHuggingFaceInfoError
from .local_client import LocalClient


class HuggingFaceClient(LocalClient):
    """Client for HuggingFace datasets to store shapiq experiment results.

    The HuggingFaceClient uses the HuggingFace Hub to access a dataset specified
    by the HF_DATASET environment variable. It expects the dataset to contain a
    file (e.g. "raw_runs.jsonl") with run records in JSON Lines format.

    The client downloads the dataset file into a local temporary directory and
    delegates all read access to LocalClient. This means the HuggingFaceClient
    does NOT modify the dataset on the HuggingFace Hub, but rather uses it as a
    read-only source of run records.

    Upon closing the client, the temporary local directory is deleted.
    """


    def __init__(self, dataset_name: str, local_dir: str, filename: str):
        self.dataset_name = dataset_name
        self._local_dir = local_dir
        super().__init__(path=filename)

    @classmethod
    def from_env(cls, args: dict) -> Self:
        load_dotenv()

        dataset_name = args.get("HF_DATASET") or os.getenv("HF_DATASET")
        if not dataset_name:
            raise MissingHuggingFaceInfoError("HF_DATASET")

        token = args.get("HF_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise MissingHuggingFaceInfoError("HF_TOKEN")
        
        filename = args.get("HF_FILE") or os.getenv("HF_FILE")        
        if not filename:
            raise MissingHuggingFaceInfoError("HF_FILE")

        local_dir = tempfile.mkdtemp(prefix="hf_dataset_")

        local_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=filename,
            token=token,
            local_dir=local_dir,
        )

        return cls(dataset_name=dataset_name, local_dir=local_dir, filename=local_path)

    def close(self) -> None:
        """Delete the temporary local copy of the dataset."""
        shutil.rmtree(self._local_dir, ignore_errors=True)