"""
LangChain Embeddings Integration for Snowflake Arctic Embed (v1.5)
Enables seamless usage with LangChain Retrievers, Chains, and VectorStores.
"""
from typing import List, Optional
import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import MODEL_NAME, DEVICE, MAX_SEQ_LENGTH, QUERY_INSTRUCTION

class ArcticLangChainEmbeddings(Embeddings):
    """
    LangChain compatible Embeddings class for Snowflake Arctic Embed (v1.5).
    
    Features:
    - Automatic asymmetric query prompt handling ("Represent this sentence...")
    - Native Matryoshka (MRL) dimension slicing (e.g. 768 -> 256 / 128)
    - GPU acceleration (FP16) on NVIDIA RTX 4050
    """
    def __init__(
        self,
        model_name_or_path: str = MODEL_NAME,
        device: str = DEVICE,
        truncate_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 64
    ):
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.model.max_seq_length = MAX_SEQ_LENGTH
        if device == "cuda":
            self.model.half()
        self.model.eval()
        
        self.truncate_dim = truncate_dim
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

    def _post_process(self, embeddings: np.ndarray) -> List[List[float]]:
        # Matryoshka dimension slicing if requested
        if self.truncate_dim is not None and self.truncate_dim < embeddings.shape[-1]:
            embeddings = embeddings[:, :self.truncate_dim]
            if self.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of catalog documents (book blurbs) using the document task.
        """
        with torch.no_grad():
            embs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")
        return self._post_process(embs)

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single search query, prepending the asymmetric instruction prompt.
        """
        prompted_query = QUERY_INSTRUCTION + text
        with torch.no_grad():
            emb = self.model.encode(
                [prompted_query],
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")
        return self._post_process(emb)[0]
