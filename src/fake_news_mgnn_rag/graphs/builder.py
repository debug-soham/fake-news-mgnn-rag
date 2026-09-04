import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch_geometric.data import HeteroData

from fake_news_mgnn_rag.encoders.text_encoder import TextEncoder
from fake_news_mgnn_rag.encoders.vision_encoder import VisionEncoder
from fake_news_mgnn_rag.rag.retriever import DynamicRAGRetriever

class GraphBuilder(nn.Module):
    """
    Constructs a PyTorch Geometric HeteroData graph from a batch of multimodal samples.
    """
    def __init__(
        self,
        text_encoder: TextEncoder,
        vision_encoder: VisionEncoder,
        retriever: DynamicRAGRetriever,
        top_k_rag: int = 3,
        device: str = "cpu"
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.retriever = retriever
        self.top_k_rag = top_k_rag
        self.device = device

    def forward(self, batch: Dict[str, Any]) -> HeteroData:
        """
        Forward pass to construct a single batched HeteroData object from a dataloader batch.
        
        Args:
            batch: Dictionary containing 'text' (List[str] or tuple), 'image' (Tensor), 'dataset_origin' (List[str] or tuple), 'label' (Tensor)
        Returns:
            HeteroData representing the entire batch as a disjoint graph collection.
        """
        # Ensure texts and origins are lists (DataLoader might yield tuples)
        texts = list(batch["text"])
        images = batch["image"].to(self.device)
        origins = list(batch["dataset_origin"])
        labels = batch["label"].to(self.device)
        
        batch_size = len(texts)
        
        # 1. Encode Texts
        text_embeddings = self.text_encoder(texts)
        
        # 2. Encode Images
        image_embeddings = self.vision_encoder(images)
        
        # 3. Encode User/Metadata (Mocked via dataset_origin as string metadata)
        # To make it distinct, prefix with "origin: "
        user_texts = [f"source origin: {orig}" for orig in origins]
        user_embeddings = self.text_encoder(user_texts)
        
        # 4. Retrieve & Encode RAG Facts
        rag_batches = self.retriever.retrieve_batch(texts, top_k=self.top_k_rag)
        # Flatten RAG facts to encode in one batch
        flat_rag_texts = []
        for claim_facts in rag_batches:
            for fact_node in claim_facts:
                flat_rag_texts.append(fact_node["content"])
        
        if len(flat_rag_texts) > 0:
            rag_embeddings = self.text_encoder(flat_rag_texts)
        else:
            # Fallback if no facts are found
            rag_embeddings = torch.zeros((0, self.text_encoder.hidden_size), device=self.device)
            
        # 5. Construct HeteroData
        data = HeteroData()
        
        # Assign Node Features
        data['text'].x = text_embeddings
        data['image'].x = image_embeddings
        data['user'].x = user_embeddings
        if len(flat_rag_texts) > 0:
            data['rag_fact'].x = rag_embeddings
        
        # Assign Ground Truth Labels to text nodes (the root of prediction)
        data['text'].y = labels
        
        # 6. Construct Edges
        # Edges within the batch are isolated per sample (disjoint graphs).
        
        # User <-> Text
        user_indices = torch.arange(batch_size, dtype=torch.long, device=self.device)
        text_indices = torch.arange(batch_size, dtype=torch.long, device=self.device)
        data['user', 'mentions', 'text'].edge_index = torch.stack([user_indices, text_indices], dim=0)
        data['text', 'mentioned_by', 'user'].edge_index = torch.stack([text_indices, user_indices], dim=0)
        
        # Text <-> Image
        image_indices = torch.arange(batch_size, dtype=torch.long, device=self.device)
        data['text', 'consistent_with', 'image'].edge_index = torch.stack([text_indices, image_indices], dim=0)
        data['image', 'consistent_with', 'text'].edge_index = torch.stack([image_indices, text_indices], dim=0)
        
        # Text <-> RAG Fact
        if len(flat_rag_texts) > 0:
            rag_src_indices = []
            rag_dst_indices = []
            current_rag_idx = 0
            for i, claim_facts in enumerate(rag_batches):
                for _ in claim_facts:
                    rag_src_indices.append(i)
                    rag_dst_indices.append(current_rag_idx)
                    current_rag_idx += 1
            
            rag_src_tensor = torch.tensor(rag_src_indices, dtype=torch.long, device=self.device)
            rag_dst_tensor = torch.tensor(rag_dst_indices, dtype=torch.long, device=self.device)
            
            data['text', 'grounded_in', 'rag_fact'].edge_index = torch.stack([rag_src_tensor, rag_dst_tensor], dim=0)
            data['rag_fact', 'supports', 'text'].edge_index = torch.stack([rag_dst_tensor, rag_src_tensor], dim=0)
            
        return data
