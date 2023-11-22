import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=1024,
        dropout=0.2,
    ):
        super().__init__()

        self.projection_dim = projection_dim
        if self.projection_dim > 1024:
            self.projection1 = nn.Linear(embedding_dim, int(projection_dim/2))
            self.gelu1 = nn.GELU()
            self.projection2 = nn.Linear(int(projection_dim/2), projection_dim)
            self.gelu2 = nn.GELU()

            self.fc = nn.Linear(projection_dim, projection_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(projection_dim)

        else: 
            self.projection = nn.Linear(embedding_dim, projection_dim)
            self.gelu = nn.GELU()
            self.fc = nn.Linear(projection_dim, projection_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x, sensor = "SPOT"):
        if self.projection_dim > 1024:
            projected = self.projection1(x)
            projected = self.gelu1(projected)
            projected = self.projection2(projected)
            x = self.gelu2(projected)

        else:
            projected = self.projection(x)
            x = self.gelu(projected)

        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
        
        
class MSADModel(nn.Module):
    def __init__(
        self,
        temperature=1,
        sat1_embedding=1024,
        sat2_embedding=1024,
        label_smoothing=0
    ):
        super().__init__()
        self.sat1_projection = ProjectionHead(embedding_dim=sat1_embedding)
        self.sat2_projection = ProjectionHead(embedding_dim=sat2_embedding)
        self.temperature = temperature

        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing = label_smoothing, reduction='none')

    def forward(self, sat1_features, sat2_features):
        # Getting Embeddings (with same dimension)
        sat1_embeddings = self.sat1_projection(sat1_features)
        sat2_embeddings = self.sat2_projection(sat2_features)

        # Calculating the Loss
        logits = (sat2_embeddings @ sat1_embeddings.T) / self.temperature
        sat1_similarity = sat1_embeddings @ sat1_embeddings.T
        sat2_similarity = sat2_embeddings @ sat2_embeddings.T
        targets = F.softmax(
            (sat1_similarity + sat2_similarity) / 2 * self.temperature, dim=-1
        )

        sat2_loss = self.cross_entropy(logits, targets)
        sat1_loss = self.cross_entropy(logits.T, targets.T)        
        loss =  (sat1_loss + sat2_loss) / 2.0 # shape: (batch_size)
        return loss.mean()