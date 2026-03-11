import torch
import torch.nn as nn
from ml.training.src.models.decoders.complex import ComplEx  # Import the unified decoder

class TextComplEx(nn.Module):
    def __init__(self, config, num_entities, num_relations, bert_emb_path):
        super(TextComplEx, self).__init__()
        # Get the training dimension d (must be even）
        self.d = config['model']['embedding_dim']
        
        # 1. Load the BERT cache (768 dimensions)
        print(f"Member A: Loading BERT embeddings from {bert_emb_path}")
        bert_emb = torch.load(bert_emb_path).float()
        self.register_buffer("bert_base", bert_emb)
        
        # 2. Mapping layer: map the 768 dimensions of BERT to the real part (re) and the imaginary part (im) of the complex space
        # The framework requires that the total dimension after splicing is d, so each part is mapped to d // 2
        self.re_map = nn.Linear(768, self.d // 2)
        self.im_map = nn.Linear(768, self.d // 2)
        
        # 3. Core: Instantiate the unified ComplEx decoder of the team leader
        self.decoder = ComplEx(num_relations, self.d)

    def get_entity_emb(self, entity_ids):
            """
            Map the BERT feature and splice it into [B, d]
            """
            base = self.bert_base[entity_ids]
            h_re = self.re_map(base)
            h_im = self.im_map(base)
            # Stitching the real part and the virtual part
            return torch.cat([h_re, h_im], dim=-1)

    def score(self, triples):
        # triples: [h, r, t]
        """
            Call the unified scoring logic
            triples: [batch_size, 3] -> (h, r, t)
        """
        h_emb = self.get_entity_emb(triples[:, 0])
        r_id  = triples[:, 1]
        t_emb = self.get_entity_emb(triples[:, 2])

        # Directly call the Re(<h, r, conj(t)> implemented
        return self.decoder.score(h_emb, r_id, t_emb)

    def forward(self, pos_triples, neg_triples):
        """
            Calculate Loss according to the interface required by the framework
        """
        pos_scores = self.score(pos_triples)
        neg_scores = self.score(neg_triples)

        batch_size = pos_scores.size(0)
        neg_scores = neg_scores.view(batch_size, -1)

        # Adopt LogSigmoid loss function (general standard for KGE tasks)
        # Logic ：-log(sigmoid(pos - neg))
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-10).mean()

        return loss
