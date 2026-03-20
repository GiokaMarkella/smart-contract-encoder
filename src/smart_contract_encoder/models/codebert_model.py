import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from smart_contract_encoder.models.base_model import BaseModel
from smart_contract_encoder.models.sentence_encoder import CardData


class CodeBERTEncoder(BaseModel):

    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self._model = AutoModel.from_pretrained("microsoft/codebert-base")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self.similarity_fn_name = 'cosine'
        self.model_card_data = CardData()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

    def encode(self, texts, batch_size=32, **kwargs):
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                output = self._model(**encoded)
            embeddings = self._mean_pooling(output, encoded["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def encode_queries(self, queries, **kwargs):
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            texts = [d.get("text", "") for d in corpus]
        else:
            texts = corpus
        return self.encode(texts, **kwargs)

    @property
    def model(self):
        return self

    @staticmethod
    def similarity(emb1, emb2):
        from sentence_transformers import util
        return util.cos_sim(emb1, emb2)
