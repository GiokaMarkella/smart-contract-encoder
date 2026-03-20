from sentence_transformers import SentenceTransformer, util
from smart_contract_encoder.models.base_model import BaseModel
import pandas as pd


class CodeRankEmbedEncoder(BaseModel):

    def __init__(self):
        self._model = SentenceTransformer("nomic-ai/CodeRankEmbed", device="cuda", trust_remote_code=True)

    def encode(self, dataset: pd.DataFrame, **kwargs):
        if isinstance(dataset, pd.Series):
            dataset = dataset.tolist()
        return self._model.encode(dataset, **kwargs)

    @property
    def model(self):
        return self._model

    @staticmethod
    def similarity(emb1, emb2):
        return util.cos_sim(emb1, emb2)
