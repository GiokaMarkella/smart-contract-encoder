from smart_contract_encoder.models.sentence_encoder import SentenceEncoder
from smart_contract_encoder.models.ngram_model import NgramEncoder
from smart_contract_encoder.models.smartembed_model import SmartEmbed
from smart_contract_encoder.models.codebert_model import CodeBERTEncoder
from smart_contract_encoder.models.coderankeembed_model import CodeRankEmbedEncoder
from smart_contract_encoder.load_data import *
from smart_contract_encoder.utils import ENC_VALS, ENC_VERSION

def load_encoder(encoder, encoder_version, model_to_load=None):
    load = encoder_version == "finetuned"
    if encoder == "sentence_encoder":
        enc = SentenceEncoder(load=load, model_to_load=model_to_load)
    elif encoder == "ngram":
        enc = NgramEncoder(load=load)
    elif encoder == "smartembed":
        enc = SmartEmbed()
    elif encoder == "codebert":
        enc = CodeBERTEncoder()
    elif encoder == "coderankeembed":
        enc = CodeRankEmbedEncoder(load=load, model_to_load=model_to_load)
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")
    return enc

def _encode_field(field, df, enc):
    if field == "func_documentation_func_code":
        cols = df['func_documentation'] + "\n" + df['func_code']
    else:
        cols = df[field]
    embeddings = enc.encode(cols)
    df['embeddings'] = [e for e in embeddings]
    df = df[['embeddings']]
    return df

def create_embeddings(split, encoder, encoder_version, field, model_to_load=None):
    df = load_dataset(file_type="merged", split=split)
    enc = load_encoder(encoder, encoder_version, model_to_load=model_to_load)
    print(f"Creating embeddings for the {split} split, using the {encoder_version} {encoder} encoder, on the {field} field")
    df_emb = _encode_field(field, df, enc)
    print(df_emb)
    save_dataset(df_emb, file_type="embeddings", split=split, encoder=encoder, encoder_version=encoder_version, field=field)
    return df_emb
