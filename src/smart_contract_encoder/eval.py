from smart_contract_encoder.data_curation import *
from smart_contract_encoder.encoder import *
from smart_contract_encoder.training_data_creation import *
from smart_contract_encoder.models.sentence_encoder import *
from smart_contract_encoder.eval_data_creation import *
from smart_contract_encoder.nicad_eval_data_creation import *
from smart_contract_encoder.graphs import *
import numpy as np
from sentence_transformers import util
import json

def main():
    # evaluate all different models
    baseline_enc = "sentence_encoder"
    baseline_enc_version = "untrained"
    baseline_field = "func_documentation"
    baseline_embeddings = create_embeddings("test", encoder=baseline_enc, encoder_version=baseline_enc_version, field=baseline_field)
    create_query_dataset("test", "sentence_encoder", "untrained", "code")
    create_query_dataset("test", "sentence_encoder", "untrained", "tac_code")
    create_query_dataset("test", "sentence_encoder", "untrained", "func_code")
    create_query_dataset("test", "sentence_encoder", "finetuned", "code", "all-mpnet-base-v2/code_translation_pairs_32_3000_50_50")
    create_query_dataset("test", "sentence_encoder", "finetuned", "tac_code", "all-mpnet-base-v2/tac_code_translation_pairs_32_3000_50_50")
    create_query_dataset("test", "ngram", "untrained", "opcode")
    create_query_dataset("test", "smartembed", "untrained", "func_code")
    create_query_dataset("test", "codebert", "untrained", "func_code")
    evaluate_nicad_results()
    # create evaluation graphs
    eval_graphs()
    dist_graphs()


if __name__ == '__main__':
    main()
