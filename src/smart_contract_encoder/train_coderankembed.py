from smart_contract_encoder.load_data import load_dataset
from smart_contract_encoder.models.coderankeembed_model import CodeRankEmbedEncoder


def main():
    field = "code"
    training_dataset_type = "translation_pairs"
    train_dataset, eval_dataset = load_dataset(
        file_type="training",
        field=field,
        split="train",
        training_dataset_type=training_dataset_type,
    )
    encoder = CodeRankEmbedEncoder(
        load=False,
        input_level=field,
        input_type=training_dataset_type,
    )
    encoder.finetune_pairs(eval_dataset, train_dataset)


if __name__ == "__main__":
    main()
