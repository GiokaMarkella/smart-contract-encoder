from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, util
from sentence_transformers.evaluation import TranslationEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from smart_contract_encoder.models.base_model import BaseModel
from smart_contract_encoder.utils import CODERANKEMBED_MODEL_DIR, INPUT_LEVEL, INPUT_TYPE


class CodeRankEmbedEncoder(BaseModel):

    def __init__(
        self,
        load: bool = False,
        model_to_load: str = None,
        input_level: str = None,
        input_type: str = None,
        batch_size: int = 32,
        save_steps: int = 3000,
        eval_steps: int = 50,
        logging_steps: int = 50,
    ):
        self.input_level = input_level
        self.input_type = input_type
        self.batch_size = batch_size
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.similarity_fn_name = "cosine"

        if not load and input_level is not None:
            self.save_path = self.args_to_path(
                input_level=input_level,
                input_type=input_type,
                batch_size=batch_size,
                save_steps=save_steps,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
            )
        elif load and model_to_load is not None:
            self.save_path = Path(model_to_load)
        else:
            self.save_path = None

        model_name = model_to_load if load else "nomic-ai/CodeRankEmbed"
        self._model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)

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

    @staticmethod
    def args_to_name(
        input_level: str = None,
        input_type: str = None,
        batch_size: int = 32,
        save_steps: int = 3000,
        eval_steps: int = 50,
        logging_steps: int = 50,
    ):
        if input_level not in INPUT_LEVEL:
            raise Exception(f'Input must be one of {", ".join(INPUT_LEVEL)}')
        if input_type not in INPUT_TYPE:
            raise Exception(f'Input must be one of {", ".join(INPUT_TYPE)}')
        return f"{input_level}_{input_type}_{batch_size}_{save_steps}_{eval_steps}_{logging_steps}"

    @classmethod
    def args_to_path(
        cls,
        input_level: str = None,
        input_type: str = None,
        batch_size: int = 32,
        save_steps: int = 3000,
        eval_steps: int = 50,
        logging_steps: int = 50,
    ):
        return Path.joinpath(
            CODERANKEMBED_MODEL_DIR,
            cls.args_to_name(
                input_level=input_level,
                input_type=input_type,
                batch_size=batch_size,
                save_steps=save_steps,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
            ),
        )

    def finetune_pairs(self, eval_dataset, train_dataset):
        if self.save_path is None:
            raise ValueError("A save path is required to fine-tune CodeRankEmbed.")
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        loss = MultipleNegativesRankingLoss(self._model)
        args = SentenceTransformerTrainingArguments(
            output_dir=CODERANKEMBED_MODEL_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_ratio=0.1,
            fp16=True,
            bf16=False,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="steps",
            save_only_model=1,
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=2,
            logging_steps=self.logging_steps,
        )
        evaluator = TranslationEvaluator(
            source_sentences=eval_dataset["anchor"],
            target_sentences=eval_dataset["positive"],
        )
        trainer = SentenceTransformerTrainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
        )
        trainer.train()
        self._model.save_pretrained(str(self.save_path))
        hist = pd.DataFrame(trainer.state.log_history)
        hist.to_pickle(f"{str(self.save_path)}.pkl")
