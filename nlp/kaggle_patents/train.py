import hydra
import pandas as pd
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from nlp.kaggle_patents import data
from nlp.kaggle_patents.callbacks import SaveBestModelCallback
from nlp.kaggle_patents.metrics import Pearsonr


@hydra.main(config_name="config.yaml")
def simple_baseline(cfg: OmegaConf):
    """simplified version of a training script with only the bare essentials"""
    df = pd.read_csv(f"{cfg.data.input_dir}/train.csv")

    if cfg.debug:
        df = df.sample(100).reset_index()

    df = data.prepare_data(df, cfg.data.cpc_scheme_xml_dir, cfg.data.cpc_title_list_dir)
    df = data.create_folds(df)

    # load pretrained model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=1
    )
    # make it a regression problem
    model.classifier = torch.nn.Linear(model.config.hidden_size, 1)
    model.criterion = torch.nn.MSELoss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    def tokenize(text):
        res = tokenizer(text, padding=True, truncation=True, max_length=512)
        return res["input_ids"], res["token_type_ids"], res["attention_mask"]

    df[["input_ids", "token_type_ids", "attention_mask"]] = df.apply(
        lambda x: tokenize(x.input_text), axis=1, result_type="expand"
    )

    # the model expects certain column names for input and output
    df.rename(columns={"score": "label"}, inplace=True)
    ds_train = Dataset.from_pandas(df[df.fold != 4])
    ds_val = Dataset.from_pandas(df[df.fold == 4])

    trainer_args = TrainingArguments(**cfg.bert)

    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        compute_metrics=Pearsonr(),
        callbacks=[SaveBestModelCallback(metric_name="pearsonr")],
    )
    trainer.train()


if __name__ == "__main__":
    simple_baseline()
