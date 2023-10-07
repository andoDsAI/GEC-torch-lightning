from typing import Any, Callable, Dict, List

import lightning
import torch
import torch.nn as nn
import wandb


class T5Module(lightning.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        compute_metrics: Callable = None,
        generation_kwargs: Dict[str, Any] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        adam_epsilon: float = 1e-8,
        loss_fn: Callable = nn.CrossEntropyLoss(),
        prepend_sentence: str = "",
        is_freeze: bool = False,
        freeze_layers: int = 0,
        unfreeze_batch_idx: int = 10000,
        ignore_index: int = -100,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.generation_kwargs = generation_kwargs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.loss_fn = loss_fn
        self.prepend_sentence = prepend_sentence
        self.is_freeze = is_freeze
        self.freeze_layers = freeze_layers
        self.unfreeze_batch_idx = unfreeze_batch_idx
        self.ignore_index = ignore_index
        self.compile = compile

        self.decoder_start_token_id = model.config.decoder_start_token_id

        self.model.train()
        if self.is_freeze:
            for layer in self.model.encoder.block[: self.freeze_layers]:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

            for layer in self.model.decoder.block[: self.freeze_layers]:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

        self.table_logging = 0
        self.validation_outs = {
            "references": [],
            "good_grammar_preds": [],
            "bad_grammar_preds": [],
        }

    def prepend_tokens(
        self, tokenized_batch: Dict[str, torch.Tensor], batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Prepend the start token to the decoder input ids and attention mask.

        :param tokenized_batch: the tokenized batch
        :param batch_size: the batch size
        :return:
        """
        prepend_input_ids = torch.tensor(
            [self.decoder_start_token_id] * batch_size, dtype=torch.long
        )[:, None].to(self.device)
        prepend_attention_mask = torch.tensor([1] * batch_size, dtype=torch.long)[:, None].to(
            self.device
        )

        input_ids = torch.cat([prepend_input_ids, tokenized_batch["input_ids"]], dim=-1)
        attention_mask = torch.cat(
            [prepend_attention_mask, tokenized_batch["attention_mask"]], dim=-1
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """Perform a forward pass through the model.

        :param inputs: the inputs
        :return: outputs of the model
        """
        return self.model(**inputs)

    def compute_loss(
        self, input_sentences: List[str], output_sentences: List[str]
    ) -> torch.Tensor:
        """Tokenize the input and output sentences and compute the loss.

        :param input_sentences: a list of input sentences.
        :param output_sentences: a list of output sentences.
        :return:
        """
        tokenized_input = self.tokenizer(
            [self.prepend_sentence + sentence for sentence in input_sentences],
            device=self.device,
        )
        tokenized_output = self.prepend_tokens(
            self.tokenizer(
                output_sentences,
                device=self.device,
            ),
            batch_size=len(output_sentences),
        )
        labels = tokenized_output["input_ids"].clone()
        labels[labels[:, :] == self.tokenizer.pad_token_id] = self.ignore_index

        inputs = {
            "input_ids": tokenized_input["input_ids"],
            "attention_mask": tokenized_input["attention_mask"],
            "decoder_input_ids": tokenized_output["input_ids"],
            "decoder_attention_mask": tokenized_output["attention_mask"],
        }
        output = self(inputs)
        logits = output.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def common_step(self, batch: Dict[str, List[str]]) -> torch.Tensor:
        """Perform a common step for training and validation.

        :param batch: the batch of examples
        :return: the loss
        """
        input_sentences = batch["input"]
        output_sentences = batch["output"]

        bad_grammar_loss = self.compute_loss(input_sentences, output_sentences)
        good_grammar_loss = self.compute_loss(output_sentences, output_sentences)
        return bad_grammar_loss + good_grammar_loss

    def training_step(self, batch: Dict[str, List[str]], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on the batch of data.

        :param batch: A batch of data (a dict) containing the keys: "input" and "output"
        :param batch_idx: The index of the current batch
        :return: A tensor containing the loss
        """
        loss = self.common_step(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch["input"]),
        )

        return loss

    def validation_step(self, batch: Dict[str, List[str]], batch_idx: int) -> torch.Tensor:
        """Perform a single validation step on the batch of data.

        :param batch: A batch of data (a dict) containing the keys: "input" and "output"
        :param batch_idx: The index of the current batch
        :return: A tensor containing the loss
        """
        loss = self.common_step(batch)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch["input"]),
        )

        if batch_idx == 0:
            self.log_examples(batch)

        return loss

    def on_validation_epoch_end(self) -> None:
        pass

    def predict(self, input_sentences: List[str]) -> List[str]:
        tokenized_input = self.tokenizer(
            [self.prepend_sentence + sentence for sentence in input_sentences],
            device=self.device,
        )
        outputs = self.model.generate(
            **tokenized_input,
            **self.generation_kwargs,
        )
        prediction = self.tokenizer.batch_decode(outputs)
        return prediction

    def log_examples(self, batch: Dict[str, List[str]]):
        input_sentences = batch["input"]
        output_sentences = batch["output"]
        bad_gram_predictions = self.predict(input_sentences)
        good_gram_predictions = self.predict(output_sentences)

        columns = ["input", "bad_gram_predictions", "good_gram_predictions", "output"]
        data = [
            [input_sentence, bad_gram_prediction, good_gram_prediction, output_sentence]
            for input_sentence, bad_gram_prediction, good_gram_prediction, output_sentence in zip(
                input_sentences, bad_gram_predictions, good_gram_predictions, output_sentences
            )
        ]
        table = wandb.Table(data=data, columns=columns)
        if self.logger is not None:
            self.table_logging += 1
            self.logger.experiment.log({f"Epoch_{self.table_logging} results": table})

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        :return: The optimizer
        """
        if self.is_freeze:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                eps=self.adam_epsilon,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                eps=self.adam_epsilon,
            )
        return optimizer
