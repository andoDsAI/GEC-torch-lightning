import logging
import os
from typing import List

import hydra
import rootutils
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, T5ForConditionalGeneration

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import C4DataModule
from src.models import GeneratorConfig, T5Module, Tokenizer
from src.utils import compute_metrics, instantiate_callbacks, instantiate_loggers

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

py_logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    py_logger.info(OmegaConf.to_yaml(cfg))

    py_logger.info(f"Initializing datamodule <{C4DataModule.__name__}>...")
    if cfg.data.num_workers > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    datamodule = C4DataModule(**cfg.data)

    py_logger.info(f"Initializing model <{T5ForConditionalGeneration.__name__}>...")
    model = T5ForConditionalGeneration.from_pretrained(cfg.model.model_name_or_path)
    tokenizer = Tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=cfg.tokenizer.model_name_or_path,
            use_fast=cfg.tokenizer.use_fast,
        ),
        max_seq_len=cfg.tokenizer.max_seq_len,
    )

    generator_config = GeneratorConfig(**cfg.generator)
    lightning_module = T5Module(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        generation_kwargs=generator_config.beam_search_params,
        **cfg.model.lightning_module,
    )

    py_logger.info("Initializing callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    py_logger.info("Initializing logger...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    py_logger.info("Initializing trainer...")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    py_logger.info("Training...")
    trainer.fit(
        model=lightning_module,
        datamodule=datamodule,
    )

    py_logger.info("Ending training...")


if __name__ == "__main__":
    main()
