from typing import Optional

import datasets
import lightning
from torch.utils.data import DataLoader


class C4DataModule(lightning.LightningDataModule):
    def __init__(
        self,
        path: str = "liweili/c4_200m",
        val_sample: int = 100000,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        streaming: bool = True,
    ) -> None:
        super().__init__()
        self.path = path
        self.val_sample = val_sample
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.streaming = streaming

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if not self.train_dataset or not self.val_dataset:
            data = datasets.load_dataset(
                self.path, streaming=self.streaming, split="train"
            ).shuffle(seed=42, buffer_size=10_000)

            self.train_dataset = data.skip(self.val_sample)
            self.val_dataset = data.take(self.val_sample)

    @staticmethod
    def group_batch(batch):
        """Group a batch of examples into a list of examples.

        :param batch: A batch of examples
        :return: A list of examples
        """
        return {k: [v] for k, v in batch.items()}

    def train_dataloader(self) -> DataLoader:
        """Create the train dataloader for the T5 model.

        :return: The train dataloader
        """
        return self.train_dataset.map(
            function=self.group_batch, batched=True, batch_size=self.train_batch_size
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader for the T5 model.

        :return: The validation dataloader
        """
        return self.val_dataset.map(
            function=self.group_batch, batched=True, batch_size=self.val_batch_size
        )
