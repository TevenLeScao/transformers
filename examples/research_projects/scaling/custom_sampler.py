import math
from itertools import chain
from typing import Optional, List

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, DataCollatorForLanguageModeling


class ConsecutiveSampler(torch.utils.data.Sampler):

    def __init__(self, data_source, bounds, batch_size, shuffle=True):
        self.data_source = data_source
        self.bounds = bounds + [len(self.data_source)]
        self.batch_size = batch_size
        self.shuffle = shuffle

    @property
    def num_samples(self):
        return self.bounds[-1]

    def __iter__(self):
        n_texts = len(self.bounds) - 1

        # shuffle texts
        if self.shuffle:
            g = torch.Generator()
            text_perm = torch.randperm(n_texts, generator=g).tolist()
            sample_order = list(chain.from_iterable([range(self.bounds[i], self.bounds[i + 1]) for i in text_perm]))
        else:
            sample_order = list(range(len(self)))

        # consecutive order
        sequence_length = len(sample_order) // self.batch_size
        clamp_len = sequence_length * self.batch_size
        batched_order = list(
            chain.from_iterable([sample_order[i:clamp_len:sequence_length] for i in range(sequence_length)]))
        return iter(batched_order)

    def __len__(self):
        return self.num_samples


class DistributedConsecutiveSampler(torch.utils.data.Sampler):

    def __init__(self, data_source, bounds, batch_size, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.data_source = data_source
        self.bounds = bounds + [len(self.data_source)]
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.total_size = int(math.ceil(len(data_source) / batch_size / num_replicas)) * batch_size * num_replicas
        self.num_samples = self.total_size // num_replicas
        self.shuffle = shuffle
        self.data_source = data_source
        self.bounds = bounds + [len(self.data_source)]
        self.batch_size = batch_size

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            n_texts = len(self.bounds) - 1
            text_perm = torch.randperm(n_texts, generator=g).tolist()
            sample_order = list(chain.from_iterable([range(self.bounds[i], self.bounds[i + 1]) for i in text_perm]))
        else:
            sample_order = list(range(len(self)))

        # add extra samples to make it evenly divisible
        sample_order += sample_order[:(self.total_size - len(sample_order))]
        assert len(sample_order) == self.total_size, \
            f"Total indices length {len(sample_order)} and dataset size {self.total_size} mismatched"

        # subsample
        indices = sample_order[self.rank * self.num_samples:(self.rank + 1) * self.num_samples]

        # reorder to be consecutive when batched
        sequence_length = len(indices) // self.batch_size
        clamp_len = sequence_length * self.batch_size
        indices = list(chain.from_iterable([indices[i:clamp_len:sequence_length] for i in range(sequence_length)]))

        assert len(indices) == self.num_samples, \
            f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class LongRangeTrainer(Trainer):

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        text_starts = [idx for idx, value in enumerate(self.train_dataset["start_of_doc"]) if value]

        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            train_sampler = None
        else:
            train_sampler = (
                ConsecutiveSampler(self.train_dataset, text_starts, self.args.train_batch_size, shuffle=True)
                if self.args.local_rank == -1
                else DistributedConsecutiveSampler(self.train_dataset, text_starts,
                                                   self.args.per_device_train_batch_size)
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            eval_sampler = None
        else:
            eval_sampler = (
                ConsecutiveSampler(eval_dataset, [0], self.args.train_batch_size, shuffle=False)
                if self.args.local_rank == -1
                else DistributedConsecutiveSampler(eval_dataset, [0],
                                                   self.args.per_device_eval_batch_size)
            )

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            test_sampler = None
        else:
            test_sampler = (
                ConsecutiveSampler(test_dataset, [0], self.args.train_batch_size)
                if self.args.local_rank == -1
                else DistributedConsecutiveSampler(test_dataset, [0],
                                                   self.args.per_device_eval_batch_size)
            )

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )


class VerboseDataCollator(DataCollatorForLanguageModeling):

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        for i, example in enumerate(self.tokenizer.batch_decode(examples)):
            print(example)
            print(f"{i}" * 89)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
