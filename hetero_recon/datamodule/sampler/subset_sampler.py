import itertools
import math
from typing import Any, Iterator, Sized

from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import (
    Dataset,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from coach_pl.configuration import configurable
from coach_pl.datamodule import SAMPLER_REGISTRY


@SAMPLER_REGISTRY.register()
class SubsetSampler(Sampler[int]):

    @configurable
    def __init__(
        self,
        data_source: Sized,
        shuffle: bool = False,
        max_num_samples: int | None = None,
    ) -> None:
        if max_num_samples is None:
            self.num_samples = len(data_source)
        else:
            self.num_samples = min(len(data_source), max_num_samples)

        self.shuffle = shuffle
        if self.shuffle:
            self.sampler = RandomSampler(data_source, num_samples=self.num_samples)
        else:
            self.sampler = SequentialSampler(data_source)

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage, dataset: Dataset) -> dict[str, Any]:
        is_training = (stage == RunningStage.TRAINING)
        is_validating = (stage == RunningStage.VALIDATING)

        max_num_samples = cfg.DATAMODULE.SAMPLER.MAX_NUM_SAMPLES
        if is_training:
            pass
        elif is_validating and max_num_samples is not None:
            max_num_samples = max_num_samples // 10
        else:
            max_num_samples = None

        shuffle = cfg.DATAMODULE.DATALOADER.TRAIN.SHUFFLE if is_training else False

        return {
            "data_source": dataset,
            "shuffle": shuffle,
            "max_num_samples": max_num_samples,
        }

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            yield from self.sampler
        else:
            yield from itertools.islice(self.sampler, self.num_samples)

    def __len__(self) -> int:
        return self.num_samples


@SAMPLER_REGISTRY.register()
class SubsetDistributedSampler(DistributedSampler):

    @configurable
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Any = None,
        rank: Any = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        max_num_samples: int | None = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        self.max_num_samples = None
        if max_num_samples is not None:
            actual_global_samples = min(len(dataset), max_num_samples)
            self.max_num_samples = math.ceil(actual_global_samples / self.num_replicas)

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage, dataset: Dataset) -> dict[str, Any]:
        is_training = (stage == RunningStage.TRAINING)
        is_validating = (stage == RunningStage.VALIDATING)

        max_num_samples = cfg.DATAMODULE.SAMPLER.MAX_NUM_SAMPLES
        if is_training:
            pass
        elif is_validating and max_num_samples is not None:
            max_num_samples = max_num_samples // 10
        else:
            max_num_samples = None

        shuffle = cfg.DATAMODULE.DATALOADER.TRAIN.SHUFFLE if is_training else False
        drop_last = cfg.DATAMODULE.DATALOADER.DROP_LAST if is_training else False

        return {
            "dataset": dataset,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "max_num_samples": max_num_samples,
        }

    def __iter__(self):
        full_iterator = super().__iter__()

        if self.max_num_samples is None:
            return full_iterator

        return itertools.islice(full_iterator, self.max_num_samples)

    def __len__(self) -> int:
        if self.max_num_samples is None:
            return super().__len__()
        return self.max_num_samples
