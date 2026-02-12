import math
from itertools import chain
from abc import ABCMeta, abstractmethod

import torch
from torch.utils.data import random_split, IterableDataset
from functools import cache

from typing import Iterable, Generic, TypeVar, Iterator

Index = TypeVar("Index")
Element = TypeVar("Element")


class GroupableDataset(IterableDataset, Generic[Index], metaclass=ABCMeta):
    """A mixin for datasets that can be grouped by a certain key.

    This is useful for datasets that have a hierarchical structure, such as micrographs and particles.
    """

    Group = TypeVar("Group", bound=Iterable[Index])

    @property
    @abstractmethod
    def __groups__(self) -> Iterable[Group]:
        """Returns an iterable of groups, where each group is an iterable of indices."""

    @abstractmethod
    def __element_iter__(self, indices: Iterable[Index]) -> Iterator[Element]:
        """Returns an iterable for each groups of its elements"""

    @cache
    def _flat_indices(self):
        return list(chain.from_iterable(self.__groups__()))

    def __len__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        indices = self._flat_indices()

        per_worker = int(math.ceil(len(indices) / num_workers))

        return per_worker * num_workers

    def __iter__(self):

        indices = self._flat_indices()

        # Split workload between workers
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            per_worker = int(math.ceil(len(indices) / float(worker_info.num_workers)))
            worker_id = worker_info.id

            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(indices))

            indices = indices[iter_start:iter_end]

        return self.__element_iter__(indices)
    

class GroupedSubset(GroupableDataset):

    def __init__(self, dataset: GroupableDataset[Index], groups: Iterable[Index]):
        super().__init__()

        self.dataset = dataset
        self.groups = groups

    def __groups__(self):
        return self.groups
    
    def __element_iter__(self, indices):
        return self.dataset.__element_iter__(indices)


def random_split_groups(dataset: GroupableDataset, lengths, generator=None):
    """Randomly split a GroupableDataset into non-overlapping new datasets of given lengths.
    The groups are kept intact, i.e. all samples from a group are assigned to the same split.
    """

    def _get_particles(groups, *, indices: Iterable[int]):
        return [groups[i] for i in indices]

    groups = list(dataset.__groups__())
    group_lengths = [len(group) for group in groups]
    indices_subgroups = random_split(group_lengths, lengths, generator)

    group_splits = [_get_particles(groups, indices=i) for i in indices_subgroups]

    return (GroupedSubset(dataset, group) for group in group_splits)