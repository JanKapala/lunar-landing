import random
from copy import deepcopy

import torch


class ReplayBuffer:
    def __init__(self, batch_size=1, max_size=None):
        self.buffer = []
        self.batch_size = batch_size
        self._max_size = int(max_size)

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, new_value):
        self._max_size = int(new_value)

    def __lshift__(self, record):
        self._add_record(record)

    def __rshift__(self, record):
        self._add_record(record)

    def _add_record(self, record):
        assert len(record) == 5
        record = deepcopy(record)
        state, action, reward, next_state, done = record

        state = torch.Tensor(state)
        action = torch.Tensor(action)
        reward = torch.Tensor([reward])
        next_state = torch.Tensor(next_state)
        done = torch.Tensor([done])

        record = state, action, reward, next_state, done

        if self._max_size is not None and len(self.buffer) >= self._max_size:
            self.buffer = self.buffer[-(self._max_size - 1):]
        self.buffer.append(record)

    def __iter__(self):
        return self

    def __next__(self):
        batch_size = min(self.batch_size, len(self.buffer))
        if batch_size == 0:
            raise Exception("Replay Buffer is Empty")

        raw_records = random.sample(self.buffer, k=batch_size)

        raw_states = []
        raw_actions = []
        raw_rewards = []
        raw_next_states = []
        raw_dones = []

        for record in raw_records:
            state, action, reward, next_state, done = record
            raw_states.append(state)
            raw_actions.append(action)
            raw_rewards.append(reward)
            raw_next_states.append(next_state)
            raw_dones.append(done)

        states = torch.stack(raw_states)
        actions = torch.stack(raw_actions)
        rewards = torch.stack(raw_rewards)
        next_states = torch.stack(raw_next_states)
        dones = torch.stack(raw_dones)

        batch = (states, actions, rewards, next_states, dones)

        return batch

    def __str__(self):
        return self.buffer.__str__()

    def __repr__(self):
        return self.__str__()
