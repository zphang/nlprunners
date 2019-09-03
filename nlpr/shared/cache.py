import hashlib
import json
import os

import torch

import pyutils.datastructures as datastructures


class BaseDCache:
    def write(self, key, data):
        raise NotImplementedError()

    def read(self, key):
        raise NotImplementedError()

    def __contains__(self, key):
        raise NotImplementedError()


class MemoryDCache(BaseDCache):
    def __init__(self):
        self.storage = {}

    def write(self, key, data):
        _, hashstring = self._hash_key(key)
        self.storage[hashstring] = data

    def read(self, key):
        _, hashstring = self._hash_key(key)
        return self.storage[hashstring]

    def __contains__(self, key):
        _, hashstring = self._hash_key(key)
        return hashstring in self.storage

    @classmethod
    def _hash_key(cls, key):
        assert isinstance(key, dict)
        string = json.dumps(datastructures.sort_dict(key)).replace("\n", "")
        hashstring = md5(string)
        return string, hashstring


class FileSystemDCache(BaseDCache):
    def __init__(self, base_path):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def write(self, key, data):
        string, hashstring = self._hash_key(key)
        self._add_log(f"{hashstring}\t{string}")
        torch.save(data, os.path.join(self.base_path, hashstring))

    def read(self, key):
        _, hashstring = self._hash_key(key)
        return torch.load(os.path.join(self.base_path, hashstring))

    def _add_log(self, write_string):
        with open(os.path.join(self.base_path, "cache_log.txt"), "a") as f:
            f.write(write_string)

    def __contains__(self, key):
        _, hashstring = self._hash_key(key)
        return os.path.exists(os.path.join(self.base_path, hashstring))

    @classmethod
    def _hash_key(cls, key):
        assert isinstance(key, dict)
        string = json.dumps(datastructures.sort_dict(key)).replace("\n", "")
        hashstring = md5(string)
        return string, hashstring


def md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()
