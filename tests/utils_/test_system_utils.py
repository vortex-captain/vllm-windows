# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path

import pytest

from vllm.utils import system_utils
from vllm.utils.system_utils import _maybe_force_spawn, unique_filepath


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_kill_process_tree_uses_psutil_kill_on_windows(monkeypatch):
    killed: list[int] = []

    class FakeProcess:
        def __init__(self, pid: int, children: list["FakeProcess"] | None = None):
            self.pid = pid
            self._children = children or []

        def children(self, recursive: bool):
            assert recursive
            return self._children

        def kill(self):
            killed.append(self.pid)

    child_one = FakeProcess(11)
    child_two = FakeProcess(12)
    parent = FakeProcess(10, [child_one, child_two])

    monkeypatch.setattr(system_utils.sys, "platform", "win32")
    monkeypatch.setattr(system_utils.psutil, "Process", lambda pid: parent)

    system_utils.kill_process_tree(10)

    assert killed == [11, 12, 10]


def test_kill_process_tree_surfaces_access_denied_after_remaining_kills(monkeypatch):
    attempted: list[int] = []
    access_denied = system_utils.psutil.AccessDenied(pid=11)

    class FakeProcess:
        def __init__(
            self,
            pid: int,
            error: Exception | None = None,
            children: list["FakeProcess"] | None = None,
        ):
            self.pid = pid
            self._error = error
            self._children = children or []

        def children(self, recursive: bool):
            assert recursive
            return self._children

        def kill(self):
            attempted.append(self.pid)
            if self._error is not None:
                raise self._error

    child_denied = FakeProcess(11, access_denied)
    child_exited = FakeProcess(12, system_utils.psutil.NoSuchProcess(pid=12))
    parent = FakeProcess(10, children=[child_denied, child_exited])

    monkeypatch.setattr(system_utils.sys, "platform", "win32")
    monkeypatch.setattr(system_utils.psutil, "Process", lambda pid: parent)

    with pytest.raises(system_utils.psutil.AccessDenied) as exc_info:
        system_utils.kill_process_tree(10)

    assert exc_info.value is access_denied
    assert attempted == [11, 12, 10]
