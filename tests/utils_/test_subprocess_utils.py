# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import multiprocessing
import signal
import socket
import subprocess
import sys
import threading
from contextlib import ExitStack
from unittest.mock import patch

import psutil
import pytest

from vllm.utils.subprocess_utils import (
    WindowsJob,
    popen_with_inherited_sockets,
    register_shutdown_signal,
    start_process_in_group,
)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
def test_popen_inherits_only_socket_handle_on_windows():
    with socket.socket() as sock:
        fd = sock.fileno()
        assert not sock.get_inheritable()

        with patch("vllm.utils.subprocess_utils.subprocess.Popen") as popen:
            process = popen_with_inherited_sockets(["vllm-rs.exe"], [sock])

        assert process is popen.return_value
        popen.assert_called_once()
        _, kwargs = popen.call_args
        assert kwargs["close_fds"] is True
        assert kwargs["startupinfo"].lpAttributeList == {"handle_list": [fd]}
        assert "pass_fds" not in kwargs
        assert not sock.get_inheritable()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
@pytest.mark.parametrize("socket_count", [3, 4])
def test_popen_inherits_zmq_and_optional_grpc_sockets(socket_count):
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(socket_count)]
        sockets[-1].set_inheritable(True)
        before = [sock.get_inheritable() for sock in sockets]
        with patch("vllm.utils.subprocess_utils.subprocess.Popen") as popen:
            process = popen_with_inherited_sockets(["vllm-rs.exe"], sockets)

        assert process is popen.return_value
        popen.assert_called_once()
        _, kwargs = popen.call_args
        assert kwargs["startupinfo"].lpAttributeList == {
            "handle_list": [sock.fileno() for sock in sockets]
        }
        assert kwargs["close_fds"] is True
        assert "pass_fds" not in kwargs
        assert [sock.get_inheritable() for sock in sockets] == before


def _wait_for_stop(ready, stop):
    ready.set()
    stop.wait()


def _spawn_grandchild(results, stop, root_exits):
    child = subprocess.Popen(
        [sys.executable, "-c", "import threading; threading.Event().wait()"],
    )
    results.send(child.pid)
    results.close()
    if not root_exits:
        stop.wait()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")
@pytest.mark.parametrize("root_exits", [False, True])
def test_windows_job_reaps_descendants_but_not_unrelated_processes(root_exits):
    context = multiprocessing.get_context("spawn")
    stop, ready = context.Event(), context.Event()
    unrelated = context.Process(target=_wait_for_stop, args=(ready, stop))
    job = WindowsJob()
    recv, send = context.Pipe(duplex=False)
    process = None
    try:
        unrelated.start()
        assert ready.wait(30)
        process = start_process_in_group(
            context,
            _spawn_grandchild,
            args=(send, context.Event(), root_exits),
            job=job,
        )
        send.close()
        assert recv.poll(30)
        grandchild = psutil.Process(recv.recv())
        if root_exits:
            process.join(30)
            assert process.exitcode == 0
        job.close()
        process.join(10)
        assert not process.is_alive()
        grandchild.wait(timeout=10)
        assert unrelated.is_alive()
    finally:
        job.close()
        recv.close()
        send.close()
        if process is not None:
            process.join(10)
        stop.set()
        if unrelated.pid is not None:
            unrelated.join(10)
            if unrelated.is_alive():
                unrelated.kill()
                unrelated.join(10)


def _write_started(path):
    path.write_text("started", encoding="utf-8")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")
def test_failed_job_assignment_does_not_run_child_target(tmp_path, monkeypatch):
    job = WindowsJob()
    started = tmp_path / "started"

    def reject_assignment(_handle):
        raise PermissionError("assignment rejected")

    monkeypatch.setattr(job, "assign", reject_assignment)
    try:
        with pytest.raises(PermissionError, match="assignment rejected"):
            start_process_in_group(
                multiprocessing.get_context("spawn"),
                _write_started,
                args=(started,),
                job=job,
            )
        assert not started.exists()
    finally:
        job.close()


def _serve_until_shutdown(ready):
    import winloop

    async def serve():
        shutdown = asyncio.Event()
        loop = asyncio.get_running_loop()
        with register_shutdown_signal(
            signal.SIGTERM,
            lambda _sig, _frame: loop.call_soon_threadsafe(shutdown.set),
        ):
            ready.set()
            await asyncio.wait_for(shutdown.wait(), 10)

    winloop.run(serve())


def _wait_for_main_thread_shutdown(ready):
    def exit_handler(_signum, _frame):
        assert threading.current_thread() is threading.main_thread()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, exit_handler)
    receive, send = multiprocessing.Pipe(duplex=False)
    try:
        ready.set()
        while True:
            receive.poll(timeout=1)
    finally:
        receive.close()
        send.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects")
@pytest.mark.parametrize(
    "target", [_serve_until_shutdown, _wait_for_main_thread_shutdown]
)
def test_shutdown_control_exits_gracefully_before_forced_cleanup(target):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    receive, send = context.Pipe(duplex=False)
    job = WindowsJob()
    process = None
    try:
        process = start_process_in_group(
            context,
            target,
            args=(ready,),
            job=job,
            shutdown_control=receive,
        )
        receive.close()
        assert ready.wait(30)
        send.send(int(signal.SIGTERM))
        process.join(15)
        assert process.exitcode == 0
    finally:
        receive.close()
        send.close()
        job.close()
        if process is not None:
            process.join(10)
