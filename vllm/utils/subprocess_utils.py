# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import _thread
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from multiprocessing.connection import _ConnectionBase
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pywintypes import HANDLEType

_shutdown_handlers: dict[int, Callable[[int, FrameType | None], object]] = {}


@contextmanager
def register_shutdown_signal(
    signum: int, handler: Callable[[int, FrameType | None], object]
) -> Iterator[None]:
    """Register a thread-safe handler for signals and owned shutdown requests."""
    previous = signal.signal(signum, handler)
    previous_relay = _shutdown_handlers.get(signum)
    _shutdown_handlers[signum] = handler
    try:
        yield
    finally:
        signal.signal(signum, previous)
        if previous_relay is None:
            _shutdown_handlers.pop(signum, None)
        else:
            _shutdown_handlers[signum] = previous_relay


class WindowsJob:
    """Own a Windows process tree, including descendants whose parent has exited."""

    _handle: "HANDLEType | None"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise NotImplementedError("Windows Job Objects require Windows")
        import win32job

        self._handle = win32job.CreateJobObject(None, "")
        try:
            info = win32job.QueryInformationJobObject(
                self._handle, win32job.JobObjectExtendedLimitInformation
            )
            info["BasicLimitInformation"]["LimitFlags"] |= (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            win32job.SetInformationJobObject(
                self._handle, win32job.JobObjectExtendedLimitInformation, info
            )
        except BaseException:
            self._handle.Close()
            raise

    def assign(self, process_handle: int) -> None:
        import win32job

        win32job.AssignProcessToJobObject(self._handle, process_handle)

    def close(self, timeout: float = 10.0) -> None:
        """Force termination and wait for every owned process before closing."""
        import win32job

        if self._handle is None:
            return
        try:
            win32job.TerminateJobObject(self._handle, 1)
            deadline = time.monotonic() + timeout
            while win32job.QueryInformationJobObject(
                self._handle, win32job.JobObjectBasicAccountingInformation
            )["ActiveProcesses"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Windows process job did not terminate")
                time.sleep(0.01)
        finally:
            self._handle.Close()
            self._handle = None


def _relay_shutdown_signal(control: _ConnectionBase) -> None:
    """Dispatch an owned request without broadcasting Windows console events."""
    with control:
        try:
            signum = control.recv()
        except EOFError:
            return
        handler = _shutdown_handlers.get(signum)
        if handler is not None:
            handler(signum, None)
        elif callable(signal.getsignal(signum)):
            # Initialization and synchronous CLI handlers may raise SystemExit
            # or KeyboardInterrupt, which must be delivered on the main thread.
            _thread.interrupt_main(signum)
        elif signal.getsignal(signum) != signal.SIG_IGN:
            # Match termination during startup, before a handler is installed.
            os._exit(1)


def _run_in_process_group(
    target: Callable[..., None],
    args: tuple[Any, ...],
    ready: Event | None,
    shutdown_control: _ConnectionBase | None,
) -> None:
    if ready is not None:
        ready.wait()
    elif sys.platform != "win32":
        os.setpgrp()
    else:
        raise ValueError("A Windows child process requires a startup gate")
    if shutdown_control is not None:
        threading.Thread(
            target=_relay_shutdown_signal, args=(shutdown_control,), daemon=True
        ).start()
    target(*args)


def start_process_in_group(
    context: BaseContext,
    target: Callable[..., None],
    *,
    args: tuple[Any, ...] = (),
    name: str | None = None,
    job: WindowsJob | None = None,
    shutdown_control: _ConnectionBase | None = None,
) -> BaseProcess:
    """Start a child only after its Windows job owns it, or in a POSIX group.

    Event-loop handlers registered with register_shutdown_signal can be invoked
    on a relay thread. Other Python handlers run on the child's main thread.
    """
    if sys.platform == "win32" and job is None:
        raise ValueError("A Windows child process requires an owning job")
    ready = context.Event() if sys.platform == "win32" else None
    process = context.get_context().Process(
        target=_run_in_process_group,
        args=(target, args, ready, shutdown_control),
        name=name,
    )
    process.start()
    try:
        if ready is not None:
            assert job is not None
            job.assign(process.sentinel)
            ready.set()
    except BaseException:
        process.terminate()
        process.join()
        raise
    return process


def popen_with_inherited_sockets(
    cmd: list[str], sockets: Sequence[socket.socket]
) -> subprocess.Popen[bytes]:
    """Launch a child process that inherits the supplied sockets."""
    fds = tuple(sock.fileno() for sock in sockets)
    if sys.platform != "win32":
        for fd in fds:
            os.set_inheritable(fd, True)
        return subprocess.Popen(cmd, pass_fds=fds)

    was_inheritable = [sock.get_inheritable() for sock in sockets]
    try:
        for sock in sockets:
            sock.set_inheritable(True)
        startup_info = subprocess.STARTUPINFO()
        startup_info.lpAttributeList = {"handle_list": list(fds)}
        return subprocess.Popen(
            cmd,
            close_fds=True,
            startupinfo=startup_info,
        )
    finally:
        for sock, inheritable in zip(sockets, was_inheritable):
            sock.set_inheritable(inheritable)
