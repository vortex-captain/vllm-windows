# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import signal
import sys

import pytest

from vllm.entrypoints.launchers.launcher import _register_shutdown_signals


@pytest.mark.skipif(sys.platform != "win32", reason="Windows winloop signals")
def test_winloop_shutdown_signals_without_uvicorn_capture():
    winloop = pytest.importorskip("winloop")

    async def check():
        loop = asyncio.get_running_loop()
        delivered = asyncio.Event()
        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGBREAK)
        previous = {sig: signal.getsignal(sig) for sig in signals}
        try:
            _register_shutdown_signals(loop, delivered.set)
            for sig in signals:
                delivered.clear()
                signal.raise_signal(sig)
                await asyncio.wait_for(delivered.wait(), timeout=2)
        finally:
            for sig, handler in previous.items():
                loop.remove_signal_handler(sig)
                signal.signal(sig, handler)

    winloop.run(check())
