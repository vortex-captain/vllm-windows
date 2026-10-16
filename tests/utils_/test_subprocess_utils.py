# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket
import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from vllm.utils.subprocess_utils import popen_with_inherited_sockets


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
        sockets = [
            stack.enter_context(socket.socket()) for _ in range(socket_count)
        ]
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
