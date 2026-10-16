# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import subprocess
import sys
from collections.abc import Sequence


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
