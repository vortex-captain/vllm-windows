# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import socket
import threading

import pytest
import zmq

from vllm.utils import network_utils
from vllm.utils.network_utils import (
    get_file_store_init_method,
    get_open_port,
    get_open_ports_list,
    get_tcp_uri,
    join_host_port,
    make_zmq_listener,
    make_zmq_path,
    make_zmq_socket,
    split_host_port,
    split_zmq_path,
)


def test_get_file_store_init_method_is_unique():
    init_methods = {get_file_store_init_method() for _ in range(2)}

    assert len(init_methods) == 2
    assert all(method.startswith("file://") for method in init_methods)


def _call_with_timeout(func, timeout: float = 10.0):
    """Run func in a daemon thread so a livelock regression fails the test
    quickly instead of hanging the suite."""
    result: dict = {}

    def target():
        try:
            result["value"] = func()
        except BaseException as e:
            result["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"call did not finish within {timeout}s"
    if "error" in result:
        raise result["error"]
    return result["value"]


def test_get_open_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", "5678")
        # make sure we can get multiple ports, even if the env var is set
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
            s1.bind(("localhost", get_open_port()))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                s2.bind(("localhost", get_open_port()))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s3:
                    s3.bind(("localhost", get_open_port()))


def test_get_open_port_vllm_port_in_dp_reserved_range(
    monkeypatch: pytest.MonkeyPatch,
):
    # VLLM_PORT falling inside the data-parallel reserved window used to make
    # get_open_port() loop forever (issue #50024). It must instead return a
    # port outside the reserved range.
    with monkeypatch.context() as m:
        m.setenv("VLLM_DP_MASTER_PORT", "5680")
        # 5682 is inside [5680, 5690).
        m.setenv("VLLM_PORT", "5682")
        port = _call_with_timeout(get_open_port)
        assert port not in range(5680, 5690)


def test_get_open_ports_list_with_vllm_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", "5678")
        ports = get_open_ports_list(5)
        assert len(ports) == 5
        assert len(set(ports)) == 5, "ports must be unique"

        # verify every port is actually bindable
        sockets = []
        try:
            for p in ports:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("localhost", p))
                sockets.append(s)
        finally:
            for s in sockets:
                s.close()


def test_get_open_port_skips_reserved_dp_master_ports(
    monkeypatch: pytest.MonkeyPatch,
):
    """VLLM_PORT == VLLM_DP_MASTER_PORT must not livelock by returning the
    same free-but-reserved port forever (upstream issue #50024)."""
    base = 42123
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", str(base))
        m.setenv("VLLM_DP_MASTER_PORT", str(base))
        port = _call_with_timeout(get_open_port)
        assert port >= base + 10


def test_get_open_ports_list_skips_reserved_dp_master_ports(
    monkeypatch: pytest.MonkeyPatch,
):
    """The VLLM_PORT band scan must not hand out ports inside the window
    reserved for the data parallel master process."""
    base = 42223
    reserved = range(base, base + 10)
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", str(base))
        m.setenv("VLLM_DP_MASTER_PORT", str(base))
        ports = _call_with_timeout(lambda: get_open_ports_list(5))
        assert len(ports) == 5
        assert len(set(ports)) == 5, "ports must be unique"
        assert all(p not in reserved for p in ports)


def test_get_open_port_ephemeral_skips_reserved_range(
    monkeypatch: pytest.MonkeyPatch,
):
    """An ephemeral port that happens to fall in the reserved range is
    replaced by a rescan starting past the range."""
    base = 42323
    with monkeypatch.context() as m:
        m.delenv("VLLM_PORT", raising=False)
        m.setenv("VLLM_DP_MASTER_PORT", str(base))
        m.setattr(
            network_utils,
            "_get_open_port",
            lambda start_port=None, max_attempts=None: (
                base if start_port is None else start_port
            ),
        )
        assert _call_with_timeout(get_open_port) == base + 10


def test_get_open_port_ephemeral_without_dp_master_port(
    monkeypatch: pytest.MonkeyPatch,
):
    """Without VLLM_PORT and VLLM_DP_MASTER_PORT, an ephemeral port is
    returned as before."""
    with monkeypatch.context() as m:
        m.delenv("VLLM_PORT", raising=False)
        m.delenv("VLLM_DP_MASTER_PORT", raising=False)
        port = _call_with_timeout(get_open_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))


@pytest.mark.parametrize(
    "path,expected",
    [
        ("ipc://some_path", ("ipc", "some_path", "")),
        ("tcp://127.0.0.1:5555", ("tcp", "127.0.0.1", "5555")),
        ("tcp://[::1]:5555", ("tcp", "::1", "5555")),  # IPv6 address
        ("inproc://some_identifier", ("inproc", "some_identifier", "")),
    ],
)
def test_split_zmq_path(path, expected):
    assert split_zmq_path(path) == expected


@pytest.mark.parametrize(
    "invalid_path",
    [
        "invalid_path",  # Missing scheme
        "tcp://127.0.0.1",  # Missing port
        "tcp://[::1]",  # Missing port for IPv6
        "tcp://:5555",  # Missing host
    ],
)
def test_split_zmq_path_invalid(invalid_path):
    with pytest.raises(ValueError):
        split_zmq_path(invalid_path)


def test_make_zmq_socket_ipv6():
    # Check if IPv6 is supported by trying to create an IPv6 socket
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.close()
    except OSError:
        pytest.skip("IPv6 is not supported on this system")

    ctx = zmq.Context()
    ipv6_path = "tcp://[::]:5555"  # IPv6 loopback address
    socket_type = zmq.REP  # Example socket type

    # Create the socket
    zsock: zmq.Socket = make_zmq_socket(ctx, ipv6_path, socket_type)

    # Verify that the IPV6 option is set
    assert zsock.getsockopt(zmq.IPV6) == 1, (
        "IPV6 option should be enabled for IPv6 addresses"
    )

    # Clean up
    zsock.close()
    ctx.term()


@pytest.mark.parametrize("scheme", ["tcp", "ipc"])
def test_make_zmq_socket_adopts_listener(tmp_path, scheme):
    path = (
        "tcp://127.0.0.1:0" if scheme == "tcp" else f"ipc://{tmp_path}/inherited.sock"
    )
    listener = make_zmq_listener(path, zmq.ROUTER)
    ctx = zmq.Context()
    router = dealer = None
    try:
        router = make_zmq_socket(
            ctx,
            listener.address,
            zmq.ROUTER,
            bind=True,
            listener=listener.socket,
        )
        assert listener.socket.fileno() == -1
        dealer = make_zmq_socket(
            ctx, listener.address, zmq.DEALER, bind=False, identity=b"client"
        )
        dealer.send(b"ping")
        assert router.recv_multipart() == [b"client", b"ping"]
    finally:
        if router is not None:
            router.close(linger=0)
        if dealer is not None:
            dealer.close(linger=0)
        ctx.term()
        listener.cleanup()


def test_make_zmq_listener_failed_ipc_bind_keeps_existing_path(tmp_path):
    family = getattr(socket, "AF_UNIX", None)
    if family is None:
        pytest.fail("Unix-domain socket paths are not supported on this platform")
    path = tmp_path / "taken.sock"
    owner = socket.socket(family, socket.SOCK_STREAM)
    owner.bind(str(path))
    owner.listen()
    try:
        with pytest.raises(OSError, match="Address already in use"):
            make_zmq_listener(f"ipc://{path}", zmq.ROUTER)

        with socket.socket(family, socket.SOCK_STREAM) as client:
            client.connect(str(path))
    finally:
        owner.close()


def test_make_zmq_listener_reports_missing_unix_socket_support(tmp_path, monkeypatch):
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    with pytest.raises(NotImplementedError, match="IPC.*AF_UNIX"):
        make_zmq_listener(f"ipc://{tmp_path}/unsupported.sock", zmq.ROUTER)
    assert not list(tmp_path.iterdir())


def test_failed_zmq_adoption_preserves_listener_ownership(monkeypatch):
    listener = make_zmq_listener("tcp://127.0.0.1:0", zmq.ROUTER)
    original_fd = listener.socket.fileno()
    failed_sockets = []
    set_option = zmq.Socket.setsockopt

    def reject_adoption(sock, option, value):
        if option == zmq.USE_FD:
            failed_sockets.append(sock)
            raise zmq.ZMQError(zmq.ENOTSUP)
        return set_option(sock, option, value)

    monkeypatch.setattr(zmq.Socket, "setsockopt", reject_adoption)
    try:
        with zmq.Context() as ctx:
            with pytest.raises(zmq.ZMQError):
                make_zmq_socket(
                    ctx, listener.address, zmq.ROUTER, listener=listener.socket
                )
            assert listener.socket.fileno() == original_fd
            assert len(failed_sockets) == 1
            assert failed_sockets[0].closed
            with socket.socket() as competitor, pytest.raises(OSError):
                competitor.bind(listener.socket.getsockname())
    finally:
        listener.cleanup()


def _reply_on_inherited_listener(listener, address, ready):
    with zmq.Context() as ctx:
        router = make_zmq_socket(ctx, address, zmq.ROUTER, listener=listener)
        try:
            router.setsockopt(zmq.RCVTIMEO, 10_000)
            ready.set()
            identity, payload = router.recv_multipart()
            router.send_multipart([identity, payload])
        finally:
            router.close(linger=0)


def test_spawned_zmq_listener_keeps_endpoint_reserved():
    """Socket sharing must preserve the bound endpoint after the parent closes it."""
    listener = make_zmq_listener("tcp://127.0.0.1:0", zmq.ROUTER)
    address = listener.address
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(
        target=_reply_on_inherited_listener,
        args=(listener.socket, address, ready),
    )
    process.start()
    listener.close()
    try:
        assert ready.wait(30)
        with socket.socket() as competitor, pytest.raises(OSError):
            competitor.bind(("127.0.0.1", int(address.rsplit(":", 1)[1])))
        with (
            zmq.Context() as zmq_context,
            make_zmq_socket(
                zmq_context, address, zmq.DEALER, bind=False, linger=0
            ) as client,
        ):
            client.setsockopt(zmq.RCVTIMEO, 10_000)
            client.send(b"inherited")
            assert client.recv() == b"inherited"
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join(10)
        listener.cleanup()


def test_make_zmq_path():
    assert make_zmq_path("tcp", "127.0.0.1", 5555) == "tcp://127.0.0.1:5555"
    assert make_zmq_path("tcp", "::1", 5555) == "tcp://[::1]:5555"


def test_get_tcp_uri():
    assert get_tcp_uri("127.0.0.1", 5555) == "tcp://127.0.0.1:5555"
    assert get_tcp_uri("::1", 5555) == "tcp://[::1]:5555"


def test_split_host_port():
    # valid ipv4
    assert split_host_port("127.0.0.1:5555") == ("127.0.0.1", 5555)
    # invalid ipv4
    with pytest.raises(ValueError):
        # multi colon
        assert split_host_port("127.0.0.1::5555")
    with pytest.raises(ValueError):
        # tailing colon
        assert split_host_port("127.0.0.1:5555:")
    with pytest.raises(ValueError):
        # no colon
        assert split_host_port("127.0.0.15555")
    with pytest.raises(ValueError):
        # none int port
        assert split_host_port("127.0.0.1:5555a")

    # valid ipv6
    assert split_host_port("[::1]:5555") == ("::1", 5555)
    # invalid ipv6
    with pytest.raises(ValueError):
        # multi colon
        assert split_host_port("[::1]::5555")
    with pytest.raises(IndexError):
        # no colon
        assert split_host_port("[::1]5555")
    with pytest.raises(ValueError):
        # none int port
        assert split_host_port("[::1]:5555a")


def test_join_host_port():
    assert join_host_port("127.0.0.1", 5555) == "127.0.0.1:5555"
    assert join_host_port("::1", 5555) == "[::1]:5555"
