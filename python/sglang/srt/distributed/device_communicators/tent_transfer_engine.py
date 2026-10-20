"""KVTransferEngine backed by Mooncake's native TENT Python module (`import tent`)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.distributed.device_communicators.kv_transfer_engine import (
    KVTransferEngine,
    TransferIntent,
)
from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

# Same budget as mooncake.engine's batch_transfer_sync: MC_TRANSFER_TIMEOUT
# seconds (default 30, floor 5) plus 1 ns per byte.
_DEFAULT_TRANSFER_TIMEOUT_S = 30
_MIN_TRANSFER_TIMEOUT_S = 5

# Requests per TENT batch and submit threads. A batch status poll scans every task,
# and submit_transfer releases the GIL, so small chunks on a pool overlap building
# requests with submitting them. Tuned for per-request CPU cost; re-measure if it changes.
_SUBMIT_CHUNK = 128
_SUBMIT_THREADS = 4
# Busy-poll this many times per batch before yielding the CPU between polls.
# Arbitrary; polling cadence had no measurable effect on throughput.
_SPIN_POLLS = 64
_POLL_SLEEP_S = 20e-6

# tent.Request arguments after `length`: (priority, transport_hint, policy_name,
# deadline_ns, intent_type). Empty means TENT's defaults.
_RequestTail = Tuple[Any, ...]


def _build_tent_config(hostname: str, ib_device: Optional[str]) -> dict:
    force_tcp = (
        envs.MOONCAKE_PROTOCOL.get() == "tcp" or os.environ.get("MC_FORCE_TCP") == "1"
    )
    config = {
        "metadata_type": "p2p",
        "rpc_server_hostname": hostname,
        "rpc_server_port": 0,
        "log_level": "warning",
        # TENT's default metrics server binds a fixed port; every TP rank would collide.
        "metrics": {"enabled": False},
        # Pin the data path: same-host peers must not silently fall back to NVLink/SHM.
        "transports": {
            "rdma": {"enable": not force_tcp},
            "tcp": {"enable": force_tcp},
            "hp_tcp": {"enable": False},
            "shm": {"enable": False},
            "nvlink": {"enable": False},
            "mnnvl": {"enable": False},
            "gds": {"enable": False},
            "io_uring": {"enable": False},
        },
    }
    if ib_device and not force_tcp:
        config["topology"] = {
            "rdma_whitelist": [d.strip() for d in ib_device.split(",") if d.strip()]
        }
    override = envs.SGLANG_TENT_CONFIG_OVERRIDE.get()
    if override:
        overrides = json.loads(override)
        if not isinstance(overrides, dict):
            raise ValueError("SGLANG_TENT_CONFIG_OVERRIDE must be a JSON object")
        _deep_merge(config, overrides)
    return config


def _deep_merge(base: dict, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _transfer_timeout_ns() -> int:
    timeout_env = os.environ.get("MC_TRANSFER_TIMEOUT")
    if not timeout_env:
        return _DEFAULT_TRANSFER_TIMEOUT_S * 1_000_000_000
    try:
        timeout_s = int(timeout_env)
    except ValueError as e:
        raise ValueError(
            f"MC_TRANSFER_TIMEOUT must be an integer number of seconds, got {timeout_env!r}"
        ) from e
    return max(_MIN_TRANSFER_TIMEOUT_S, timeout_s) * 1_000_000_000


class TentTransferEngine(KVTransferEngine):
    """Shared TENT engine for RDMA transfers; one per process, never torn down.

    Like the classic engine it is a process singleton, so its submit pool lives
    until the process exits; KVManager teardown on a P<->D role switch reuses it.
    """

    def __init__(self, hostname: str, ib_device: Optional[str]):
        try:
            import tent
        except ImportError as e:
            raise ImportError(
                "SGLANG_MOONCAKE_TRANSFER_ENGINE=tent needs the native `tent` module; "
                "build Mooncake with -DUSE_TENT=ON and put tent.*.so on PYTHONPATH."
            ) from e
        self._tent = tent
        # Passed positionally: the intent_type= keyword doubles request build cost.
        self._request_tail: Dict[Optional[TransferIntent], _RequestTail] = {
            None: (),
            TransferIntent.FOREGROUND: (
                tent.PRIO_HIGH,
                tent.TransportType.UNSPEC,
                None,
                0,
                tent.IntentType.FOREGROUND_GET,
            ),
            TransferIntent.BACKGROUND: (
                tent.PRIO_LOW,
                tent.TransportType.UNSPEC,
                None,
                0,
                tent.IntentType.BACKGROUND_PREFETCH,
            ),
        }
        self._failed_states = (
            tent.TransferStatusEnum.FAILED,
            tent.TransferStatusEnum.TIMEOUT,
            tent.TransferStatusEnum.CANCELED,
            tent.TransferStatusEnum.INVALID,
        )
        self.ib_device = ib_device

        config = _build_tent_config(hostname, ib_device)
        with tempfile.NamedTemporaryFile(
            "w", prefix="sglang-tent-", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            config_path = f.name
        try:
            self.engine = tent.TransferEngine(config_path)
        finally:
            os.unlink(config_path)
        if not self.engine.available():
            raise RuntimeError(
                f"TENT TransferEngine failed to start (hostname={hostname!r}, "
                f"ib_device={ib_device!r})"
            )

        # p2p segment names are "host:rpc_port", the same format as a classic
        # session id, so the PD bootstrap exchange needs no change.
        self.session_id = NetworkAddress(
            hostname, self.engine.get_rpc_server_port()
        ).to_host_port_str()
        self._segment_handles: Dict[str, int] = {}
        self._segment_lock = threading.Lock()
        # Registered region sizes, since TENT's unregister needs the length.
        self._registered: Dict[int, int] = {}
        self._submit_pool = ThreadPoolExecutor(
            max_workers=_SUBMIT_THREADS, thread_name_prefix="tent-submit"
        )

        self._timeout_ns = _transfer_timeout_ns()
        logger.info(
            "TENT transfer engine ready: session_id=%s ib_device=%s",
            self.session_id,
            ib_device,
        )

    def register(self, ptr: int, length: int) -> None:
        try:
            self.engine.register_local_memory(ptr, length)
        except self._tent.TentException as exc:
            raise RuntimeError("TENT memory registration failed") from exc
        self._registered[ptr] = length

    def deregister(self, ptr: int) -> None:
        try:
            self.engine.unregister_local_memory(ptr, self._registered.pop(ptr, 0))
        except self._tent.TentException as exc:
            raise RuntimeError("TENT memory deregistration failed") from exc

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        try:
            self.engine.register_local_memory_batch(list(ptrs), list(lengths))
        except self._tent.TentException as exc:
            logger.debug("TENT batch memory registration failed: %s", exc)
            return -1
        self._registered.update(zip(ptrs, lengths))
        return 0

    def batch_deregister(self, ptrs: List[int]) -> int:
        lengths = [self._registered.pop(ptr, 0) for ptr in ptrs]
        try:
            self.engine.unregister_local_memory_batch(list(ptrs), lengths)
        except self._tent.TentException as exc:
            logger.debug("TENT batch memory deregistration failed: %s", exc)
            return -1
        return 0

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
        intent: Optional[TransferIntent] = None,
    ) -> int:
        if not (len(buffers) == len(peer_buffer_addresses) == len(lengths)):
            logger.error("buffers, peer_buffer_addresses and lengths differ in size")
            return -1
        if not buffers:
            return 0
        try:
            handle = self._open_segment(session_id)
            rc = self._write_and_wait(
                handle=handle,
                buffers=buffers,
                peer_buffer_addresses=peer_buffer_addresses,
                lengths=lengths,
                request_tail=self._request_tail[intent],
            )
        except self._tent.TentException as exc:
            logger.error(
                "TENT batch transfer to %s failed (%d requests): %s",
                session_id,
                len(buffers),
                exc,
            )
            rc = -1
        if rc != 0:
            # Re-resolve the peer next time, in case it restarted on a new segment.
            self._close_segment(session_id)
        return rc

    def send_probe(self, peer_session_id: str) -> int:
        # TENT has no liveness RPC; a successful segment-info fetch proves the
        # peer's RPC server answers. Always re-resolve so a restarted peer is seen.
        self._close_segment(peer_session_id)
        try:
            handle = self._open_segment(peer_session_id)
            self.engine.get_segment_info(handle)
        except self._tent.TentException:
            self._close_segment(peer_session_id)
            return -1
        return 0

    def _open_segment(self, session_id: str) -> int:
        handle = self._segment_handles.get(session_id)
        if handle is not None:
            return handle
        with self._segment_lock:
            handle = self._segment_handles.get(session_id)
            if handle is None:
                handle = self.engine.open_segment(session_id)
                self._segment_handles[session_id] = handle
            return handle

    def _close_segment(self, session_id: str) -> None:
        with self._segment_lock:
            handle = self._segment_handles.pop(session_id, None)
        if handle is None:
            return
        try:
            self.engine.close_segment(handle)
        except self._tent.TentException as exc:
            logger.debug("TENT close_segment(%s) failed: %s", session_id, exc)

    def _write_and_wait(
        self,
        *,
        handle: int,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
        request_tail: _RequestTail,
    ) -> int:
        deadline_ns = time.monotonic_ns() + self._timeout_ns + sum(lengths)
        batch_ids = self._submit_all(
            handle=handle,
            buffers=buffers,
            peer_buffer_addresses=peer_buffer_addresses,
            lengths=lengths,
            request_tail=request_tail,
        )
        try:
            for batch_id in batch_ids:
                rc = self._wait_batch(batch_id=batch_id, deadline_ns=deadline_ns)
                if rc != 0:
                    return rc
            return 0
        finally:
            self._free_batches(batch_ids)

    def _submit_all(
        self,
        *,
        handle: int,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
        request_tail: _RequestTail,
    ) -> List[int]:
        """Submit every chunk; on any failure free what was submitted and re-raise."""

        def submit(start: int) -> int:
            end = start + _SUBMIT_CHUNK
            return self._submit_chunk(
                handle=handle,
                buffers=buffers[start:end],
                peer_buffer_addresses=peer_buffer_addresses[start:end],
                lengths=lengths[start:end],
                request_tail=request_tail,
            )

        starts = range(0, len(buffers), _SUBMIT_CHUNK)
        if len(starts) == 1:
            return [submit(0)]
        futures = [self._submit_pool.submit(submit, start) for start in starts]
        batch_ids: List[int] = []
        first_error: Optional[BaseException] = None
        for future in futures:
            try:
                batch_ids.append(future.result())
            except BaseException as exc:
                first_error = first_error or exc
        if first_error is not None:
            self._free_batches(batch_ids)
            raise first_error
        return batch_ids

    def _free_batches(self, batch_ids: List[int]) -> None:
        for batch_id in batch_ids:
            try:
                self.engine.free_transfer_batch(batch_id)
            except self._tent.TentException as exc:
                logger.debug("TENT free_transfer_batch failed: %s", exc)

    def _submit_chunk(
        self,
        *,
        handle: int,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
        request_tail: _RequestTail,
    ) -> int:
        request = self._tent.Request
        write = self._tent.OpCode.WRITE
        requests = [
            request(write, src, handle, dst, length, *request_tail)
            for src, dst, length in zip(buffers, peer_buffer_addresses, lengths)
        ]
        batch_id = self.engine.allocate_transfer_batch(len(requests))
        if not batch_id:
            raise self._tent.InternalError("allocate_transfer_batch returned 0")
        try:
            self.engine.submit_transfer(batch_id, requests)
        except BaseException:
            self.engine.free_transfer_batch(batch_id)
            raise
        return batch_id

    def _wait_batch(self, *, batch_id: int, deadline_ns: int) -> int:
        completed = self._tent.TransferStatusEnum.COMPLETED
        polls = 0
        while True:
            state = self.engine.get_transfer_status_overall(batch_id).state
            if state == completed:
                return 0
            if state in self._failed_states:
                logger.error("TENT transfer ended in state %s", state)
                return -1
            if time.monotonic_ns() > deadline_ns:
                logger.error("TENT transfer timed out")
                return -1
            polls += 1
            if polls > _SPIN_POLLS:
                time.sleep(_POLL_SLEEP_S)
