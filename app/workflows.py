"""Leave It — the durable half.

A fuse is not a `sleep`. A `sleep` dies with the process, and the whole point of
sealing something for two weeks is that your laptop will not be awake for two
weeks. So each seal is a Temporal workflow whose timer lives on the server:

    SealWorkflow(item, fuse)
        waits on either  (a) a `rescue` signal  -> unseal now, or
                         (b) the durable timer -> resurface it once
    kill -9 the worker and the timer is untouched: on restart the same workflow
    run resumes and fires at the same wall-clock moment.

That gives the product one guarantee it could not otherwise make: a decision to
ignore something cannot be lost by a crash, a reboot, or a closed laptop lid.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    import core


@dataclass
class SealRequest:
    item_id: str
    p_handle: float
    fuse_seconds: int
    threshold: float = core.DEFAULT_THRESHOLD


@activity.defn
async def record_seal(req: SealRequest) -> str:
    info = activity.info()
    verdict = core.decide(req.p_handle, req.threshold)
    seal = core.Seal(item_id=req.item_id, p_handle=req.p_handle, verdict=verdict,
                     reason=core.reason_line(req.p_handle, verdict),
                     sealed_at=__import__("time").time(), fuse_seconds=req.fuse_seconds,
                     workflow_id=info.workflow_id, run_id=info.workflow_run_id or "")
    core.append(seal)
    return seal.to_json()


@activity.defn
async def resurface(item_id: str, waited_seconds: int) -> str:
    core.update_state(item_id, "LAPSED",
                      f"fuse expired after {waited_seconds}s; nobody ever engaged")
    return item_id


@activity.defn
async def rescue(item_id: str, why: str) -> str:
    core.update_state(item_id, "RESCUED", why)
    return item_id


@workflow.defn
class SealWorkflow:
    def __init__(self) -> None:
        self._rescued = False
        self._why = ""

    @workflow.signal
    async def on_attention(self, why: str) -> None:
        """Called the moment anyone engages with a sealed item."""
        self._why = why
        self._rescued = True

    @workflow.run
    async def run(self, req: SealRequest) -> str:
        await workflow.execute_activity(record_seal, req,
                                        start_to_close_timeout=timedelta(seconds=30))
        if req.fuse_seconds <= 0:
            return "NO_FUSE"
        # One wait, two ways out. The timer is server-side and survives the worker.
        try:
            await workflow.wait_condition(lambda: self._rescued,
                                          timeout=timedelta(seconds=req.fuse_seconds))
        except TimeoutError:
            await workflow.execute_activity(resurface, args=[req.item_id, req.fuse_seconds],
                                            start_to_close_timeout=timedelta(seconds=30),
                                            retry_policy=RetryPolicy(maximum_attempts=5))
            return "LAPSED"
        await workflow.execute_activity(rescue, args=[req.item_id, self._why],
                                        start_to_close_timeout=timedelta(seconds=30))
        return "RESCUED"
