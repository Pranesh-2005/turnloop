"""The boundary between the agent loop and whatever is displaying it.

One rule, and the entire concurrency design follows from it:

    **The agent loop never touches a widget, and the UI never awaits the model.**

Communication is one memory object stream outbound (events) and one one-shot
stream per permission request inbound (the answer). The loop's task parks on
`ask()` while the UI keeps repainting, handling keystrokes and streaming the tool
output that is already on screen.

Three implementations share this interface: the Textual app, the headless printer,
and a null channel for tests and experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from turnloop.core.events import UIEvent
from turnloop.permissions.engine import PermissionDecision, PermissionRequest, Scope


@dataclass(slots=True)
class PendingPermission:
    request: PermissionRequest
    reply_send: MemoryObjectSendStream[PermissionDecision]
    reply_recv: MemoryObjectReceiveStream[PermissionDecision]

    @classmethod
    def create(cls, request: PermissionRequest) -> PendingPermission:
        send, recv = anyio.create_memory_object_stream[PermissionDecision](1)
        return cls(request=request, reply_send=send, reply_recv=recv)

    async def answer(self, decision: PermissionDecision) -> None:
        await self.reply_send.send(decision)

    async def wait(self) -> PermissionDecision:
        return await self.reply_recv.receive()


class UIChannel(ABC):
    """What the agent loop is given instead of a display."""

    @abstractmethod
    async def send(self, event: UIEvent) -> None: ...

    @abstractmethod
    async def ask(self, request: PermissionRequest) -> PermissionDecision: ...

    def for_subagent(self, subagent_id: str) -> UIChannel:
        """A channel that tags everything with a subagent id.

        Subagent output has to be visually separable — a fan-out of three children
        interleaving into one transcript is unreadable — and its permission
        prompts must still reach the one modal the parent owns.
        """
        return TaggedChannel(self, subagent_id)


@dataclass
class TaggedChannel(UIChannel):
    inner: UIChannel
    subagent_id: str

    async def send(self, event: UIEvent) -> None:
        if hasattr(event, "subagent_id") and getattr(event, "subagent_id", None) is None:
            try:
                object.__setattr__(event, "subagent_id", self.subagent_id)
            except AttributeError:  # slots without the field
                pass
        await self.inner.send(event)

    async def ask(self, request: PermissionRequest) -> PermissionDecision:
        request.subagent_id = self.subagent_id
        return await self.inner.ask(request)


@dataclass
class StreamChannel(UIChannel):
    """Pushes events into a memory stream. Used by the Textual app.

    The send stream is buffered: a burst of token deltas must not block the agent
    loop waiting for the UI to catch up, and if the buffer does fill, dropping
    display events is better than stalling model consumption.
    """

    events: MemoryObjectSendStream[UIEvent]
    permissions: MemoryObjectSendStream[PendingPermission]

    async def send(self, event: UIEvent) -> None:
        try:
            self.events.send_nowait(event)
        except anyio.WouldBlock:
            # Buffer full: wait, but only briefly. A permanently stuck UI should
            # not hang the model stream.
            with anyio.move_on_after(1.0):
                await self.events.send(event)
        except anyio.BrokenResourceError:
            pass  # UI is gone; the loop is being torn down

    async def ask(self, request: PermissionRequest) -> PermissionDecision:
        pending = PendingPermission.create(request)
        await self.permissions.send(pending)
        return await pending.wait()

    @staticmethod
    def create(buffer: int = 4096):
        ev_send, ev_recv = anyio.create_memory_object_stream[UIEvent](buffer)
        pm_send, pm_recv = anyio.create_memory_object_stream[PendingPermission](8)
        return StreamChannel(events=ev_send, permissions=pm_send), ev_recv, pm_recv


@dataclass
class NullChannel(UIChannel):
    """Discards output; auto-answers permission prompts.

    Used by experiments and tests. `auto_approve=False` is the interesting
    setting: it exercises the denial path, which is how the permission-safety
    grader checks that plan mode and deny rules actually hold.
    """

    auto_approve: bool = True
    seen: list[UIEvent] = field(default_factory=list)
    asked: list[PermissionRequest] = field(default_factory=list)
    record: bool = False

    async def send(self, event: UIEvent) -> None:
        if self.record:
            self.seen.append(event)

    async def ask(self, request: PermissionRequest) -> PermissionDecision:
        self.asked.append(request)
        return PermissionDecision(
            approved=self.auto_approve,
            scope=Scope.ONCE,
            reason="" if self.auto_approve else "non-interactive session",
        )
