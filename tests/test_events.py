"""Event bus tests: fan-out, non-blocking publish, and the drop-oldest backpressure policy."""

from __future__ import annotations

from pharos.events import EventBus, RequestFailed, RequestStarted


def _started(request_id: int) -> RequestStarted:
    return RequestStarted(
        request_id=request_id,
        method="POST",
        path="/api/chat",
        endpoint="ollama-chat",
        model="m",
        stream=True,
    )


async def test_fan_out_delivers_to_every_subscriber() -> None:
    bus = EventBus()
    first, second = bus.subscribe(), bus.subscribe()
    event = _started(1)
    bus.publish(event)
    assert first.get_nowait() is event
    assert second.get_nowait() is event


def test_publish_without_subscribers_is_a_noop() -> None:
    EventBus().publish(_started(1))  # must not raise


async def test_full_queue_drops_oldest_never_blocks() -> None:
    bus = EventBus(queue_size=2)
    queue = bus.subscribe()
    bus.publish(_started(1))
    bus.publish(_started(2))
    bus.publish(_started(3))  # queue full: event 1 is dropped to admit event 3
    assert queue.qsize() == 2
    kept = [queue.get_nowait().request_id, queue.get_nowait().request_id]
    assert kept == [2, 3]


async def test_unsubscribe_stops_delivery_and_is_idempotent() -> None:
    bus = EventBus()
    queue = bus.subscribe()
    bus.unsubscribe(queue)
    bus.unsubscribe(queue)  # second removal is ignored
    bus.publish(RequestFailed(request_id=1, error="x"))
    assert queue.qsize() == 0


async def test_dropped_count_is_tracked_per_subscriber() -> None:
    bus = EventBus(queue_size=2)
    lagging = bus.subscribe()
    for i in range(5):
        bus.publish(_started(i))
    assert bus.dropped_count(lagging) == 3

    fresh = bus.subscribe()
    assert bus.dropped_count(fresh) == 0
    assert lagging.qsize() == 2  # still holds the 2 newest events
