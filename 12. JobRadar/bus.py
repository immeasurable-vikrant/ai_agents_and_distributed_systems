"""
bus.py — the EVENT BUS. Kafka, with an in-memory fallback.

WHY AN EVENT BUS IN A JOB SEARCH TOOL:

Ingestion is fast and free (HTTP calls). Enrichment is slow and costs
money (LLM calls). Wiring them together directly means a slow or failing
agent blocks new jobs from arriving. Putting a log between them decouples
the two completely — and lets multiple independent agents react to the
same event.

THE TOPOLOGY:

    ingest workers ──→ [jobs.discovered] ─┬→ classifier  (group: classify)
                                          └→ metrics     (group: metrics)
    classifier ──────→ [jobs.classified] ──→ salary      (group: salary)
    salary ──────────→ [jobs.enriched]   ──→ dashboard

⭐ THE ONE IDEA THAT MATTERS — consumer groups:

    SAME group_id      → consumers SHARE the work    (queue semantics)
                         scale by adding consumers
    DIFFERENT group_id → each group gets EVERY message (pub/sub fan-out)
                         add a new agent without touching the producer

`classify` and `metrics` both read jobs.discovered with DIFFERENT group
ids, so both see every job. Run two classifier processes with the SAME
group id and they split the load. One primitive, two patterns, chosen by
a string.

TWO BACKENDS, auto-detected:
  kafka     real durability, replay, multiple consumer groups
  memory    asyncio.Queue — same API, no Kafka container needed

The memory backend is NOT durable: kill the process and in-flight events
are gone. That's exactly what makes the Kafka comparison instructive.
"""

import asyncio
import json
import os
from collections import defaultdict
from typing import AsyncIterator

KAFKA_URL = os.getenv("KAFKA_URL", "")
BACKEND = "kafka" if KAFKA_URL else "memory"

TOPIC_DISCOVERED = "jobs.discovered"
TOPIC_CLASSIFIED = "jobs.classified"
TOPIC_ENRICHED = "jobs.enriched"


def backend_name() -> str:
    return BACKEND


# ============================================================
# IN-MEMORY BACKEND
# ============================================================
# One asyncio.Queue PER (topic, group). That's what reproduces pub/sub
# fan-out: publishing writes a copy into every group's queue, so two
# groups each get their own copy — exactly like Kafka consumer groups.

_queues: dict[tuple[str, str], asyncio.Queue] = {}
_groups: dict[str, set[str]] = defaultdict(set)


def _register(topic: str, group: str) -> asyncio.Queue:
    key = (topic, group)
    if key not in _queues:
        _queues[key] = asyncio.Queue(maxsize=1000)
        #   maxsize IS backpressure. With an unbounded queue, a fast
        #   producer and slow LLM consumers means the queue grows until
        #   you OOM. Bounded means the producer WAITS — slowness
        #   propagates backward as a signal instead of forward as a crash.
        _groups[topic].add(group)
    return _queues[key]


async def _mem_publish(topic: str, event: dict):
    # Fan out to every registered group — the pub/sub part.
    for group in list(_groups[topic]):
        await _queues[(topic, group)].put(event)


async def _mem_consume(topic: str, group: str) -> AsyncIterator[dict]:
    q = _register(topic, group)
    while True:
        yield await q.get()


# ============================================================
# KAFKA BACKEND
# ============================================================

_producer = None


async def _get_producer():
    global _producer
    if _producer is None:
        from aiokafka import AIOKafkaProducer
        _producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_URL,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        await _producer.start()
    return _producer


async def _kafka_publish(topic: str, event: dict):
    p = await _get_producer()
    await p.send_and_wait(topic, event)


async def _kafka_consume(topic: str, group: str) -> AsyncIterator[dict]:
    from aiokafka import AIOKafkaConsumer
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_URL,
        group_id=group,                      # ⭐ the pub/sub vs queue switch
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        #   MANUAL COMMIT, deliberately. With auto-commit, Kafka marks a
        #   message consumed as soon as it's delivered — so a crash
        #   mid-processing LOSES the job silently. Committing only after
        #   successful processing gives at-least-once delivery, which
        #   plus idempotent consumers = effectively exactly-once.
    )
    await consumer.start()
    try:
        async for msg in consumer:
            yield msg.value
            await consumer.commit()          # ← only after processing
    finally:
        await consumer.stop()


# ============================================================
# PUBLIC API
# ============================================================

async def publish(topic: str, event: dict):
    if BACKEND == "kafka":
        await _kafka_publish(topic, event)
    else:
        await _mem_publish(topic, event)


async def consume(topic: str, group: str) -> AsyncIterator[dict]:
    """
    Subscribe to a topic as part of a consumer group.

        consume("jobs.discovered", "classify")   ─┐ different groups →
        consume("jobs.discovered", "metrics")    ─┘ BOTH see every job

        two processes with group="classify"        → they SPLIT the jobs
    """
    if BACKEND == "kafka":
        async for e in _kafka_consume(topic, group):
            yield e
    else:
        async for e in _mem_consume(topic, group):
            yield e


def subscribe_group(topic: str, group: str):
    """Pre-register a memory-backend group so it doesn't miss early events."""
    if BACKEND == "memory":
        _register(topic, group)


async def close():
    global _producer
    if _producer is not None:
        await _producer.stop()
        _producer = None


def stats() -> dict:
    """Queue depths — the backpressure signal, visible in the dashboard."""
    if BACKEND != "memory":
        return {"backend": "kafka", "note": "use kafka-consumer-groups for lag"}
    return {
        "backend": "memory",
        "queues": {f"{t}:{g}": q.qsize() for (t, g), q in _queues.items()},
    }