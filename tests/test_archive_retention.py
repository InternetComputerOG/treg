"""Archive row retention: whole questions no one asked for go, with their snapshots, marks and
orphaned objects; everything something still needs stays (docs/context/architecture/archive.md,
"Row retention")."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from treg import archive, bootstrap, worker
from treg.application import archive_retention
from treg.config import get_settings
from treg.domain.catalog import store as catalog_store
from treg.infra.db import session_maker
from treg.models import (
    ArchiveEndpointStat, ArchiveKey, ArchiveKeyOrg, ArchiveRetentionCursor, ArchiveSnapshot,
    IdempotentCall,
)
from treg.timeutil import utcnow_naive
from tests.fake_object_store import MemoryObjectStore
from tests.test_archive import EP, platform_on, serve  # noqa: F401 - fixtures


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def _key(name: str, *, age_days: float, asked_days: float | None = None,
               endpoint: str = EP, policy: str = "transient", req_url: str | None = None,
               bodies: tuple[tuple[str, str | None], ...] = (("a", None),)) -> int:
    """One key created `age_days` ago, last asked `asked_days` ago (default: when created), with
    one snapshot per `(body name, body_storage)`; a dedup chain like the recorder writes it."""
    now = utcnow_naive()
    created = now - timedelta(days=age_days)
    asked = now - timedelta(days=age_days if asked_days is None else asked_days)
    async with session_maker() as s:
        key = ArchiveKey(key_hash=_h(name), endpoint_id=endpoint, provider="tikhub", policy=policy,
                         fetched_at=created, created_at=created, last_requested_at=asked,
                         req_method="GET", req_url=req_url or f"https://api.tikhub.io/x?q={name}")
        s.add(key)
        await s.flush()
        carrier = None
        for version, (body, storage) in enumerate(bodies, start=1):
            raw = body.encode()
            snap = ArchiveSnapshot(key_id=key.id, version=version, content_hash=_h(body),
                                   size_bytes=len(raw), fetched_at=created, body_storage=storage)
            if storage in (None, "db", "both"):
                if carrier is not None and carrier[1] == _h(body):
                    snap.body_of = carrier[0]
                else:
                    snap.body = raw
            s.add(snap)
            await s.flush()
            if snap.body is not None:
                carrier = (snap.id, _h(body))
        key.result_state, key.result_snapshot_id = "found", snap.id
        await s.commit()
        return int(key.id)


async def _mark(name: str, *, org_id: int = 1, days_ago: float = 40) -> None:
    at = utcnow_naive() - timedelta(days=days_ago)
    async with session_maker() as s:
        s.add(ArchiveKeyOrg(org_id=org_id, key_hash=_h(name), first_call_at=at, last_call_at=at))
        await s.commit()


async def _left() -> tuple[set[str], int, int]:
    async with session_maker() as s:
        keys = set((await s.execute(select(ArchiveKey.key_hash))).scalars().all())
        snaps = len((await s.execute(select(ArchiveSnapshot.id))).all())
        marks = len((await s.execute(select(ArchiveKeyOrg.key_hash))).all())
    return keys, snaps, marks


@pytest.fixture
def retention_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "archive_retention_enabled", True)
    monkeypatch.setattr(get_settings(), "archive_retention_pause_s", 0)


@pytest.fixture
def store():
    s = MemoryObjectStore()
    bootstrap.configure_archive_object_store(s)
    yield s
    bootstrap.configure_archive_object_store(None)


async def test_a_question_no_one_asked_for_goes_whole_and_everything_still_needed_stays(
        clients: AsyncClient, retention_on, monkeypatch):
    other_ep = next(e for e in catalog_store.load().by_id if e != EP)
    monkeypatch.setitem(catalog_store.load().by_id[other_ep], "cache", "archive")
    await _key("dormant", age_days=40, bodies=(("a", None), ("a", None), ("b", None)))
    await _mark("dormant", org_id=1)
    await _mark("dormant", org_id=2)
    await _key("never-asked", age_days=40, asked_days=None)
    async with session_maker() as s:            # NULL demand: a key only treg itself fetched
        key = (await s.execute(select(ArchiveKey).where(ArchiveKey.key_hash == _h("never-asked")))).scalar_one()
        key.last_requested_at = None
        await s.commit()
    await _key("asked-lately", age_days=40, asked_days=3)
    await _key("young", age_days=5)
    await _key("archive-recorded", age_days=40, policy="archive")
    await _key("archive-now", age_days=40, endpoint=other_ep)
    await _key("terminal", age_days=40, req_url="treg://asynctasks/call-1")
    await _key("paid-lately", age_days=40)
    await _mark("paid-lately", days_ago=2)
    await _key("retry-row", age_days=40)
    async with session_maker() as s:
        s.add(IdempotentCall(org_id=1, membership_id=1, key="k", status="done",
                             expires_at=utcnow_naive() + timedelta(hours=12),
                             archive_key_hash=_h("retry-row"), archive_content_hash=_h("a")))
        s.add(ArchiveEndpointStat(endpoint_id=EP, provider="tikhub", keys=9, snapshots=11,
                                  bodies_kept=11, kept_bytes=11))
        await s.commit()

    result = await archive_retention.run()

    assert (result["keys"], result["snapshots"], result["marks"]) == (2, 4, 2)
    assert result["pass_complete"] and result["stopped"] == "pass_complete"
    keys, snaps, marks = await _left()
    assert keys == {_h(n) for n in ("asked-lately", "young", "archive-recorded", "archive-now",
                                    "terminal", "paid-lately", "retry-row")}
    assert (snaps, marks) == (7, 1)
    async with session_maker() as s:
        stat = await s.get(ArchiveEndpointStat, EP)
        # 2 keys, 4 snapshots; 3 of them kept bytes (one is a dedup reference), 1+1+1 bytes
        assert (stat.keys, stat.snapshots, stat.bodies_kept, stat.kept_bytes) == (7, 7, 8, 8)
        # Every pointer a surviving key or snapshot holds still names a row.
        ids = set((await s.execute(select(ArchiveSnapshot.id))).scalars().all())
        pointers = set((await s.execute(select(ArchiveKey.result_snapshot_id))).scalars().all())
        carriers = set((await s.execute(select(ArchiveSnapshot.body_of)
                                        .where(ArchiveSnapshot.body_of.is_not(None)))).scalars().all())
        assert pointers - {None} <= ids and carriers <= ids
    # The team-facing read of a deleted question answers cleanly: no longer on file.
    assert await archive.resolve_result(_h("dormant"), _h("a")) is None
    assert await archive.resolve_result(_h("asked-lately"), _h("a")) is not None


async def test_off_by_default_and_a_dry_run_counts_without_deleting(clients: AsyncClient, monkeypatch):
    await _key("dormant", age_days=40, bodies=(("a", "r2"), ("b", None)))
    await _mark("dormant")
    assert get_settings().archive_retention_enabled is False
    off = await archive_retention.run()
    assert off["stopped"] == "disabled" and off["keys"] == 0
    dry = await archive_retention.run(dry_run=True, pause_s=0)
    assert (dry["keys"], dry["snapshots"], dry["marks"], dry["objects"]) == (1, 2, 1, 1)
    assert (await _left())[0] == {_h("dormant")}
    async with session_maker() as s:
        assert await s.get(ArchiveRetentionCursor, archive_retention.CURSOR_ID) is None


async def test_orphaned_objects_go_and_shared_ones_stay(clients: AsyncClient, retention_on, store):
    for body in ("only-here", "shared"):
        store.objects[_h(body)] = body.encode()
    await _key("dormant", age_days=40, bodies=(("only-here", "r2"), ("shared", "both")))
    await _key("kept", age_days=40, asked_days=1, bodies=(("shared", "r2"),))
    result = await archive_retention.run()
    assert (result["keys"], result["objects"]) == (1, 1)
    assert set(store.objects) == {_h("shared")}


async def test_no_rows_go_while_their_bytes_would_stay(clients: AsyncClient, retention_on, store):
    await _key("dormant", age_days=40, bodies=(("a", "r2"),))
    store.objects[_h("a")] = b"a"

    async def broken(_hash):
        raise RuntimeError("store down")
    store.delete = broken
    failed = await archive_retention.run()
    assert failed["keys"] == 0 and failed["skipped_batches"] == 1
    assert (await _left())[0] == {_h("dormant")}

    bootstrap.configure_archive_object_store(None)
    missing = await archive_retention.run(after=0)
    assert missing["stopped"] == "error" and "object store" in missing["error"]
    assert (await _left())[0] == {_h("dormant")}


async def test_a_run_stops_at_its_time_budget_and_the_next_resumes(clients: AsyncClient, retention_on):
    ids = [await _key(f"dormant-{i}", age_days=40) for i in range(3)]
    ticks = iter([0, 0, 0, 10, 10])        # start, batch 1, batch 2, then over budget
    first = await archive_retention.run(batch=1, max_seconds=5, after=ids[0] - 1,
                                        clock=lambda: next(ticks))
    assert (first["keys"], first["stopped"], first["next_after"]) == (2, "time_budget", ids[1])
    async with session_maker() as s:
        assert (await s.get(ArchiveRetentionCursor, archive_retention.CURSOR_ID)).after_id == ids[1]
    second = await archive_retention.run(batch=1)
    assert (second["start_after"], second["keys"], second["pass_complete"]) == (ids[1], 1, True)
    async with session_maker() as s:
        cursor = await s.get(ArchiveRetentionCursor, archive_retention.CURSOR_ID)
        assert (cursor.after_id, cursor.passes) == (0, 1)


async def test_a_recording_whose_key_went_meanwhile_records_it_again(clients: AsyncClient, serve,
                                                                      monkeypatch):
    """The recorder reads the key unlocked, then locks it; retention may delete it in between."""
    from sqlalchemy.exc import NoResultFound

    url = "https://api.tikhub.io/x?aweme_id=9"
    await archive._store(method="GET", endpoint_id=EP, provider="tikhub", url=url, caller_body=b"",
                         headers={}, status_code=200, media_type="application/json", body=b'{"n":1}')
    real, calls = archive._lock_archive_key, []

    async def deleted_then_real(s, key_id):
        calls.append(key_id)
        if len(calls) == 1:
            async with session_maker() as other:
                from treg.domain.governance.teams import erase_archive_keys
                await erase_archive_keys(other, [key_id])
                await other.commit()
            raise NoResultFound()
        return await real(s, key_id)
    monkeypatch.setattr(archive, "_lock_archive_key", deleted_then_real)
    await archive._store(method="GET", endpoint_id=EP, provider="tikhub", url=url, caller_body=b"",
                         headers={}, status_code=200, media_type="application/json", body=b'{"n":2}')
    async with session_maker() as s:
        keys = (await s.execute(select(ArchiveKey))).scalars().all()
        snaps = (await s.execute(select(ArchiveSnapshot))).scalars().all()
    assert len(keys) == 1 and [x.version for x in snaps] == [1] and len(calls) == 2


async def test_the_worker_command(clients: AsyncClient, monkeypatch, capsys):
    import treg.infra.db as infra_db

    async def verified():      # the server's startup check, not this command's, to test
        return None
    monkeypatch.setattr(infra_db, "verify_db", verified)
    await _key("dormant", age_days=40)
    args = argparse.Namespace(dry_run=True, days=None, batch_size=None, pause_seconds=0,
                              max_seconds=None, after=None)
    assert await worker._archive_retain(args) == 0
    assert json.loads(capsys.readouterr().out)["keys"] == 1
    args.dry_run = False
    assert await worker._archive_retain(args) == 0
    assert json.loads(capsys.readouterr().out)["stopped"] == "disabled"
    monkeypatch.setattr(get_settings(), "archive_retention_enabled", True)
    assert await worker._archive_retain(args) == 0
    assert json.loads(capsys.readouterr().out)["keys"] == 1
    assert (await _left())[0] == set()


async def test_every_statement_is_capped_on_postgres(clients: AsyncClient, retention_on):
    from sqlalchemy import event
    await _key("dormant", age_days=40)
    engine = session_maker.kw["bind"].sync_engine
    seen: list[str] = []

    def record(conn, cursor, statement, *args):
        seen.append(statement)
    event.listen(engine, "before_cursor_execute", record)
    try:
        assert (await archive_retention.run())["keys"] == 1
    finally:
        event.remove(engine, "before_cursor_execute", record)
    caps = [s for s in seen if s.startswith("SET LOCAL statement_timeout")]
    if engine.dialect.name != "postgresql":
        assert caps == []
        return
    deletes = [s for s in seen if s.lstrip().upper().startswith("DELETE")]
    assert deletes and len(caps) >= 3
    assert not any("NOT IN (SELECT" in s.upper() for s in seen)
