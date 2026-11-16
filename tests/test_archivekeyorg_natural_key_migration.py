"""0106/0107: a paid mark is keyed by (org, question), so settles survive an exhausted id sequence."""
import asyncio

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.exc import DBAPIError

from treg import archive, audit
from treg.application.call.settle import DeferredSettle, close_deferred
from treg.config import get_settings
from treg.domain import money
from treg.infra import db
from treg.infra.db import session_maker
from treg.maintenance import _alembic_config
from tests.test_alembic_baseline import _drop_everything, _insert_org_at_revision, _upgrade_to

INT4_MAX = 2_147_483_647

# The mark statement code older than 0107 sends: it assigns `id` from the serial sequence.
LEGACY_MARK = sa.text("""
    INSERT INTO archivekeyorg (id, org_id, key_hash, first_call_at, last_call_at, calls)
    VALUES (COALESCE((SELECT id FROM archivekeyorg WHERE org_id = :org AND key_hash = :key),
                     nextval(pg_get_serial_sequence('archivekeyorg', 'id'))),
            :org, :key, now(), now(), 1)
    ON CONFLICT (org_id, key_hash) DO UPDATE SET calls = archivekeyorg.calls + excluded.calls
""")
LEGACY_PROBE = sa.text("SELECT id FROM archivekeyorg WHERE org_id = :org AND key_hash = :key LIMIT 1")


@pytest.fixture
async def at_0105(monkeypatch):
    monkeypatch.setattr(get_settings(), "platform_margin", 0)
    await audit.drain()
    await _drop_everything()
    try:
        await _upgrade_to("0105")
        yield db._engine.dialect.name
    finally:
        await _drop_everything()
        await db.reset_db()


async def _org_with_marks(*keys: str) -> int:
    async with session_maker() as s:
        org_id = await _insert_org_at_revision(s, name="marks", slug="marks")
        for key in keys:
            await s.execute(sa.text(
                "INSERT INTO archivekeyorg (org_id, key_hash, first_call_at, last_call_at, calls) "
                "VALUES (:org, :key, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)"),
                {"org": org_id, "key": key})
        await s.commit()
    return org_id


async def _scalar(sql: str, **params):
    async with db._engine.connect() as connection:
        return (await connection.execute(sa.text(sql), params)).scalar()


async def _primary_key() -> list[str]:
    async with db._engine.connect() as connection:
        return await connection.run_sync(
            lambda c: sa.inspect(c).get_pk_constraint("archivekeyorg")["constrained_columns"])


async def _settle(org_id: int, keys: list[str]) -> int:
    async with session_maker() as s:
        items = []
        for i, key in enumerate(keys):
            call = await money.reserve(s, org_id, "test.archive-mark", 100, call_id=f"mark-{key}-{i}")
            items.append(DeferredSettle(call, True, 70, (org_id, key), "", {}, payer_org_id=org_id))
    return await close_deferred(items, charge=True)


async def _assert_billed_once(org_id: int, calls: int) -> None:
    async with session_maker() as s:
        assert await money.balance_of(s, org_id) == 10_000 - calls * 70
        assert await money.open_holds_of(s, org_id) == []
        settled = [e for e in await money.entries_of(s, org_id) if e.kind == "settle"]
        assert len(settled) == len({e.call_id for e in settled}) == calls


async def test_first_time_settles_bill_once_after_the_id_sequence_is_exhausted(at_0105):
    org_id = await _org_with_marks("old-a", "old-b")
    if at_0105 == "postgresql":
        await _scalar(f"SELECT setval(pg_get_serial_sequence('archivekeyorg', 'id'), {INT4_MAX})")
        async with db._engine.connect() as connection:
            with pytest.raises(DBAPIError, match="reached maximum value"):
                await connection.execute(LEGACY_MARK, {"org": org_id, "key": "refused"})
        heap = await _scalar("SELECT relfilenode FROM pg_class WHERE relname = 'archivekeyorg'")

    await _upgrade_to("head")

    assert await _primary_key() == ["org_id", "key_hash"]
    if at_0105 == "postgresql":
        assert await _scalar("SELECT relfilenode FROM pg_class WHERE relname = 'archivekeyorg'") == heap
        assert await _scalar("SELECT count(*) FROM pg_constraint WHERE conname = 'uq_archive_key_org'") == 0
    async with session_maker() as s:
        await money.grant(s, org_id, amount_micro=10_000)
        await s.commit()
    # A routed call: three children, every question new to the team, marked in one transaction.
    assert await _settle(org_id, ["new-1", "new-2", "new-3"]) == 210
    # A direct call: one first-time question, and one repeat of a question marked before 0107.
    assert await _settle(org_id, ["new-4"]) == 70
    assert await _settle(org_id, ["old-a"]) == 70
    await _assert_billed_once(org_id, 5)
    async with db._engine.connect() as connection:
        marks = dict((await connection.execute(sa.text(
            "SELECT key_hash, calls FROM archivekeyorg WHERE org_id = :org"), {"org": org_id})).all())
    assert marks == {"old-a": 2, "old-b": 1, "new-1": 1, "new-2": 1, "new-3": 1, "new-4": 1}


async def test_code_older_than_0107_still_marks_during_a_rolling_deploy(at_0105):
    if at_0105 != "postgresql":
        pytest.skip("the legacy statement is the PostgreSQL one")
    org_id = await _org_with_marks("old")
    await _upgrade_to("head")
    async with session_maker() as s:
        await archive.note_org_uses_in_transaction(s, [(org_id, "new")])
        await s.commit()
    async with db._engine.begin() as connection:
        for key in ("old", "new", "legacy-new"):
            await connection.execute(LEGACY_MARK, {"org": org_id, "key": key})
        for key in ("old", "new", "legacy-new"):
            assert (await connection.execute(LEGACY_PROBE, {"org": org_id, "key": key})).first() is not None
    async with db._engine.connect() as connection:
        marks = dict((await connection.execute(sa.text(
            "SELECT key_hash, calls FROM archivekeyorg WHERE org_id = :org"), {"org": org_id})).all())
    assert marks == {"old": 2, "new": 2, "legacy-new": 1}


async def test_downgrade_numbers_new_marks_and_restores_the_serial_key(at_0105):
    org_id = await _org_with_marks("old")
    await _upgrade_to("head")
    async with session_maker() as s:
        await archive.note_org_uses_in_transaction(s, [(org_id, "new-1"), (org_id, "new-2")])
        await s.commit()
    await db._engine.dispose()
    await asyncio.to_thread(command.downgrade, _alembic_config(), "0105")

    assert await _primary_key() == ["id"]
    async with db._engine.connect() as connection:
        ids = (await connection.execute(sa.text("SELECT id FROM archivekeyorg"))).scalars().all()
    assert len(ids) == len(set(ids)) == 3 and None not in ids
    await _upgrade_to("head")
    assert await _primary_key() == ["org_id", "key_hash"]
