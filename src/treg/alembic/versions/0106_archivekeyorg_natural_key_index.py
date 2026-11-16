"""archivekeyorg: build the unique index that 0107 turns into the primary key

Revision ID: 0106
Revises: 0105
Create Date: 2026-11-16

0107 makes (org_id, key_hash) the primary key of `archivekeyorg`, so a first paid mark no longer
takes a number from the table's 32-bit serial sequence inside the settle transaction. PostgreSQL
can attach an existing unique index as a primary key without a scan (`ADD PRIMARY KEY USING
INDEX`), but not an index a constraint already owns, and `uq_archive_key_org` is a constraint's
index. This revision builds a second unique index on the same columns for 0107 to attach.

PostgreSQL builds it with CREATE UNIQUE INDEX CONCURRENTLY outside a transaction, so settles keep
writing marks while it builds. On a large table that takes longer than the 120s statement timeout
env.py sets for every migration statement, so this statement alone runs without one, and both
settings are restored after (the pattern of 0072). An index of that name left INVALID by an
interrupted build is dropped and rebuilt; a valid one is left alone. SQLite: nothing to do, 0107
rebuilds the table there.

Rollback floor (pro forma, like 0072): the expand-safety linter counts the autocommit block and
the invalid-index drop as non-additive. Older code ignores the extra index.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0106"
down_revision: str | Sequence[str] | None = "0105"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
contract = True  # pro forma, see the rollback floor note: the operation adds an index

NAME = "ix_archivekeyorg_org_key_unique"

_STATE = sa.text("""
    SELECT ix.indisvalid AS valid,
           t.relname = 'archivekeyorg' AND ix.indisunique AND ix.indnatts = 2
           AND ix.indkey[0] = (SELECT attnum FROM pg_attribute
                               WHERE attrelid = t.oid AND attname = 'org_id')
           AND ix.indkey[1] = (SELECT attnum FROM pg_attribute
                               WHERE attrelid = t.oid AND attname = 'key_hash')
           AND ix.indexprs IS NULL AND ix.indpred IS NULL AND am.amname = 'btree' AS expected
    FROM pg_class i
    JOIN pg_namespace n ON n.oid = i.relnamespace
    JOIN pg_index ix ON ix.indexrelid = i.oid
    JOIN pg_class t ON t.oid = ix.indrelid
    JOIN pg_am am ON am.oid = i.relam
    WHERE n.nspname = current_schema() AND i.relname = :name
""")

_PRIMARY_KEY_IS_NATURAL = sa.text("""
    SELECT array_agg(a.attname::text ORDER BY k.ord) = ARRAY['org_id', 'key_hash']
    FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord)
    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
    WHERE n.nspname = current_schema() AND t.relname = 'archivekeyorg' AND c.contype = 'p'
""")


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    if bind.execute(_PRIMARY_KEY_IS_NATURAL).scalar():
        return
    # CONCURRENTLY cannot run inside a transaction; alembic opens one by default.
    with op.get_context().autocommit_block():
        bind = op.get_bind()
        state = bind.execute(_STATE, {"name": NAME}).mappings().one_or_none()
        if state is not None and state["expected"] is not True:
            raise RuntimeError(
                f"Refusing index migration: {NAME} exists but is not unique archivekeyorg(org_id, key_hash)")
        if state is not None and state["valid"]:
            return
        settings = {name: bind.execute(sa.text(f"SHOW {name}")).scalar_one()
                    for name in ("lock_timeout", "statement_timeout")}
        try:
            bind.execute(sa.text("SET lock_timeout = '180s'"))
            bind.execute(sa.text("SET statement_timeout = 0"))   # this build only; restored below
            if state is not None:   # left INVALID by an interrupted concurrent build
                op.drop_index(NAME, table_name="archivekeyorg", postgresql_concurrently=True,
                              if_exists=True)
            bind.execute(sa.text(
                f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS {NAME} "
                "ON archivekeyorg (org_id, key_hash)"))
        finally:
            for name, value in settings.items():
                bind.execute(sa.text("SELECT set_config(:name, :value, false)"),
                             {"name": name, "value": value})


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    with op.get_context().autocommit_block():
        op.drop_index(NAME, table_name="archivekeyorg", postgresql_concurrently=True, if_exists=True)
