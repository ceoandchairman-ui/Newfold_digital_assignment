def upgrade():
    # -- contract-ok: column unused since release 42
    op.execute("ALTER TABLE customers DROP COLUMN legacy")
