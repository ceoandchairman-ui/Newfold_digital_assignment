def upgrade():
    op.execute("ALTER TABLE x ADD COLUMN y TEXT")


def downgrade():
    op.execute("ALTER TABLE x DROP COLUMN y")
