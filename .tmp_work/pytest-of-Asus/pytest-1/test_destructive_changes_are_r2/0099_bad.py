def upgrade():
    op.execute("ALTER TABLE bills RENAME COLUMN status TO state")


def downgrade():
    pass
