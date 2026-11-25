def upgrade():
    op.execute("TRUNCATE audit_events")


def downgrade():
    pass
