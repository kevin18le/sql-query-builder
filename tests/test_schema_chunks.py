"""Tests for schema chunk formatting in the index builder."""

from src.data_prep.build_schema_index import format_schema_chunks


def _table_info(**overrides):
    base = {
        "table_name": "customers",
        "columns": [
            {"name": "customer_id", "data_type": "integer", "max_length": None,
             "nullable": False, "default": None, "position": 1},
            {"name": "email", "data_type": "text", "max_length": None,
             "nullable": False, "default": None, "position": 2},
        ],
        "primary_keys": ["customer_id"],
        "foreign_keys": [],
        "unique_columns": ["email"],
        "check_constraints": [],
        "indexes": [],
        "row_count": 50,
    }
    base.update(overrides)
    return base


class TestFormatSchemaChunks:
    def test_overview_and_columns_chunks(self):
        chunks = format_schema_chunks([_table_info()])
        types = [c["type"] for c in chunks]
        assert "table_overview" in types
        assert "columns" in types

        overview = next(c for c in chunks if c["type"] == "table_overview")
        assert "Table: customers" in overview["content"]
        assert "Row count: 50" in overview["content"]
        assert "customer_id" in overview["content"]

        columns = next(c for c in chunks if c["type"] == "columns")
        assert "customer_id: integer NOT NULL" in columns["content"]

    def test_foreign_key_chunks(self):
        fk = {"column": "customer_id", "references_table": "customers",
              "references_column": "customer_id", "constraint_name": "fk_cust"}
        chunks = format_schema_chunks([_table_info(table_name="accounts", foreign_keys=[fk])])
        types = [c["type"] for c in chunks]
        assert "foreign_keys" in types
        assert "relationship" in types

        rel = next(c for c in chunks if c["type"] == "relationship")
        assert "accounts.customer_id references customers.customer_id" in rel["content"]

    def test_optional_chunks_omitted_when_empty(self):
        chunks = format_schema_chunks([_table_info()])
        types = [c["type"] for c in chunks]
        assert "foreign_keys" not in types
        assert "constraints_indexes" not in types
        assert "relationship" not in types

    def test_constraints_and_indexes_chunk(self):
        chunks = format_schema_chunks([_table_info(
            check_constraints=[{"name": "chk", "clause": "customer_id > 0"}],
            indexes=[{"name": "idx", "definition": "CREATE INDEX idx ON customers (email)"}],
        )])
        ci = next(c for c in chunks if c["type"] == "constraints_indexes")
        assert "chk: customer_id > 0" in ci["content"]
        assert "CREATE INDEX idx" in ci["content"]
