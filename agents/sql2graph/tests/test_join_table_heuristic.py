"""Unit tests for DatabaseAnalyzer.is_join_table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.analyzer import DatabaseAnalyzer
from database.models import ColumnInfo, ForeignKeyInfo, TableInfo, TableType


class _Analyzer(DatabaseAnalyzer):
    """Minimal concrete analyzer so the abstract base can be instantiated."""

    def __init__(self):
        pass

    def _get_database_type(self):
        return "test"

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_tables(self):
        return []

    def get_table_schema(self, table_name):
        return []

    def get_foreign_keys(self, table_name):
        return []

    def get_table_data(self, table_name, limit=100):
        return []

    def get_table_row_count(self, table_name):
        return 0

    def get_indexes(self, table_name):
        return []

    def is_view(self, table_name):
        return False


def _table(name, columns, fks, pks):
    fk_names = {fk[0] for fk in fks}
    return TableInfo(
        name=name,
        table_type=TableType.ENTITY,
        columns=[
            ColumnInfo(
                name=c,
                data_type="int",
                is_nullable=False,
                is_primary_key=c in pks,
                is_foreign_key=c in fk_names,
            )
            for c in columns
        ],
        foreign_keys=[ForeignKeyInfo(column_name=c, referenced_table=t, referenced_column=f"{t}_id") for c, t in fks],
        row_count=0,
        primary_keys=list(pks),
        indexes=[],
    )


ANALYZER = _Analyzer()


def test_composite_fk_primary_key_is_join_table():
    film_actor = _table(
        "film_actor",
        ["actor_id", "film_id", "last_update"],
        [("actor_id", "actor"), ("film_id", "film")],
        ["actor_id", "film_id"],
    )
    assert ANALYZER.is_join_table(film_actor)


def test_bare_id_surrogate_key_is_still_join_table():
    user_role = _table(
        "user_role",
        ["id", "user_id", "role_id"],
        [("user_id", "user"), ("role_id", "role")],
        ["id"],
    )
    assert ANALYZER.is_join_table(user_role)


def test_own_surrogate_key_makes_entity_even_with_two_fks():
    # Sakila: store(store_id, manager_staff_id, address_id, last_update)
    store = _table(
        "store",
        ["store_id", "manager_staff_id", "address_id", "last_update"],
        [("manager_staff_id", "staff"), ("address_id", "address")],
        ["store_id"],
    )
    assert not ANALYZER.is_join_table(store)

    # Sakila: inventory(inventory_id, film_id, store_id, last_update)
    inventory = _table(
        "inventory",
        ["inventory_id", "film_id", "store_id", "last_update"],
        [("film_id", "film"), ("store_id", "store")],
        ["inventory_id"],
    )
    assert not ANALYZER.is_join_table(inventory)


def test_single_fk_is_never_join_table():
    city = _table("city", ["city_id", "city", "country_id"], [("country_id", "country")], ["city_id"])
    assert not ANALYZER.is_join_table(city)
