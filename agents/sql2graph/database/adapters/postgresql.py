"""PostgreSQL-specific database analyzer implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import psycopg2  # type: ignore[import-not-found]
    import psycopg2.extras  # type: ignore[import-not-found]
    from psycopg2 import sql  # type: ignore[import-not-found]
except ImportError as import_error:  # pragma: no cover - optional dependency
    psycopg2 = None  # type: ignore[assignment]
    sql = None  # type: ignore[assignment]
    _PSYCOPG2_IMPORT_ERROR = import_error
else:
    _PSYCOPG2_IMPORT_ERROR = None

from ..analyzer import DatabaseAnalyzer
from ..models import ColumnInfo, ForeignKeyInfo

logger = logging.getLogger(__name__)


class PostgreSQLAnalyzer(DatabaseAnalyzer):
    """PostgreSQL-specific implementation of DatabaseAnalyzer."""

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 5432,
        schema: str = "public",
    ):
        connection_config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
            "port": port,
            "schema": schema,
        }
        self._schema = schema
        super().__init__(connection_config)

    def _get_database_type(self) -> str:
        return "postgresql"

    def connect(self) -> bool:
        if psycopg2 is None:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support"
            ) from _PSYCOPG2_IMPORT_ERROR

        try:
            connect_config = {
                key: value
                for key, value in self.connection_config.items()
                if key != "schema"
            }
            self.connection = psycopg2.connect(**connect_config)
            logger.info("Successfully connected to PostgreSQL database")
            return True
        except psycopg2.Error as exc:
            logger.error("Error connecting to PostgreSQL: %s", exc)
            self.connection = None
            return False

    def disconnect(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("PostgreSQL connection closed")

    def get_tables(self) -> List[str]:
        connection = self._require_connection()
        schema = self._schema_name()
        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_name
        """

        cursor = connection.cursor()
        cursor.execute(query, (schema,))
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables

    def get_table_schema(self, table_name: str) -> List[ColumnInfo]:
        connection = self._require_connection()
        schema = self._schema_name()

        column_query = """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length,
            numeric_precision,
            numeric_scale
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position
        """

        cursor = connection.cursor()
        cursor.execute(column_query, (schema, table_name))
        column_rows = cursor.fetchall()
        cursor.close()

        primary_keys = set(self._get_primary_key_columns(table_name))
        foreign_keys = self.get_foreign_keys(table_name)
        fk_column_names = {fk.column_name for fk in foreign_keys}

        columns: List[ColumnInfo] = []
        for (
            column_name,
            data_type,
            is_nullable,
            column_default,
            char_max_length,
            numeric_precision,
            numeric_scale,
        ) in column_rows:
            auto_increment = False
            if isinstance(column_default, str):
                auto_increment = column_default.lower().startswith("nextval(")

            max_length = int(char_max_length) if char_max_length is not None else None
            precision = (
                int(numeric_precision) if numeric_precision is not None else None
            )
            scale = int(numeric_scale) if numeric_scale is not None else None

            columns.append(
                ColumnInfo(
                    name=column_name,
                    data_type=data_type,
                    is_nullable=is_nullable == "YES",
                    is_primary_key=column_name in primary_keys,
                    is_foreign_key=column_name in fk_column_names,
                    default_value=column_default,
                    auto_increment=auto_increment,
                    max_length=max_length,
                    precision=precision,
                    scale=scale,
                )
            )

        return columns

    def get_foreign_keys(self, table_name: str) -> List[ForeignKeyInfo]:
        connection = self._require_connection()
        schema = self._schema_name()
        query = """
        SELECT
            kcu.column_name,
            ccu.table_name AS referenced_table,
            ccu.column_name AS referenced_column,
            tc.constraint_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.table_schema = %s
          AND tc.table_name = %s
          AND tc.constraint_type = 'FOREIGN KEY'
        ORDER BY tc.constraint_name, kcu.ordinal_position
        """

        cursor = connection.cursor()
        cursor.execute(query, (schema, table_name))
        foreign_keys = [
            ForeignKeyInfo(
                column_name=row[0],
                referenced_table=row[1],
                referenced_column=row[2],
                constraint_name=row[3],
            )
            for row in cursor.fetchall()
        ]
        cursor.close()
        return foreign_keys

    def get_table_data(
        self, table_name: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        connection = self._require_connection()
        schema = self._schema_name()
        if sql is None or psycopg2 is None:  # pragma: no cover - import guard
            raise ImportError("psycopg2 is required for PostgreSQL support")

        query = sql.SQL("SELECT * FROM {}.{}").format(
            sql.Identifier(schema), sql.Identifier(table_name)
        )

        params: Optional[Tuple[int, ...]] = None
        if limit is not None:
            query = query + sql.SQL(" LIMIT %s")
            params = (limit,)

        cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]

    def get_table_row_count(self, table_name: str) -> int:
        connection = self._require_connection()
        schema = self._schema_name()
        if sql is None:  # pragma: no cover - import guard
            raise ImportError("psycopg2 is required for PostgreSQL support")

        query = sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
            sql.Identifier(schema), sql.Identifier(table_name)
        )

        cursor = connection.cursor()
        cursor.execute(query)
        count = cursor.fetchone()[0]
        cursor.close()
        return int(count)

    def is_view(self, table_name: str) -> bool:
        connection = self._require_connection()
        schema = self._schema_name()
        query = """
        SELECT table_type
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_name = %s
        """

        cursor = connection.cursor()
        cursor.execute(query, (schema, table_name))
        result = cursor.fetchone()
        cursor.close()
        if result:
            return result[0] == "VIEW"
        return False

    def get_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        connection = self._require_connection()
        schema = self._schema_name()
        query = """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = %s
          AND tablename = %s
        ORDER BY indexname
        """

        cursor = connection.cursor()
        cursor.execute(query, (schema, table_name))

        indexes: List[Dict[str, Any]] = []
        for index_name, index_def in cursor.fetchall():
            index_info: Dict[str, Any] = {
                "name": index_name,
                "columns": self._parse_index_columns(index_def),
                "is_unique": index_def.upper().startswith("CREATE UNIQUE"),
                "type": self._parse_index_type(index_def),
                "definition": index_def,
            }
            indexes.append(index_info)

        cursor.close()
        return indexes

    def _schema_name(self) -> str:
        return self.connection_config.get("schema", self._schema or "public")

    def _require_connection(self) -> Any:
        if self.connection is None:
            raise ConnectionError("Not connected to database")
        return self.connection

    def _get_primary_key_columns(self, table_name: str) -> List[str]:
        connection = self._require_connection()
        schema = self._schema_name()
        query = """
        SELECT kcu.column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema = %s
          AND tc.table_name = %s
          AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position
        """

        cursor = connection.cursor()
        cursor.execute(query, (schema, table_name))
        primary_keys = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return primary_keys

    def _parse_index_columns(self, index_def: str) -> List[str]:
        if "(" not in index_def or ")" not in index_def:
            return []

        try:
            columns_part = index_def.split("(", 1)[1].rsplit(")", 1)[0]
        except (IndexError, ValueError):
            return []

        columns = []
        for raw_column in columns_part.split(","):
            column = raw_column.strip().strip('"')
            if column:
                columns.append(column)
        return columns

    def _parse_index_type(self, index_def: str) -> Optional[str]:
        marker = " USING "
        upper_def = index_def.upper()
        if marker not in upper_def:
            return None

        try:
            start_index = upper_def.index(marker) + len(marker)
            postfix = index_def[start_index:]
            index_type = postfix.split(" ", 1)[0].strip()
            return index_type.lower() if index_type else None
        except ValueError:
            return None
