from typing import List, Optional
from datetime import datetime, timezone

from memgraph_toolbox.api.memgraph import Memgraph

from .models import Skill


class SkillGraph:
    """Persist and retrieve AI skill files from Memgraph.

    Stores skills as (:Skill) nodes with optional (:Tag) relationships
    and (:Skill)-[:DEPENDS_ON]->(:Skill) dependency edges.
    """

    def __init__(self, memgraph: Optional[Memgraph] = None, **kwargs):
        """Initialize SkillGraph.

        Args:
            memgraph: An existing Memgraph client instance. If not provided,
                      a new one is created using kwargs / environment variables.
            **kwargs: Forwarded to Memgraph() when memgraph is None.
        """
        self._db = memgraph or Memgraph(**kwargs)

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create constraints and indexes required for skill storage."""
        self._db.query("CREATE CONSTRAINT ON (s:Skill) ASSERT s.name IS UNIQUE;")
        self._db.query("CREATE INDEX ON :Skill(name);")
        self._db.query("CREATE INDEX ON :Tag(name);")

    def drop(self) -> None:
        """Remove all skill-related constraints and indexes."""
        self._db.query("DROP CONSTRAINT ON (s:Skill) ASSERT s.name IS UNIQUE;")
        self._db.query("DROP INDEX ON :Skill(name);")
        self._db.query("DROP INDEX ON :Tag(name);")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_skill(self, skill: Skill) -> Skill:
        """Persist a skill to Memgraph.

        Creates the :Skill node, links it to :Tag nodes (MERGE-ed),
        and returns the stored skill.
        """
        self._db.query(
            """
            CREATE (s:Skill {
                name: $name,
                description: $description,
                content: $content,
                created_at: $created_at,
                updated_at: $updated_at
            })
            """,
            params={
                "name": skill.name,
                "description": skill.description,
                "content": skill.content,
                "created_at": skill.created_at,
                "updated_at": skill.updated_at,
            },
        )

        if skill.tags:
            self._db.query(
                """
                MATCH (s:Skill {name: $name})
                UNWIND $tags AS tag_name
                MERGE (t:Tag {name: tag_name})
                MERGE (s)-[:HAS_TAG]->(t)
                """,
                params={"name": skill.name, "tags": skill.tags},
            )

        return skill

    def get_skill(self, name: str) -> Optional[Skill]:
        """Retrieve a single skill by name, including its tags."""
        rows = self._db.query(
            """
            MATCH (s:Skill {name: $name})
            OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag)
            RETURN s.name AS name,
                   s.description AS description,
                   s.content AS content,
                   s.created_at AS created_at,
                   s.updated_at AS updated_at,
                   collect(t.name) AS tags
            """,
            params={"name": name},
        )

        if not rows:
            return None

        row = rows[0]
        return Skill(
            name=row["name"],
            description=row["description"],
            content=row["content"],
            tags=[t for t in row["tags"] if t is not None],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_skill(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Skill]:
        """Update an existing skill. Only provided fields are changed."""
        sets: list[str] = []
        params: dict = {
            "name": name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if description is not None:
            sets.append("s.description = $description")
            params["description"] = description
        if content is not None:
            sets.append("s.content = $content")
            params["content"] = content

        sets.append("s.updated_at = $updated_at")

        self._db.query(
            f"MATCH (s:Skill {{name: $name}}) SET {', '.join(sets)}",
            params=params,
        )

        if tags is not None:
            # Remove old tag relationships and set new ones
            self._db.query(
                "MATCH (s:Skill {name: $name})-[r:HAS_TAG]->() DELETE r",
                params={"name": name},
            )
            if tags:
                self._db.query(
                    """
                    MATCH (s:Skill {name: $name})
                    UNWIND $tags AS tag_name
                    MERGE (t:Tag {name: tag_name})
                    MERGE (s)-[:HAS_TAG]->(t)
                    """,
                    params={"name": name, "tags": tags},
                )

        return self.get_skill(name)

    def delete_skill(self, name: str) -> bool:
        """Delete a skill and its tag relationships. Returns True if deleted."""
        rows = self._db.query(
            """
            MATCH (s:Skill {name: $name})
            DETACH DELETE s
            RETURN count(s) AS deleted
            """,
            params={"name": name},
        )
        return bool(rows and rows[0].get("deleted", 0) > 0)

    # ------------------------------------------------------------------
    # Query / Search
    # ------------------------------------------------------------------

    def list_skills(self) -> List[Skill]:
        """Return all stored skills."""
        rows = self._db.query(
            """
            MATCH (s:Skill)
            OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag)
            RETURN s.name AS name,
                   s.description AS description,
                   s.content AS content,
                   s.created_at AS created_at,
                   s.updated_at AS updated_at,
                   collect(t.name) AS tags
            ORDER BY s.name
            """
        )
        return [self._row_to_skill(r) for r in rows]

    def search_by_tags(self, tags: List[str]) -> List[Skill]:
        """Find skills that have *all* of the given tags."""
        rows = self._db.query(
            """
            MATCH (s:Skill)
            WHERE ALL(tag IN $tags WHERE (s)-[:HAS_TAG]->(:Tag {name: tag}))
            OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag)
            RETURN s.name AS name,
                   s.description AS description,
                   s.content AS content,
                   s.created_at AS created_at,
                   s.updated_at AS updated_at,
                   collect(t.name) AS tags
            ORDER BY s.name
            """,
            params={"tags": tags},
        )
        return [self._row_to_skill(r) for r in rows]

    def search_by_name(self, pattern: str) -> List[Skill]:
        """Find skills whose name contains the given substring (case-insensitive)."""
        rows = self._db.query(
            """
            MATCH (s:Skill)
            WHERE toLower(s.name) CONTAINS toLower($pattern)
            OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag)
            RETURN s.name AS name,
                   s.description AS description,
                   s.content AS content,
                   s.created_at AS created_at,
                   s.updated_at AS updated_at,
                   collect(t.name) AS tags
            ORDER BY s.name
            """,
            params={"pattern": pattern},
        )
        return [self._row_to_skill(r) for r in rows]

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------

    def add_dependency(self, skill_name: str, depends_on: str) -> None:
        """Record that *skill_name* depends on *depends_on*."""
        self._db.query(
            """
            MATCH (a:Skill {name: $skill_name}), (b:Skill {name: $depends_on})
            MERGE (a)-[:DEPENDS_ON]->(b)
            """,
            params={"skill_name": skill_name, "depends_on": depends_on},
        )

    def remove_dependency(self, skill_name: str, depends_on: str) -> None:
        """Remove a dependency edge between two skills."""
        self._db.query(
            """
            MATCH (a:Skill {name: $skill_name})-[r:DEPENDS_ON]->(b:Skill {name: $depends_on})
            DELETE r
            """,
            params={"skill_name": skill_name, "depends_on": depends_on},
        )

    def get_dependencies(self, skill_name: str) -> List[Skill]:
        """Return skills that *skill_name* depends on."""
        rows = self._db.query(
            """
            MATCH (a:Skill {name: $skill_name})-[:DEPENDS_ON]->(s:Skill)
            OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag)
            RETURN s.name AS name,
                   s.description AS description,
                   s.content AS content,
                   s.created_at AS created_at,
                   s.updated_at AS updated_at,
                   collect(t.name) AS tags
            ORDER BY s.name
            """,
            params={"skill_name": skill_name},
        )
        return [self._row_to_skill(r) for r in rows]

    def get_dependents(self, skill_name: str) -> List[Skill]:
        """Return skills that depend on *skill_name*."""
        rows = self._db.query(
            """
            MATCH (s:Skill)-[:DEPENDS_ON]->(b:Skill {name: $skill_name})
            OPTIONAL MATCH (s)-[:HAS_TAG]->(t:Tag)
            RETURN s.name AS name,
                   s.description AS description,
                   s.content AS content,
                   s.created_at AS created_at,
                   s.updated_at AS updated_at,
                   collect(t.name) AS tags
            ORDER BY s.name
            """,
            params={"skill_name": skill_name},
        )
        return [self._row_to_skill(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_skill(row: dict) -> Skill:
        return Skill(
            name=row["name"],
            description=row["description"],
            content=row["content"],
            tags=[t for t in row["tags"] if t is not None],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
