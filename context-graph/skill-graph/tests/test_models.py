import pytest
from skill_graph.models import Skill, SkillValidationError


# ------------------------------------------------------------------
# Name validation
# ------------------------------------------------------------------


class TestNameValidation:
    def test_valid_simple_name(self):
        s = Skill(name="pdf-processing", description="does stuff", content="body")
        assert s.name == "pdf-processing"

    def test_valid_single_char(self):
        s = Skill(name="a", description="does stuff", content="body")
        assert s.name == "a"

    def test_valid_digits_and_hyphens(self):
        s = Skill(name="my-skill-2", description="does stuff", content="body")
        assert s.name == "my-skill-2"

    def test_valid_all_digits(self):
        s = Skill(name="123", description="does stuff", content="body")
        assert s.name == "123"

    def test_max_length_name(self):
        name = "a" * 64
        s = Skill(name=name, description="does stuff", content="body")
        assert s.name == name

    def test_empty_name_raises(self):
        with pytest.raises(SkillValidationError, match="1-64 characters"):
            Skill(name="", description="does stuff", content="body")

    def test_name_too_long_raises(self):
        with pytest.raises(SkillValidationError, match="1-64 characters"):
            Skill(name="a" * 65, description="does stuff", content="body")

    def test_uppercase_raises(self):
        with pytest.raises(SkillValidationError, match="lowercase"):
            Skill(name="PDF-Processing", description="does stuff", content="body")

    def test_leading_hyphen_raises(self):
        with pytest.raises(SkillValidationError, match="must not start or end"):
            Skill(name="-pdf", description="does stuff", content="body")

    def test_trailing_hyphen_raises(self):
        with pytest.raises(SkillValidationError, match="must not start or end"):
            Skill(name="pdf-", description="does stuff", content="body")

    def test_consecutive_hyphens_raises(self):
        with pytest.raises(SkillValidationError, match="consecutive hyphens"):
            Skill(name="pdf--processing", description="does stuff", content="body")

    def test_spaces_raise(self):
        with pytest.raises(SkillValidationError, match="lowercase"):
            Skill(name="pdf processing", description="does stuff", content="body")

    def test_underscores_raise(self):
        with pytest.raises(SkillValidationError, match="lowercase"):
            Skill(name="pdf_processing", description="does stuff", content="body")


# ------------------------------------------------------------------
# Description validation
# ------------------------------------------------------------------


class TestDescriptionValidation:
    def test_valid_description(self):
        s = Skill(name="s1", description="Processes PDFs.", content="body")
        assert s.description == "Processes PDFs."

    def test_empty_description_raises(self):
        with pytest.raises(SkillValidationError, match="1-1024 characters"):
            Skill(name="s1", description="", content="body")

    def test_description_too_long_raises(self):
        with pytest.raises(SkillValidationError, match="1-1024 characters"):
            Skill(name="s1", description="x" * 1025, content="body")

    def test_max_length_description(self):
        desc = "x" * 1024
        s = Skill(name="s1", description=desc, content="body")
        assert s.description == desc


# ------------------------------------------------------------------
# Compatibility validation
# ------------------------------------------------------------------


class TestCompatibilityValidation:
    def test_none_is_valid(self):
        s = Skill(
            name="s1", description="does stuff", content="body", compatibility=None
        )
        assert s.compatibility is None

    def test_valid_compatibility(self):
        s = Skill(
            name="s1",
            description="does stuff",
            content="body",
            compatibility="Requires Python 3.10+",
        )
        assert s.compatibility == "Requires Python 3.10+"

    def test_empty_compatibility_raises(self):
        with pytest.raises(SkillValidationError, match="1-500 characters"):
            Skill(name="s1", description="does stuff", content="body", compatibility="")

    def test_compatibility_too_long_raises(self):
        with pytest.raises(SkillValidationError, match="1-500 characters"):
            Skill(
                name="s1",
                description="does stuff",
                content="body",
                compatibility="x" * 501,
            )

    def test_max_length_compatibility(self):
        compat = "x" * 500
        s = Skill(
            name="s1", description="does stuff", content="body", compatibility=compat
        )
        assert s.compatibility == compat


# ------------------------------------------------------------------
# Optional spec fields defaults
# ------------------------------------------------------------------


class TestOptionalFieldDefaults:
    def test_license_default_none(self):
        s = Skill(name="s1", description="does stuff", content="body")
        assert s.license is None

    def test_metadata_default_empty(self):
        s = Skill(name="s1", description="does stuff", content="body")
        assert s.metadata == {}

    def test_allowed_tools_default_empty(self):
        s = Skill(name="s1", description="does stuff", content="body")
        assert s.allowed_tools == []

    def test_tags_default_empty(self):
        s = Skill(name="s1", description="does stuff", content="body")
        assert s.tags == []


# ------------------------------------------------------------------
# Full construction with all fields
# ------------------------------------------------------------------


class TestFullConstruction:
    def test_all_fields(self):
        s = Skill(
            name="pdf-processing",
            description="Extract PDF text and fill forms.",
            content="# Instructions\nUse pdfplumber.",
            license="Apache-2.0",
            compatibility="Requires Python 3.10+",
            metadata={"author": "example-org", "version": "1.0"},
            allowed_tools=["Bash(git:*)", "Read"],
            tags=["pdf", "extraction"],
        )
        assert s.name == "pdf-processing"
        assert s.license == "Apache-2.0"
        assert s.compatibility == "Requires Python 3.10+"
        assert s.metadata == {"author": "example-org", "version": "1.0"}
        assert s.allowed_tools == ["Bash(git:*)", "Read"]
        assert s.tags == ["pdf", "extraction"]
        assert s.created_at is not None
        assert s.updated_at is not None


# ------------------------------------------------------------------
# Timestamps
# ------------------------------------------------------------------


class TestTimestamps:
    def test_auto_timestamps(self):
        s = Skill(name="s1", description="does stuff", content="body")
        assert s.created_at is not None
        assert s.updated_at is not None
        assert s.created_at == s.updated_at

    def test_explicit_timestamps_preserved(self):
        s = Skill(
            name="s1",
            description="does stuff",
            content="body",
            created_at="2025-01-01T00:00:00",
            updated_at="2025-06-01T00:00:00",
        )
        assert s.created_at == "2025-01-01T00:00:00"
        assert s.updated_at == "2025-06-01T00:00:00"
