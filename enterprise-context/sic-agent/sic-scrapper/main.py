"""
OSHA SIC (Standard Industrial Classification) Manual Scraper

This script scrapes the hierarchical SIC code structure from OSHA's website
and generates Cypher queries to import the data into Memgraph as a tree structure.

Hierarchy:
- Division (A-J)
  - Major Group (2-digit codes)
    - Industry Group (3-digit codes)
      - Industry (4-digit codes)
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


BASE_URL = "https://www.osha.gov"
SIC_MANUAL_URL = f"{BASE_URL}/data/sic-manual"

# Respectful scraping delay (seconds)
REQUEST_DELAY = 0.5


@dataclass
class Industry:
    """4-digit SIC code - leaf node"""

    code: str
    name: str
    description: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class IndustryGroup:
    """3-digit SIC code"""

    code: str
    name: str
    industries: list[Industry] = field(default_factory=list)


@dataclass
class MajorGroup:
    """2-digit SIC code"""

    code: str
    name: str
    description: str = ""
    url: str = ""
    industry_groups: list[IndustryGroup] = field(default_factory=list)


@dataclass
class Division:
    """Top-level division (A-J)"""

    code: str  # A, B, C, etc.
    name: str
    url: str = ""
    major_groups: list[MajorGroup] = field(default_factory=list)


class SICScraper:
    """Scraper for OSHA SIC Manual"""

    def __init__(self, delay: float = REQUEST_DELAY):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) SIC-Research-Bot/1.0"
            }
        )

    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a URL with rate limiting"""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def scrape_main_page(self) -> list[Division]:
        """Scrape the main SIC manual page to get all divisions and major groups"""
        print(f"Fetching main SIC manual page: {SIC_MANUAL_URL}")
        soup = self._fetch(SIC_MANUAL_URL)
        if not soup:
            return []

        divisions = []
        current_division = None

        # Find all links in the page - they appear in order (division, then its major groups)
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)

            # Check for division links
            if "/division-" in href.lower():
                division_match = re.search(
                    r"Division\s+([A-J]):\s*(.+)", text, re.IGNORECASE
                )
                if division_match:
                    code = division_match.group(1).upper()
                    name = division_match.group(2).strip()
                    current_division = Division(
                        code=code,
                        name=name,
                        url=BASE_URL + href if href.startswith("/") else href,
                    )
                    divisions.append(current_division)
                    print(f"  Found Division {code}: {name}")

            # Check for major group links
            elif "/major-group-" in href.lower() and current_division:
                mg_match = re.search(
                    r"Major\s+Group\s+(\d+):\s*(.+)", text, re.IGNORECASE
                )
                if mg_match:
                    code = mg_match.group(1).zfill(2)
                    name = mg_match.group(2).strip()
                    major_group = MajorGroup(
                        code=code,
                        name=name,
                        url=BASE_URL + href if href.startswith("/") else href,
                    )
                    current_division.major_groups.append(major_group)
                    print(f"    Found Major Group {code}: {name}")

        return divisions

    def scrape_major_group(self, major_group: MajorGroup) -> None:
        """Scrape a major group page to get industry groups and industries"""
        if not major_group.url:
            return

        print(f"  Scraping Major Group {major_group.code}: {major_group.name}")
        soup = self._fetch(major_group.url)
        if not soup:
            return

        # Use article tag which contains the actual content
        content = soup.find("article") or soup.find("main") or soup

        # Get the description (first paragraph usually)
        description_parts = []
        for p in content.find_all("p"):
            p_text = p.get_text(strip=True)
            if p_text and not p_text.startswith("Industry Group"):
                description_parts.append(p_text)
            if len(description_parts) >= 2:
                break
        major_group.description = " ".join(description_parts)

        # Find industry groups and their industries
        current_industry_group = None
        text_content = content.get_text()

        # Find all industry group headers
        ig_pattern = r"Industry\s+Group\s+(\d{3}):\s*([^\n•]+)"
        for match in re.finditer(ig_pattern, text_content):
            code = match.group(1)
            name = match.group(2).strip()
            current_industry_group = IndustryGroup(code=code, name=name)
            major_group.industry_groups.append(current_industry_group)
            print(f"      Found Industry Group {code}: {name}")

        # Find all industry links (4-digit codes)
        for link in content.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)

            # Match 4-digit SIC codes in links
            if "/sic-manual/" in href:
                code_match = re.search(r"/sic-manual/(\d{4})$", href)
                if code_match:
                    code = code_match.group(1)
                    # Find the matching industry group (first 3 digits)
                    ig_code = code[:3]

                    for ig in major_group.industry_groups:
                        if ig.code == ig_code:
                            industry = Industry(code=code, name=text)
                            ig.industries.append(industry)
                            print(f"        Found Industry {code}: {text}")
                            break
                    else:
                        # Create industry group if not found
                        ig = IndustryGroup(code=ig_code, name="Unknown")
                        ig.industries.append(Industry(code=code, name=text))
                        major_group.industry_groups.append(ig)

    def scrape_industry(self, industry: Industry, major_group_code: str) -> None:
        """Scrape an individual industry page for detailed description"""
        url = f"{BASE_URL}/sic-manual/{industry.code}"
        print(f"          Scraping Industry {industry.code}")

        soup = self._fetch(url)
        if not soup:
            return

        # Use article tag which contains the actual content
        content = soup.find("article") or soup.find("main") or soup

        # Get description
        paragraphs = content.find_all("p")
        descriptions = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if (
                text
                and not text.startswith("Division")
                and "Industry Group" not in text
            ):
                descriptions.append(text)

        industry.description = " ".join(descriptions)

        # Get examples (usually in bullet points)
        for ul in content.find_all("ul"):
            for li in ul.find_all("li"):
                example = li.get_text(strip=True)
                if example:
                    industry.examples.append(example)

    def scrape_all(self, scrape_industries: bool = True) -> list[Division]:
        """Scrape the entire SIC hierarchy"""
        print("Starting SIC Manual scrape...")
        print("=" * 60)

        divisions = self.scrape_main_page()

        for division in divisions:
            print(f"\nProcessing Division {division.code}: {division.name}")
            for major_group in division.major_groups:
                self.scrape_major_group(major_group)

                if scrape_industries:
                    for ig in major_group.industry_groups:
                        for industry in ig.industries:
                            self.scrape_industry(industry, major_group.code)

        print("\n" + "=" * 60)
        print("Scraping complete!")
        return divisions


class CypherExporter:
    """Export SIC data to Cypher queries for Memgraph"""

    @staticmethod
    def escape_string(s: str) -> str:
        """Escape special characters for Cypher strings"""
        return (
            s.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace('"', '\\"')
            .replace("\n", " ")
        )

    def generate_cypher(self, divisions: list[Division]) -> str:
        """Generate Cypher queries to create the SIC tree in Memgraph"""
        queries = []

        # Create constraints and indexes
        queries.append("CREATE INDEX ON :Division(code);")
        queries.append("CREATE INDEX ON :MajorGroup(code);")
        queries.append("CREATE INDEX ON :IndustryGroup(code);")
        queries.append("CREATE INDEX ON :Industry(code);")

        # Create root node
        queries.append(
            "CREATE (:SICManual {name: 'Standard Industrial Classification Manual', source: 'OSHA'});"
        )

        # Create divisions
        for div in divisions:
            name = self.escape_string(div.name)
            queries.append(
                f"CREATE (:Division {{code: '{div.code}', name: '{name}'}});"
            )

        # Link divisions to root
        for div in divisions:
            queries.append(
                f"MATCH (root:SICManual), (d:Division {{code: '{div.code}'}}) "
                f"CREATE (root)-[:HAS_DIVISION]->(d);"
            )

        # Create major groups
        for div in divisions:
            for mg in div.major_groups:
                name = self.escape_string(mg.name)
                desc = self.escape_string(
                    mg.description[:500] if mg.description else ""
                )
                queries.append(
                    f"CREATE (:MajorGroup {{code: '{mg.code}', name: '{name}', description: '{desc}'}});"
                )

        # Link major groups to divisions
        for div in divisions:
            for mg in div.major_groups:
                queries.append(
                    f"MATCH (d:Division {{code: '{div.code}'}}), (mg:MajorGroup {{code: '{mg.code}'}}) "
                    f"CREATE (d)-[:HAS_MAJOR_GROUP]->(mg);"
                )

        # Create industry groups
        for div in divisions:
            for mg in div.major_groups:
                for ig in mg.industry_groups:
                    name = self.escape_string(ig.name)
                    queries.append(
                        f"CREATE (:IndustryGroup {{code: '{ig.code}', name: '{name}'}});"
                    )

        # Link industry groups to major groups
        for div in divisions:
            for mg in div.major_groups:
                for ig in mg.industry_groups:
                    queries.append(
                        f"MATCH (mg:MajorGroup {{code: '{mg.code}'}}), (ig:IndustryGroup {{code: '{ig.code}'}}) "
                        f"CREATE (mg)-[:HAS_INDUSTRY_GROUP]->(ig);"
                    )

        # Create industries
        for div in divisions:
            for mg in div.major_groups:
                for ig in mg.industry_groups:
                    for ind in ig.industries:
                        name = self.escape_string(ind.name)
                        desc = self.escape_string(
                            ind.description[:500] if ind.description else ""
                        )
                        examples = (
                            json.dumps(ind.examples[:10]) if ind.examples else "[]"
                        )
                        queries.append(
                            f"CREATE (:Industry {{code: '{ind.code}', name: '{name}', "
                            f"description: '{desc}', examples: {examples}}});"
                        )

        # Link industries to industry groups
        for div in divisions:
            for mg in div.major_groups:
                for ig in mg.industry_groups:
                    for ind in ig.industries:
                        queries.append(
                            f"MATCH (ig:IndustryGroup {{code: '{ig.code}'}}), (i:Industry {{code: '{ind.code}'}}) "
                            f"CREATE (ig)-[:HAS_INDUSTRY]->(i);"
                        )

        return "\n".join(queries)


def export_to_json(divisions: list[Division], filepath: str) -> None:
    """Export scraped data to JSON format"""
    data = []
    for div in divisions:
        div_dict = {
            "type": "Division",
            "code": div.code,
            "name": div.name,
            "major_groups": [],
        }
        for mg in div.major_groups:
            mg_dict = {
                "type": "MajorGroup",
                "code": mg.code,
                "name": mg.name,
                "description": mg.description,
                "industry_groups": [],
            }
            for ig in mg.industry_groups:
                ig_dict = {
                    "type": "IndustryGroup",
                    "code": ig.code,
                    "name": ig.name,
                    "industries": [],
                }
                for ind in ig.industries:
                    ind_dict = {
                        "type": "Industry",
                        "code": ind.code,
                        "name": ind.name,
                        "description": ind.description,
                        "examples": ind.examples,
                    }
                    ig_dict["industries"].append(ind_dict)
                mg_dict["industry_groups"].append(ig_dict)
            div_dict["major_groups"].append(mg_dict)
        data.append(div_dict)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Exported data to {filepath}")


def print_statistics(divisions: list[Division]) -> None:
    """Print statistics about the scraped data"""
    total_divisions = len(divisions)
    total_major_groups = sum(len(d.major_groups) for d in divisions)
    total_industry_groups = sum(
        len(mg.industry_groups) for d in divisions for mg in d.major_groups
    )
    total_industries = sum(
        len(ig.industries)
        for d in divisions
        for mg in d.major_groups
        for ig in mg.industry_groups
    )

    print("\n" + "=" * 60)
    print("SIC Manual Statistics:")
    print("=" * 60)
    print(f"  Divisions:        {total_divisions}")
    print(f"  Major Groups:     {total_major_groups}")
    print(f"  Industry Groups:  {total_industry_groups}")
    print(f"  Industries:       {total_industries}")
    print(
        f"  Total Nodes:      {total_divisions + total_major_groups + total_industry_groups + total_industries + 1}"
    )
    print("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape OSHA SIC Manual and export to Memgraph"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./output",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--no-industry-details",
        action="store_true",
        help="Skip scraping individual industry pages (faster but less detail)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Delay between requests in seconds (default: {REQUEST_DELAY})",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scrape data
    scraper = SICScraper(delay=args.delay)
    divisions = scraper.scrape_all(scrape_industries=not args.no_industry_details)

    # Print statistics
    print_statistics(divisions)

    # Export to JSON
    json_path = output_dir / "sic_data.json"
    export_to_json(divisions, str(json_path))

    # Export to Cypher (.cypherl format for Memgraph direct loading)
    exporter = CypherExporter()
    cypher = exporter.generate_cypher(divisions)
    cypher_path = output_dir / "sic_import.cypherl"

    with open(cypher_path, "w", encoding="utf-8") as f:
        f.write(cypher)
    print(f"Exported Cypher queries to {cypher_path}")

    print("\n" + "=" * 60)
    print("To import into Memgraph:")
    print("=" * 60)
    print(f"  1. Start Memgraph")
    print(f"  2. Run: mgconsole < {cypher_path}")
    print("  Or use Memgraph Lab to execute the queries")
    print("=" * 60)


if __name__ == "__main__":
    main()
