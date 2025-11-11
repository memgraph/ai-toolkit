import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


pypdf_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "pdf", "sample-files")
docx_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "doc")
xls_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "xls")
RANDOM = [
    os.path.join(
        pypdf_samples_dir, "011-google-doc-document", "google-doc-document.pdf"
    ),
    os.path.join(docx_samples_dir, "sample3.docx"),
    # os.path.join(xls_samples_dir, "financial-sample.xlsx"),
    # "https://memgraph.com/docs/ai-ecosystem/graph-rag",
]
MEMGRAPH_DOCS = [
    "https://memgraph.com/docs/querying/clauses",
    "https://memgraph.com/docs/clustering/high-availability",
]

MEMGRAPH_DOCS_GITHUB_LATEST = [
    "https://github.com/memgraph/documentation/pull/1452/files"
]

MEMGRAPH_DOCS_GITHUB_LATEST_RAW = [
    "https://raw.githubusercontent.com/memgraph/documentation/f6f165649b89efc51fa4153fffc08ff5304ca0c9/pages/database-management/authentication-and-authorization/mlbac-migration-guide.mdx",
    # "https://raw.githubusercontent.com/memgraph/documentation/f6f165649b89efc51fa4153fffc08ff5304ca0c9/pages/database-management/authentication-and-authorization/role-based-access-control.mdx",
    # "https://raw.githubusercontent.com/memgraph/documentation/40ab6644f7113aa5cb86faa48961d2cb2c34f2cc/pages/data-migration/parquet.mdx",
]
