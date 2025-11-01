import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


pypdf_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "pdf", "sample-files")
docx_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "doc")
xls_samples_dir = os.path.join(SCRIPT_DIR, "..", "sample-data", "xls")
SOURCES = [
    os.path.join(
        pypdf_samples_dir, "011-google-doc-document", "google-doc-document.pdf"
    ),
    os.path.join(docx_samples_dir, "sample3.docx"),
    # os.path.join(xls_samples_dir, "financial-sample.xlsx"),
    # "https://memgraph.com/docs/ai-ecosystem/graph-rag",
]
