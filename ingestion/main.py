import os

import docx
import pandas
import pdfplumber
import requests
from bs4 import BeautifulSoup


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_pdf(pdf_path):
    # NOTE:
    #   * pypdf -> slow, high quality, custom permissive license
    #   * pdfplumber -> fast, some errors, MIT license <--- we'll use this one
    #   * pymupdf -> fast, AGPL license
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() for page in pdf.pages]


def parse_docx(docx_path):
    doc = docx.Document(docx_path)
    # NOTE: python-docx does NOT support extracting pages directly -> we'll extract paragraphs instead.
    return [para.text for para in doc.paragraphs if para.text.strip()]


def parse_xls(xls_path):
    # NOTE: engines=['openpyxl', 'xlrd']
    dfs = pandas.read_excel(xls_path, sheet_name=None, engine="openpyxl")
    return [(sheet_name, df) for sheet_name, df in dfs.items()]


def parse_url(url):
    # TODO(gitbuda): Figure out what's the best to put under the user agent
    try:
        response = requests.get(url, headers={"User-Agent": "Memgraph AI Toolkit"})
        response.raise_for_status()
        text_xml = BeautifulSoup(response.text, "html.parser")
        return text_xml.get_text(separator="\n", strip=True)
    except Exception as e:
        raise ValueError(f"Error while fetching or parsing from URL: {url} - {e}")


if __name__ == "__main__":
    pypdf_samples_dir = os.path.join(SCRIPT_DIR, "sample-data", "pdf", "sample-files")
    test_pdfs = [
        os.path.join(
            pypdf_samples_dir, "011-google-doc-document", "google-doc-document.pdf"
        )
    ]
    pages_per_document = [parse_pdf(pdf_path) for pdf_path in test_pdfs]
    print(pages_per_document)

    docx_samples_dir = os.path.join(SCRIPT_DIR, "sample-data", "doc")
    test_docxs = [os.path.join(docx_samples_dir, "sample3.docx")]
    paragraphs_per_document = [parse_docx(docx_path) for docx_path in test_docxs]
    print(paragraphs_per_document)

    xls_samples_dir = os.path.join(SCRIPT_DIR, "sample-data", "xls")
    test_xls = [os.path.join(xls_samples_dir, "financial-sample.xlsx")]
    sheets_per_document = [parse_xls(xls_path) for xls_path in test_xls]
    print(sheets_per_document)

    url_samples = [
        "https://memgraph.com/docs/ai-ecosystem/graph-rag",
    ]
    page_per_url = [parse_url(url) for url in url_samples]
    print(page_per_url)
