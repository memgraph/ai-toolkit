#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/sample-data/pdf
if [ ! -d $SCRIPT_DIR/sample-data/pdf/sample-files ]; then
    git clone https://github.com/py-pdf/sample-files.git $SCRIPT_DIR/sample-data/pdf/sample-files
fi

mkdir -p $SCRIPT_DIR/sample-data/doc
if [ ! -f $SCRIPT_DIR/sample-data/doc/sample3.docx ]; then
    curl -X GET https://www2.hu-berlin.de/stadtlabor/wp-content/uploads/2021/12/sample3.docx -o $SCRIPT_DIR/sample-data/doc/sample3.docx
fi

mkdir -p $SCRIPT_DIR/sample-data/xls
if [ ! -f $SCRIPT_DIR/sample-data/xls/financial-sample.xlsx ]; then
    curl -X GET https://go.microsoft.com/fwlink/?LinkID=521962 -L -o $SCRIPT_DIR/sample-data/xls/financial-sample.xlsx
fi
