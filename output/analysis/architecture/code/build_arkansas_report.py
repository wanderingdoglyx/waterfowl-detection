#!/usr/bin/env python
"""
Build the Word report: Arkansas_Specification_And_Protocol.md -> <repo>/Arkansas_Specification_And_Protocol.docx

Same recipe as the flight-altitude report: derive a reference document that pins US Letter with
1 in margins (6.5 in text width, the width the figures are inserted at) and run pandoc with it.

    python output/analysis/architecture/code/build_report.py
"""

import os
import subprocess
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(REPORT_DIR, "..", "..", ".."))

SRC = os.path.join(REPORT_DIR, "Arkansas_Specification_And_Protocol.md")
DST = os.path.join(REPO, "Arkansas_Specification_And_Protocol.docx")
PAGE = ('<w:pgSz w:w="12240" w:h="15840" />'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0" />')


def reference_doc(path: str) -> None:
    default = path + ".default"
    subprocess.run(["pandoc", "-o", default, "--print-default-data-file", "reference.docx"], check=True)
    with zipfile.ZipFile(default) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = data.decode().replace("<w:sectPr>", "<w:sectPr>" + PAGE, 1).encode()
            zout.writestr(item, data)
    os.remove(default)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "reference.docx")
        reference_doc(ref)
        subprocess.run(["pandoc", SRC, "-o", DST, "--reference-doc", ref,
                        "--resource-path", REPORT_DIR], check=True)
    print("wrote", DST)


if __name__ == "__main__":
    main()
