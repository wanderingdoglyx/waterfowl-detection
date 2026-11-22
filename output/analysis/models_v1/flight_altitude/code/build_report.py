"""
Build the Word report: Flight_Altitude_Equation.md  ->  <repo>/Flight_Altitude_Equation.docx

pandoc's default reference document leaves the page size and margins unset, so Word falls back to
its own defaults and full-width images can overflow. This script derives a reference document that
pins US Letter with 1 in margins (6.5 in text width, the width the figures and equation images are
inserted at) and runs pandoc with it.
"""
import os
import subprocess
import tempfile
import zipfile

from common import REPO, REPORT_DIR

SRC = os.path.join(REPORT_DIR, "Flight_Altitude_Equation.md")
DST = os.path.join(REPO, "Flight_Altitude_Equation.docx")
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
        subprocess.run(["pandoc", SRC, "-o", DST, "--reference-doc", ref, "--resource-path", REPORT_DIR],
                       check=True)
    print("wrote", DST)


if __name__ == "__main__":
    main()
