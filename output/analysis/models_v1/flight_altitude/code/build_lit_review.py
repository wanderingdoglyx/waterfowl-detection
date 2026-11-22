"""
Build the Word report: Literature_Review.md  ->  <repo>/Flight_Altitude_Literature_Review.docx

Shares the page setup of build_report.py: pandoc's default reference document leaves the page size
and margins unset, so Word falls back to its own defaults. This pins US Letter with 1 in margins.
"""
import os
import subprocess
import tempfile

from build_report import reference_doc
from common import REPO, REPORT_DIR

SRC = os.path.join(REPORT_DIR, "Literature_Review.md")
DST = os.path.join(REPO, "Flight_Altitude_Literature_Review.docx")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "reference.docx")
        reference_doc(ref)
        subprocess.run(["pandoc", SRC, "-o", DST, "--reference-doc", ref, "--resource-path", REPORT_DIR,
                        "--toc", "--toc-depth=1"],
                       check=True)
    print("wrote", DST)


if __name__ == "__main__":
    main()
