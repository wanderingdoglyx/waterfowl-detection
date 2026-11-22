"""
Step 6 — typeset the report's display equations with LaTeX as fixed-width, high-resolution PNGs.

Every equation is set on the same 6.5 in canvas (the report's text width) at 14.4 pt, centred, with its
number flush right, so all equations print at one consistent size in Word or any other viewer.
Needs pdflatex (newtx fonts) and Ghostscript.

Output: ../equations/eqNN.png
"""
import os
import shutil
import subprocess
import tempfile

from common import REPORT_DIR

OUT = os.path.join(REPORT_DIR, "equations")
WIDTH_IN = 6.5
DPI = 600

REC, OPT, BOX = r"H_{\mathrm{rec}}", r"P_{\mathrm{optimum}}", r"P_{\mathrm{box}}"
GSDOPT = r"\mathrm{GSD}_{\mathrm{optimum}}"
MAIN = rf"{REC} = \frac{{k \times S \times f \times N}}{{w \times {OPT}}}"
WORDS = (r"\text{Recommended height} = \frac{k \times \text{bird size} \times \text{focal length} \times \text{image width}}"
         r"{\text{sensor width} \times \text{optimum pixel size}}")
EQUATIONS = [  # (file stem, LaTeX, number or None, boxed)
    # the result (Summary and Section 2.1)
    ("eq00", MAIN, None, True),
    ("eq00w", WORDS, None, False),
    ("eq01", MAIN, "1", True),
    ("eq01w", WORDS, None, False),
    # Section 2.2 — the derivation
    ("eq02", r"\mathrm{GSD} = \frac{H \times w}{f \times N}", "2", False),
    ("eq03", r"C = \frac{f \times N}{w} \qquad\text{so}\qquad \mathrm{GSD} = \frac{H}{C}", "3", False),
    ("eq04", r"P = \frac{S}{\mathrm{GSD}} = \frac{S \times C}{H}", "4", False),
    ("eq05", rf"{BOX} = \sqrt{{P^{{2}} + b^{{2}}}}", "5", False),
    ("eq06", rf"\sqrt{{\left(\frac{{S \times C}}{{{REC}}}\right)^{{2}} + b^{{2}}}} = {OPT}", "6", False),
    ("eq07", rf"{REC} = \frac{{S \times C}}{{\sqrt{{{OPT}^{{2}} - b^{{2}}}}}} = \frac{{k \times S \times C}}{{{OPT}}}", "7", False),
    ("eq08", rf"k = \frac{{{OPT}}}{{\sqrt{{{OPT}^{{2}} - b^{{2}}}}}}", "8", False),
    # Section 2.3 — k without measuring b
    ("eq09", rf"\kappa = \frac{{{OPT}}}{{b}} \approx 2.5 \qquad\Longrightarrow\qquad "
             r"k = \frac{1}{\sqrt{1 - 1/\kappa^{2}}} \approx 1.09", "9", False),
    ("eq10", r"b^{2} = b_{\text{optics}}^{2} + b_{\text{margin}}^{2}", "10", False),
    ("eq11", r"S_{\text{eff}} = \sqrt{S^{2} + \left(v \times t_{\exp}\right)^{2}}", "11", False),
    # Section 2.4 — other forms of the same equation
    ("eq12", rf"{GSDOPT} = \frac{{k \times S}}{{{OPT}}}", "12", False),
    ("eq13", rf"{REC} = {GSDOPT} \times C", "13", False),
    ("eq14", rf"H_{{\mathrm{{set}}}} = {REC} - h_{{\mathrm{{takeoff}}}}", "14", False),
    ("eq15", r"C \approx \frac{f_{35} \times D}{43.27~\mathrm{mm}}", "15", False),
    # Section 5.2 — how this study measured b
    ("eq16", r"a^{2} = \frac{\left(P_{1}^{2} - P_{2}^{2}\right) H_{1}^{2}\, H_{2}^{2}}{H_{2}^{2} - H_{1}^{2}}"
             r"\qquad (a = S \times C)", "16", False),
    ("eq17", r"b = \sqrt{P_{1}^{2} - \frac{a^{2}}{H_{1}^{2}}}", "17", False),
    # Sections 6 and 7
    ("eq18", rf"\operatorname{{logit}} \Pr(\text{{detected}}) = \beta_{{\text{{dataset}}}} + g\bigl(\log {BOX}\bigr)", "18", False),
    ("eq19", rf"{OPT} = \text{{the box size }} {BOX} \text{{ at which }} R({BOX}) \text{{ is highest}}", "19", False),
    ("eq20", r"S_i = \sqrt{P_{\mathrm{box},i}^{2} - b^{2}} \times \frac{H_i}{C}", "20", False),
]

TEMPLATE = r"""\documentclass[12pt,border={0pt 6pt 0pt 6pt}]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage{xcolor}
\setlength{\fboxsep}{9pt}
\setlength{\fboxrule}{0.8pt}
\begin{document}
\large
\makebox[%(width)sin]{\hfill %(body)s\hfill\llap{%(tag)s}}
\end{document}
"""


def render(stem: str, latex: str, tag: str | None, boxed: bool, tmp: str) -> str:
    body = rf"$\displaystyle {latex}$"
    if boxed:
        body = rf"\fcolorbox{{black!45}}{{black!4}}{{{body}}}"
    tex = TEMPLATE % dict(width=WIDTH_IN, body=body, tag=f"({tag})" if tag else "")
    with open(os.path.join(tmp, stem + ".tex"), "w") as f:
        f.write(tex)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"],
                       cwd=tmp, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pdflatex failed for {stem}:\n{r.stdout[-2000:]}")
    png = os.path.join(OUT, stem + ".png")
    subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", f"-r{DPI}",
                    "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", f"-sOutputFile={png}",
                    os.path.join(tmp, stem + ".pdf")], check=True)
    return png


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="eqs_")
    try:
        for stem, latex, tag, boxed in EQUATIONS:
            print("wrote", render(stem, latex, tag, boxed, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
