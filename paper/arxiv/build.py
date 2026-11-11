#!/usr/bin/env python3
"""Build and check the preprint, then package only its TeX dependencies."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile

BASE = Path(__file__).resolve().parent
BUILD = BASE / "build"
OUTPUT = BASE / "output" / "pdf"
FILES = ["main.tex", "main_text.tex", "supplement_text.tex", "supplement_tables.tex",
         "factorial_values.tex", "references.bib", "main.bbl", "figures/transfer_overview.pdf"]


def run(args, cwd, logfile, env=None):
    with logfile.open("w") as log:
        subprocess.run(args, cwd=cwd, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def main():
    BUILD.mkdir(exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
               "-output-directory=build", "main.tex"]
    run(command, BASE, BUILD / "pass1.txt")
    env = dict(os.environ, BIBINPUTS=".:")
    run(["bibtex", "build/main"], BASE, BUILD / "bibtex.txt", env)
    for i in (2, 3):
        run(command, BASE, BUILD / f"pass{i}.txt")
    log = (BUILD / "main.log").read_text()
    for bad in ["Overfull", "undefined", "multiply defined", "LaTeX Warning"]:
        assert bad not in log, f"Build warning: {bad}"
    bbl = (BUILD / "main.bbl").read_text()
    shutil.copy2(BUILD / "main.bbl", BASE / "main.bbl")
    sources = "\n".join((BASE / f).read_text() for f in FILES if f.endswith(".tex"))
    assert chr(0x2014) not in sources + bbl, "Em dash in source"
    assert "---" not in sources + bbl, "TeX em dash in source"
    assert "Anonymous Authors" not in sources
    cited = {key.strip() for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", sources)
             for key in group.split(",")}
    printed = set(re.findall(r"\\bibitem(?:\[[\s\S]*?\])?\{([^{}]+)\}", bbl))
    assert cited == printed, (cited, printed)
    pdftext = subprocess.check_output(["pdftotext", str(BUILD / "main.pdf"), "-"], text=True)
    assert "??" not in pdftext and chr(0x2014) not in pdftext
    page_count = len([p for p in pdftext.split("\f") if p.strip()])
    assert all(f"Table S{i}:" in pdftext for i in range(1, 9)), "Missing supplement table"

    # Check that formatting and manuscript corrections preserved the verified results.
    workshop = BASE.parent / "neuroai2027"
    assert (BASE / "factorial_values.tex").read_bytes() == (workshop / "factorial_values.tex").read_bytes()
    assert (BASE / "figures/transfer_overview.pdf").read_bytes() == (workshop / "figures/transfer_overview.pdf").read_bytes()
    expected_tables = (workshop / "supplement_tables.tex").read_text().replace("{table*}", "{table}")
    assert (BASE / "supplement_tables.tex").read_text() == expected_tables

    archive = BASE / "opm_bridge_arxiv_source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in FILES:
            tar.add(BASE / name, arcname=name)
    # Compile only extracted upload files. This catches accidental dependencies
    # on the workshop folder or the existing build directory.
    with tempfile.TemporaryDirectory(prefix="opm-arxiv-") as folder:
        clean = Path(folder)
        with tarfile.open(archive) as tar:
            tar.extractall(clean, filter="data")
        # Three passes: with the current bibliography two leave labels unsettled.
        for i in (1, 2, 3):
            run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                clean, BUILD / f"archive-pass{i}.txt")
        clean_log = (clean / "main.log").read_text()
        assert not any(word in clean_log for word in ["undefined", "Overfull", "LaTeX Warning"])
        clean_text = subprocess.check_output(["pdftotext", str(clean / "main.pdf"), "-"], text=True)
        assert pdftext == clean_text, "Archive PDF differs from local PDF text"

    shutil.copy2(BUILD / "main.pdf", OUTPUT / "opm_bridge_arxiv.pdf")
    manifest = {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in FILES}
    report = dict(pages=page_count, bibliography_entries=len(printed),
                  standalone_archive_compile="passed", scientific_values="unchanged",
                  source_sha256=manifest)
    (BUILD / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {page_count} pages, {len(printed)} references, all 8 supplementary tables, "
          "no unresolved references or overfull boxes; extracted archive compiles independently.")


if __name__ == "__main__":
    main()
