"""Regenerate, build, verify and publish the two local submission PDFs."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
BUILD = BASE / "build"
BUILD.mkdir(exist_ok=True)
env = dict(os.environ, MPLCONFIGDIR="/tmp/opm-mpl", BIBINPUTS=".:", BSTINPUTS=".:")
commands = [
    [sys.executable, "generate_figures.py"],
    [sys.executable, "make_supplement.py"],
    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "short_paper.tex"],
    ["bibtex", "build/short_paper"],
]
for _ in range(2):
    commands.append(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "short_paper.tex"])
commands.append(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "supplement.tex"])
commands.append(["bibtex", "build/supplement"])
for _ in range(2):
    commands.append(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "supplement.tex"])
commands.append([sys.executable, "audit/verify.py"])
with (BASE / "audit/rebuild-log.txt").open("w") as log:
    for command in commands:
        print("Running", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=BASE, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write("$ " + " ".join(command) + "\n" + result.stdout + f"\nexit_status={result.returncode}\n")
        log.flush()
        if result.returncode:
            print(result.stdout[-6000:])
            raise SystemExit(result.returncode)
        if command[-1] == "audit/verify.py":
            (BASE / "audit/verification.txt").write_text(result.stdout)
            print(result.stdout, end="")
for name in ("short_paper.pdf", "supplement.pdf"):
    shutil.copyfile(BUILD / name, BASE / name)
print("Published verified short_paper.pdf and supplement.pdf.")
