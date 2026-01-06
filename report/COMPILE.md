# Quick Compilation Guide

## Option 1: Use Overleaf (Recommended for Beginners)

1. Go to [Overleaf](https://www.overleaf.com)
2. Create free account
3. Click "New Project" → "Upload Project"
4. Upload the entire `report/` folder as ZIP
5. Click "Recompile"
6. Download PDF

**Benefits:** No local installation needed, all packages available

---

## Option 2: Docker (Works Everywhere)

```bash
cd /workspace/saumilya/vasu/FED-MED/report

# Pull LaTeX Docker image
docker pull texlive/texlive:latest

# Compile using Docker
docker run --rm -v $(pwd):/workspace -w /workspace texlive/texlive pdflatex main.tex
docker run --rm -v $(pwd):/workspace -w /workspace texlive/texlive bibtex main
docker run --rm -v $(pwd):/workspace -w /workspace texlive/texlive pdflatex main.tex
docker run --rm -v $(pwd):/workspace -w /workspace texlive/texlive pdflatex main.tex

# Output: main.pdf
```

---

## Option 3: Local Installation

### Ubuntu/Debian

```bash
# Install full LaTeX distribution (4+ GB)
sudo apt-get update
sudo apt-get install -y texlive-full

# OR minimal install
sudo apt-get install -y texlive-latex-base \
                         texlive-latex-extra \
                         texlive-science \
                         texlive-pictures \
                         bibtex

# Compile
cd /workspace/saumilya/vasu/FED-MED/report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### macOS

```bash
# Install MacTeX
brew install --cask mactex

# Compile
cd /workspace/saumilya/vasu/FED-MED/report
pdflatex main.tex
bibtex main.tex
pdflatex main.tex
pdflatex main.tex
```

### Windows

1. Download [MiKTeX](https://miktex.org/download)
2. Install MiKTeX with automatic package installation enabled
3. Open Command Prompt:
```cmd
cd C:\path\to\FED-MED\report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Missing Package Issues

If you see errors like:
```
! LaTeX Error: File `algorithm.sty' not found.
```

### Solution 1: Install specific package

```bash
# Ubuntu/Debian
sudo apt-get install texlive-science

# macOS
sudo tlmgr install algorithms

# Windows (MiKTeX)
# Packages auto-install on first use
```

### Solution 2: Comment out algorithm package

Edit `main.tex` line 10-11:

```latex
% \usepackage{algorithm}
% \usepackage{algpseudocode}
```

The algorithms in the document will render as code listings instead.

---

## Expected Output

- **File:** `main.pdf`
- **Pages:** 30-35
- **Size:** ~2-3 MB
- **Time:** 30-60 seconds (first compilation)

---

## Troubleshooting

### "Undefined citations"

Run the full compile sequence:
```bash
pdflatex main.tex
bibtex main     # Note: no .tex extension for bibtex
pdflatex main.tex
pdflatex main.tex
```

### "Out of memory"

Increase TeX memory limit:

```bash
# Edit /etc/texmf/texmf.cnf
main_memory = 12000000
```

Or use online compiler (Overleaf).

### "File not found: ../src/..."

Compile from the `report/` directory, not the project root:

```bash
cd /workspace/saumilya/vasu/FED-MED/report  # Important!
pdflatex main.tex
```

---

## Clean Build

Remove auxiliary files:

```bash
rm -f *.aux *.log *.bbl *.blg *.toc *.out *.lot *.lof
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Automated with Latexmk

```bash
sudo apt-get install latexmk

# One command compilation
latexmk -pdf main.tex

# Watch mode (recompile on save)
latexmk -pvc -pdf main.tex

# Clean
latexmk -c
```

---

## Verification

After compilation, check:

```bash
ls -lh main.pdf
# Should show file ~2-3 MB

pdfinfo main.pdf
# Should show 30+ pages
```

Open `main.pdf` and verify:
- Table of contents
- All chapters present
- Bibliography populated
- Diagrams rendered
- Code syntax highlighted
