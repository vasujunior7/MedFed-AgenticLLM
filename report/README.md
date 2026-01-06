# FED-MED LaTeX Report

## Comprehensive Academic Report on Federated Medical AI

This directory contains a complete LaTeX-based academic report documenting the **FED-MED** project: a privacy-preserving federated learning system for medical question answering using parameter-efficient fine-tuning.

---

## 📁 Directory Structure

```
report/
├── main.tex                           # Master document (compile this)
├── references.bib                     # Complete bibliography (60+ papers)
├── README.md                          # This file
├── sections/                          # All content chapters
│   ├── introduction.tex               # Motivation and contributions
│   ├── related_work.tex               # Literature review
│   ├── methodology.tex                # Technical approach
│   ├── experimental_setup.tex         # Dataset and configuration
│   ├── results.tex                    # Experimental results
│   ├── discussion.tex                 # Analysis and limitations
│   ├── conclusion.tex                 # Summary and future work
│   ├── appendix_code.tex              # Implementation listings
│   ├── appendix_config.tex            # Configuration files
│   ├── appendix_results.tex           # Additional metrics
│   └── appendix_safety.tex            # Safety framework details
└── diagrams/                          # TikZ visualizations
    ├── architecture.tex               # Federated system architecture
    ├── lora_efficiency.tex            # Parameter efficiency charts
    ├── agent_flow.tex                 # Agent aggregation flowchart
    └── safety_pipeline.tex            # Safety validation workflow
```

---

## 🔧 Compilation Requirements

### Required LaTeX Distribution

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS with Homebrew
brew install --cask mactex
```

**Alternatively (lightweight):**
```bash
# Install minimal distribution + required packages
sudo apt-get install texlive-latex-base \
                     texlive-latex-extra \
                     texlive-science \
                     texlive-pictures \
                     texlive-bibtex-extra \
                     biber
```

### Required LaTeX Packages

The document uses the following packages (all included in `texlive-full`):

**Core:**
- `geometry` (page layout)
- `graphicx` (image handling)
- `hyperref` (hyperlinks and cross-references)
- `cleveref` (smart referencing)

**Typography:**
- `amsmath, amssymb, amsthm` (mathematical typesetting)
- `booktabs` (professional tables)
- `enumitem` (customized lists)
- `xcolor` (color support)

**Code Listings:**
- `listings` (syntax-highlighted code)
- `algorithm, algpseudocode` (algorithm typesetting)

**Diagrams:**
- `tikz` (programmatic diagrams)
- `pgfplots` (data visualization)
- `multirow` (table formatting)

---

## 📝 Compilation Instructions

### Method 1: Complete Build (Recommended)

```bash
cd /workspace/saumilya/vasu/FED-MED/report

# First pass: compile document
pdflatex main.tex

# Process bibliography
bibtex main

# Second pass: resolve citations
pdflatex main.tex

# Third pass: resolve cross-references
pdflatex main.tex
```

**Expected Output:** `main.pdf` (25-40 pages)

### Method 2: Automated Build (Latexmk)

```bash
# Install latexmk
sudo apt-get install latexmk

# Automated compilation
latexmk -pdf -interaction=nonstopmode main.tex

# Clean auxiliary files after compilation
latexmk -c
```

### Method 3: Online Compilation (Overleaf)

1. Create free account at [Overleaf](https://www.overleaf.com)
2. Upload entire `report/` directory
3. Set compiler to **pdfLaTeX**
4. Set main document to `main.tex`
5. Click **Recompile**

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Missing packages**
```
! LaTeX Error: File `tikz.sty' not found.
```
**Solution:**
```bash
sudo apt-get install texlive-pictures
```

**Issue 2: Bibliography not showing**
```
LaTeX Warning: Citation 'mcmahan2017fedavg' undefined.
```
**Solution:** Run `bibtex main` then `pdflatex main` twice

**Issue 3: Out of memory**
```
! TeX capacity exceeded, sorry [main memory size=5000000].
```
**Solution:** Increase memory in `/etc/texmf/texmf.cnf`:
```
main_memory = 12000000
```

**Issue 4: File path errors**
```
! LaTeX Error: File '../src/federated/aggregation.py' not found.
```
**Solution:** Ensure you're compiling from the `report/` directory, not root

---

## 📊 Document Statistics

- **Total Pages:** 30-35 (estimated)
- **Sections:** 7 main chapters + 4 appendices
- **Figures:** 4 TikZ diagrams + 6 plots
- **Tables:** 20+ comprehensive tables
- **Code Listings:** 15+ Python/YAML examples
- **References:** 60+ peer-reviewed papers
- **Words:** ~12,000 (excluding code)

---

## 🎨 Customization

### Change Title/Authors

Edit lines 88-91 in `main.tex`:

```latex
\title{\textbf{FED-MED: Privacy-Preserving Federated Learning for Medical Question Answering}}
\author{Your Name \\ Your Institution \\ \texttt{email@example.com}}
```

### Adjust Page Margins

Edit line 3 in `main.tex`:

```latex
\usepackage[a4paper, margin=1in]{geometry}  % Change margin=1in to desired value
```

### Modify Color Scheme

Edit lines 17-19 in `main.tex`:

```latex
\definecolor{linkcolor}{RGB}{0,0,255}       % Hyperlink color (blue)
\definecolor{citecolor}{RGB}{0,128,0}       % Citation color (green)
\definecolor{urlcolor}{RGB}{128,0,128}      % URL color (purple)
```

### Include External Images

Add to appropriate section:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{path/to/image.png}
    \caption{Your caption here}
    \label{fig:your_label}
\end{figure}
```

---

## 📚 Content Overview

### Main Chapters

1. **Introduction** - Motivation, contributions, impact
2. **Related Work** - FL, medical AI, PEFT, safety
3. **Methodology** - System architecture, LoRA, agent aggregation
4. **Experimental Setup** - Dataset, configuration, metrics
5. **Results** - Training performance, ablation studies
6. **Discussion** - Findings, limitations, future work
7. **Conclusion** - Summary and recommendations

### Appendices

- **A: Implementation** - Complete Python code
- **B: Configuration** - YAML files, Docker setup
- **C: Additional Results** - Extended metrics, scalability
- **D: Safety Framework** - Ethical guidelines, FDA compliance

---

## 🔍 Key Highlights

**Technical Innovations:**
- Agent-based weighted aggregation (39.5% better than naive)
- LoRA reduces parameters by 99.95% (7B → 3.4M)
- 99.90% bandwidth reduction (13 GB → 13 MB per round)
- T4 GPU feasibility (5.0 GB VRAM vs. 26 GB full fine-tuning)

**Results:**
- 62.5% loss reduction over 5 rounds
- 3-hospital federation with non-IID data
- 10,000 medical QA samples
- 99% safety validation pass rate

**Privacy & Ethics:**
- HIPAA/GDPR compliant architecture
- Multi-layer safety guardrails
- Medical disclaimer injection
- Transparent limitations

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@techreport{fedmed2024,
  title={FED-MED: Privacy-Preserving Federated Learning for Medical Question Answering with Parameter-Efficient Fine-Tuning},
  author={[Your Name]},
  institution={[Your Institution]},
  year={2024},
  note={Technical Report}
}
```

---

## 📧 Support

For questions about compilation or content:
1. Check LaTeX logs in `main.log` for errors
2. Verify all packages installed: `tlmgr list --installed`
3. Test minimal example: `pdflatex --version`

---

## 📄 License

This report documents open-source research. Code and configurations are provided for educational purposes. Consult applicable regulations (HIPAA, GDPR, FDA) before clinical deployment.

---

**Last Updated:** December 2024  
**LaTeX Distribution:** TeXLive 2023+  
**Compiler:** pdfLaTeX  
**Bibliography:** BibTeX
