# W&B LaTeX Source

This directory was generated from the report named [Report test ](https://wandb.ai/zcemg08/uncategorized/reports/Report-test---VmlldzozNDIzODk).

All charts are rendered as high resolution PNG's in the **charts** directory.  The tables in
this report were generated from the pinned columns in the open runsets.  The underlying data
is available in the **runsets** directory.

## Quickstart

You can upload this zip into a new project at [Overleaf](https://www.overleaf.com/).
Choose "Upload Project" from the "New Project" menu to edit online.

## Rendering Locally

You'll need a tex distribution, specifically the `pdflatex` command.  The following are good options:

1. [TinyTex](https://yihui.name/tinytex/) - `wget -qO- "https://yihui.name/gh/tinytex/tools/install-unx.sh" | sh`
2. [TexLive](https://tug.org/texlive/) - `apt-get install texlive`
3. [BasicTex](http://tug.org/cgi-bin/mactex-download/BasicTeX.pkg) *(Mac)* - `brew cask install basictex`
4. [MiKTeX](https://miktex.org/download) *(Windows)*

Once installed, run: `pdflatex report.tex` to generate a pdf.