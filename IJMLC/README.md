# TriFuse — IJMLC Submission Package

Standalone folder for **International Journal of Machine Learning and Cybernetics** (Springer Nature).

**Portal:** https://www.editorialmanager.com/mlcy/

## Folder contents (IJMLC only)

```
IJMLC/
├── IJMLC_TriFuse.tex          Main manuscript (preview / compile without sn-jnl)
├── IJMLC_TriFuse_sn.tex       Official Springer sn-jnl version (upload this)
├── IJMLC_body.tex             Paper sections I–IX
├── IJMLC_bibliography.tex     References
├── IJMLC_preamble.tex         Packages and figure setup
├── figures/                   All TikZ figure sources (self-contained)
├── COVER_LETTER.txt           Cover letter for Editorial Manager
├── sync_from_master.sh        Refresh body from master manuscript
├── fetch_springer_template.sh Download sn-jnl.cls
├── build_upload_zip.sh        Create flat zip for upload
└── README.md                  This file
```

## Quick start

### 1. Download Springer template (once)

From https://www.springernature.com/gp/authors/campaigns/latex-author-support

Copy into this folder:
- `sn-jnl.cls`
- `sn-mathphys-num.bst`

Or run: `bash fetch_springer_template.sh`

### 2. Compile PDF

```bash
cd IJMLC
pdflatex IJMLC_TriFuse_sn.tex
pdflatex IJMLC_TriFuse_sn.tex
```

Preview without `sn-jnl.cls`: use `IJMLC_TriFuse.tex` instead.

### 3. Sync after editing master paper

If you change `paper/IEEE_TCSS_TriFuse.tex`:

```bash
cd IJMLC
bash sync_from_master.sh
```

### 4. Create upload zip

```bash
cd IJMLC
bash build_upload_zip.sh
```

Upload `IJMLC_submission.zip` to Editorial Manager (tag all `.tex`, `.cls`, `.bst` as **Manuscript**).

## Submit checklist

- [ ] `sn-jnl.cls` in this folder; PDF compiles without errors
- [ ] Cover letter pasted from `COVER_LETTER.txt`
- [ ] Dr. Pardeep Garg approved final PDF
- [ ] Turnitin / similarity check done
- [ ] Co-authors approved
- [ ] At acceptance: choose **subscription** route unless APC is funded

## Note

This folder is **self-contained**. You can zip the entire `IJMLC/` directory and share or upload it without other project folders.
