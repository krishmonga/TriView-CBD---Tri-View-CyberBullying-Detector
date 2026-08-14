# TriFuse IJMLC — Overleaf Setup Guide

## FIX: "File sn-jnl.cls not found" (do this now)

**Do NOT compile `IJMLC_TriFuse_sn.tex` unless `sn-jnl.cls` is in your project.**

### Quick fix (works immediately)

1. In Overleaf: **Menu (top left) → Main document**
2. Select **`main.tex`** (or `IJMLC_TriFuse.tex`)
3. Click **Recompile**

These files use standard `article` class — no `sn-jnl.cls` needed.

---

## Option A — Recommended: Start from Springer template on Overleaf

1. Go to [Overleaf Springer Nature template](https://www.overleaf.com/latex/templates/springer-nature-latex-template/myxmhdsbzkyd)
2. Click **Open as Template** → **New Project**
3. In the new project, set the document class line to:
   ```latex
   \documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
   ```
4. Replace the template body with files from this folder:
   - Copy content of `IJMLC_body.tex` into the main file (after `\maketitle`)
   - Or upload `IJMLC_body.tex` and add `\input{IJMLC_body.tex}`
5. Upload the **`figures/`** folder (all `.tex` files + `ieee_styles.tex`)
6. Upload `IJMLC_preamble.tex` and add `\input{IJMLC_preamble.tex}` in the preamble (before `\begin{document}`)
7. Replace abstract, keywords, title, and authors with content from `IJMLC_TriFuse_sn.tex`
8. Add Declarations, Acknowledgments, and `\input{IJMLC_bibliography.tex}` from `IJMLC_TriFuse_sn.tex`
9. Set **Main document** (Menu → Main document): your main `.tex` file
10. Recompile with **pdfLaTeX**

## Option B — Upload this folder as a zip

1. Run locally:
   ```bash
   cd IJMLC
   bash build_overleaf_zip.sh
   ```
2. On Overleaf: **New Project** → **Upload Project**
3. Upload `IJMLC_overleaf.zip`
4. If `sn-jnl.cls` is missing from the zip, the project uses Overleaf’s built-in Springer template files — open Menu → Template Selector or copy `sn-jnl.cls` from Option A’s template into your project
5. Set **Main document** to `IJMLC_TriFuse_sn.tex`
6. Recompile

## Files to upload (minimum)

| File / folder | Required |
|---------------|----------|
| `IJMLC_TriFuse_sn.tex` | Yes — main document |
| `IJMLC_body.tex` | Yes |
| `IJMLC_bibliography.tex` | Yes |
| `IJMLC_preamble.tex` | Yes |
| `figures/` (all `.tex`) | Yes |
| `sn-jnl.cls` | Yes (included in Springer Overleaf template) |
| `sn-mathphys-num.bst` | Optional for now; needed if you switch to BibTeX later |

## Overleaf settings

- **Compiler:** pdfLaTeX (default)
- **Main document:** `IJMLC_TriFuse_sn.tex`
- **TeX Live version:** 2023 or newer (Menu → Settings)

## After editing on Overleaf

When you change text on Overleaf, copy updates back to `paper/IEEE_TCSS_TriFuse.tex` if that remains your master draft, or keep Overleaf as the source of truth for IJMLC only.

To refresh from the local master before uploading again:
```bash
bash sync_from_master.sh
bash build_overleaf_zip.sh
```

## Download for Springer submission

When the PDF looks correct on Overleaf:

1. **Download PDF** — for your records and optional upload
2. **Download Source** (Menu → Download → Source) — zip of all `.tex` files
3. For Editorial Manager (https://www.editorialmanager.com/mlcy/):
   - Upload the **source zip** OR individual files
   - Tag `.tex`, `.cls`, `.bst` as **Manuscript**
   - Paste cover letter from `COVER_LETTER.txt`

## Common Overleaf errors

| Error | Fix |
|-------|-----|
| `File 'sn-jnl.cls' not found` | Use Springer Nature template project or upload `sn-jnl.cls` |
| `File 'fig01_preprocessing.tex' not found` | Upload entire `figures/` folder |
| `Undefined control sequence \fnm` | Main file must use `sn-jnl` document class, not `article` |
| Figures overflow page | Normal for draft; Springer production will adjust |

## Journal line (optional)

In `IJMLC_TriFuse_sn.tex`, this line is already set:
```latex
\journalname{International Journal of Machine Learning and Cybernetics}
```
