# Springer — International Journal of Machine Learning and Cybernetics (IJMLC)

| Item | Detail |
|------|--------|
| **Journal** | International Journal of Machine Learning and Cybernetics |
| **Publisher** | Springer Nature |
| **Portal** | https://www.editorialmanager.com/mlcy/ |
| **Indexing** | SCIE, Scopus |
| **Impact Factor (2024)** | ~2.7 |

## Files in this folder

| File | Purpose |
|------|---------|
| `IJMLC_TriFuse.tex` | Main manuscript (Springer `sn-jnl` or article fallback) |
| `IJMLC_body.tex` | Sections I–IX (synced from master `IEEE_TCSS_TriFuse.tex`) |
| `IJMLC_bibliography.tex` | Reference list |
| `IJMLC_preamble.tex` | Shared packages and figure path |
| `figures/` | Symlink to `paper/figures/` |
| `COVER_LETTER.txt` | Cover letter for Editorial Manager |
| `fetch_springer_template.sh` | Download official `sn-jnl.cls` bundle |

## Step 1 — Get official Springer template (required for upload)

Download the **December 2024** journal article zip from:

https://www.springernature.com/gp/authors/campaigns/latex-author-support

Copy into **this folder** (flat directory, no subfolders):

- `sn-jnl.cls`
- `sn-mathphys-num.bst` (numbered references — standard for CS/ML journals)

Or run:

```bash
cd paper/journal_submissions/11_Springer_IJMLC
bash fetch_springer_template.sh
```

## Step 2 — Sync content from master manuscript

After editing `paper/IEEE_TCSS_TriFuse.tex`, refresh the IJMLC body:

```bash
cd paper/journal_submissions/11_Springer_IJMLC
bash sync_from_master.sh
```

## Step 3 — Compile

```bash
cd paper/journal_submissions/11_Springer_IJMLC
pdflatex IJMLC_TriFuse.tex
pdflatex IJMLC_TriFuse.tex
```

Output: `IJMLC_TriFuse.pdf`

## Step 4 — Upload to Editorial Manager

Upload **in one flat zip** (all `.tex`, `.cls`, `.bst`, `figures/*.tex`):

1. `IJMLC_TriFuse.tex` — tag as **Manuscript**
2. `IJMLC_body.tex`, `IJMLC_bibliography.tex`, `IJMLC_preamble.tex` — **Manuscript**
3. `sn-jnl.cls`, `sn-mathphys-num.bst` — **Manuscript**
4. All files in `figures/` — **Manuscript**
5. `IJMLC_TriFuse.pdf` — optional preview
6. Cover letter — paste from `COVER_LETTER.txt`

**Do not** use subfolders inside the zip. Editorial Manager compiles with pdflatex.

## Before submit checklist

- [ ] `sn-jnl.cls` present and PDF compiles locally without errors
- [ ] Supervisor (Dr. Pardeep Garg) approved final PDF
- [ ] Turnitin / similarity check done
- [ ] Co-authors (Shivansh, Pardeep) approved
- [ ] Cover letter pasted in submission portal
- [ ] Decline Open Choice at acceptance unless APC is funded (subscription route = no fee)

## Notes

- Content is ML-focused for IJMLC (multi-view learning, adaptive fusion, cross-dataset evaluation).
- At acceptance, choose **subscription publication** to avoid article processing charges unless your institution funds open access.
