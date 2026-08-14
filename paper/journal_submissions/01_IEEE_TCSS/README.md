# IEEE Transactions on Computational Social Systems (TCSS)

| Item | Detail |
|------|--------|
| **Author fee** | **FREE** (traditional subscription publication) |
| **Portal** | https://mc.manuscriptcentral.com/tcss-ieee |
| **Format** | IEEEtran journal (`\documentclass[journal]{IEEEtran}`) |
| **Difficulty** | Harder — expect rigorous review |

## Files

- `TriFuse.tex` — copy of master manuscript (use this or `../IEEE_TCSS_TriFuse.tex`)
- `COVER_LETTER.txt`

## Compile

```bash
cd paper/journal_submissions/01_IEEE_TCSS
pdflatex TriFuse.tex && pdflatex TriFuse.tex
```

Requires `figures/` — symlink included as `figures -> ../../figures`.

## Upload

Submit `TriFuse.tex`, entire `figures/` folder, and compiled PDF.
