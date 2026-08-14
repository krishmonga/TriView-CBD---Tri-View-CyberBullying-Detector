# TriFuse — Multi-Journal Submission Formats

This folder contains **ready-to-adapt** submission packages for journals and conferences where the TriFuse paper can be submitted. All packages share the same scientific content via `shared/TriFuse_body.tex`.

## Folder structure

```
journal_submissions/
├── README.md
├── sync_from_master.sh
├── shared/
│   ├── TriFuse_body.tex
│   ├── TriFuse_bibliography.tex
│   ├── TriFuse_figure_packages.tex
│   └── TriFuse_abstract.txt
├── 01_IEEE_TCSS/             FREE — IEEE Transactions (primary)
├── 02_Springer_SNAM/         FREE — subscription route
├── 03_Elsevier_OSNEM/        FREE — subscription route
├── 04_Elsevier_ESWA/         FREE — subscription route
├── 05_Elsevier_CEE/          FREE — subscription route
├── 06_IAES_IJ-AI/            FREE — Scopus, easier
├── 07_IJACSA/                FREE — applied CS
├── 08_IEEE_INDICON/          FREE publish* (*registration fee)
├── 09_IEEE_Access/           PAID (~USD 1,850 APC)
├── 10_MDPI_Information/    PAID (~CHF 2,000 APC)
└── 11_Springer_IJMLC/      FREE — subscription route (current target)
```

## After editing the master paper

```bash
cd paper/journal_submissions
bash sync_from_master.sh
```

Figures are read from `paper/figures/` (no duplication).

## Recommended free venues (easiest first)

| Priority | Folder | Fee | Difficulty |
|----------|--------|-----|------------|
| 1 | `06_IAES_IJ-AI` | Free | Easiest |
| 2 | `07_IJACSA` | Free | Easiest |
| 3 | `02_Springer_SNAM` | Free (non-OA) | Medium |
| 4 | `03_Elsevier_OSNEM` | Free (non-OA) | Medium |
| 5 | `01_IEEE_TCSS` | Free | Harder |

## Before any submission

- Run `sync_from_master.sh` after editing `IEEE_TCSS_TriFuse.tex`
- Supervisor approval, plagiarism check, co-author sign-off
- Download official publisher template if editor requires it
