# IEEE TCSS Submission Checklist — TriFuse (Final)

**Multi-journal formats:** see `paper/journal_submissions/` (10 venue folders + shared content).

**Status:** Ready for ScholarOne upload (IEEE TCSS) or other journals in `journal_submissions/`.

## Upload these files

| File | Purpose |
|------|---------|
| `IEEE_TCSS_TriFuse.tex` | Main manuscript |
| `figures/` (entire folder) | 8 vector figures + `ieee_styles.tex` |
| `IEEE_TCSS_TriFuse.pdf` | Compiled PDF |
| `COVER_LETTER_TCSS.txt` | Cover letter (optional but recommended) |

## Compile

```bash
cd paper
pdflatex IEEE_TCSS_TriFuse.tex
pdflatex IEEE_TCSS_TriFuse.tex
```

## Manuscript completeness

- [x] IEEEtran journal format, 10 pages
- [x] Title page: authors, affiliations, emails (Krish → Shivansh → Pardeep)
- [x] Abstract, keywords, 9 sections, 8 figures, 8 tables, 31 references
- [x] Dataset counts aligned (I: 229,228; II: 24,783)
- [x] Results consistent across abstract, tables, text, conclusion
- [x] Six limitations in Section VII-E (incremental gains, no sig. tests, SOTA protocol, annotation mixing, DistilBERT, English/binary scope + complexity)
- [x] Ethics, Data Availability, Conflict of Interest, Acknowledgment
- [x] ref16 (IEEE TCSS) cited — aligns with target journal
- [x] Compact IEEE layout (`[!t]` floats, tight spacing)

## Before you click Submit (mandatory)

- [ ] **Supervisor proofread** — Dr. Garg final approval
- [ ] **Plagiarism check** — Turnitin / iThenticate (<15% excluding refs)
- [ ] **Co-author approval** — Shivansh and Pardeep sign off
- [ ] Skim PDF pages 1, 4–5, 7–9 (figures and tables)
- [ ] Register on ScholarOne: https://mc.manuscriptcentral.com/tcss-ieee
- [ ] Suggested reviewers (optional): 3–5 names from refs 12–16, 21–22 area
- [ ] ORCID IDs for all authors (recommended)

## ScholarOne submission tips

1. **Article type:** Regular Paper
2. **Cover letter:** Paste from `COVER_LETTER_TCSS.txt`
3. **Highlights (if prompted):** multi-view fusion; dual-dataset evaluation; ablation study; reproducible protocol
4. **Supplementary material:** Upload TriView-CBD code zip if portal allows
5. **Response to reviewers (later):** Emphasize cross-corpus consistency and honest limitation discussion

## Key results (verified)

| Metric | Dataset I | Dataset II |
|--------|-----------|------------|
| TriFuse test accuracy | 94.32% | 95.87% |
| 5-fold CV accuracy | 94.29 ± 0.13% | 95.37 ± 0.23% |
| Best neural baseline | BiLSTM 94.28% | Tuned LSTM 95.84% |

## Realistic expectation

This paper is **submittable and competently written**, but IEEE TCSS acceptance is **not guaranteed**. Reviewers may request: (a) significance tests, (b) BERT/RoBERTa baselines on same splits, (c) error analysis. The limitations section already anticipates (a) and (b). Be prepared to address reviewer comments in a revision.
