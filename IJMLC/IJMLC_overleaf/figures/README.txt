IEEE TCSS TriFuse — Paper Figures
=================================

Location: paper/figures/  (loaded by IEEE_TCSS_TriFuse.tex)

Fig.  File                         Section   Description
---  ---------------------------  --------  ------------------------------------------
 1   fig01_preprocessing.tex       III       Text preprocessing pipeline
 2   fig02_architecture.tex        IV        End-to-end TriFuse architecture
 3   fig03_branches_fusion.tex     IV        Unified encoding + fusion pipeline (ONE flow)
 4   fig04_training_pipeline.tex   VI        Experimental training protocol
 5   fig05_baseline_comparison.tex VII      Baseline bar chart (Table IV)
 6   fig06_single_branch.tex       VII       Single-branch bar chart (Table VI)
 7   fig07_ablation.tex            VIII      Ablation bar chart (Table VIII)
 8   fig08_training.tex            VIII      Training loss / accuracy / LR curves

Notes:
- Figs. 1–4 were fully redesigned (Jun 2026): numbered preprocessing stages, block
  architecture overview, columnar encoding+fusion flow, phased experimental pipeline.
- Fig. 3 shows cross-view interaction between branch outputs and enriched views.
- Fig. 7 and Fig. 8 are separate: ablation results vs. training dynamics.
- Bar charts use zoomed y-axes and value labels for IEEE readability.

Compile:  cd paper && pdflatex IEEE_TCSS_TriFuse.tex
