# Manuscript to source map

| Manuscript item | Code or archived evidence |
|---|---|
| Fourteen factors | research/config/main_factors.json; docs/DATA.md |
| Main spatial protocol, Tables 4–6 | research/scripts/03_main.py and exp_lib.py; the separately retained author results |
| GIS preparation | research/scripts/00_prepare.py; legacy/ArcgisTools.py |
| Sampling sensitivity | research/scripts/04_buffer.py; the separately retained author results |
| Soil/rainfall checks | research/scripts/05_ablation.py, 17_rain_selection.py |
| Seed control | research/scripts/07_seed.py; the separately retained author results |
| Learning curves | research/scripts/08_diagnose.py; the separately retained author results |
| Corrected 300-step loss and early stopping | research/scripts/loss_audit.py, mlp_stopping_audit.py; the separately retained author results mlp_stopping_audit.csv |
| Matched PSLSA comparison | research/scripts/26_pslsa_common.py, pslsa_source_adapter.py; research/vendor/; the separately retained author results |
| Mapping and domain | research/scripts/09_shap_map.py, 22_full_domain.py |
| Ranking intervals | research/scripts/28_statistics.py; the separately retained author results |
| Accepted grid-budget aggregate | the separately retained author results; see REPRODUCIBILITY.md for archived-grid source distinction |

The original GUI/random-split sources are retained in legacy/ for provenance. They
are distinct from the accepted final fourteen-factor spatial analysis.


Source-only public repository: https://github.com/Eurekaxj/autoslide-llm-landslide-susceptibility. Numerical outputs and study inputs are not included.
