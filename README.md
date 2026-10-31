# AutoSlide

An LLM-enabled framework for automated landslide susceptibility assessment.

Source code accompanying **A Large Language Model Framework for Automated Landslide Susceptibility Assessment**.

Repository: https://github.com/Eurekaxj/autoslide-llm-landslide-susceptibility

This repository contains source code, configuration templates, dependency declarations and documentation only. Study datasets, fold assignments, numerical result tables, fitted models, manuscripts and reviewer correspondence are not distributed here.

## Structure

- `research/scripts/`: fourteen-factor spatial modelling, preprocessing, sampling, seed experiments, learning diagnostics and toolkit comparison.
- `research/config/`: factor definitions and user-configurable input paths.
- `research/vendor/`: PSLSA comparator source, extracted functions and original licence.
- `legacy/`: original application and development code, distinct from the final scientific workflow.
- `docs/`: data schema, execution guidance and manuscript-to-code map.

## Main analysis

Install `requirements-models.txt`, supply your processed input table as `research/data/modeling_base.csv`, then run:

```bash
python research/scripts/03_main.py --groups G2 --models RF XGBoost CatBoost LogReg MLP SVM
```

The final configuration uses fourteen raw factors, five spatial outer regions, cell-grouped inner validation, training-only transformations and pooled-inner-OOF AUC selection. Provide your frozen `fold` and `cell_id` columns. GIS/rainfall stages require separately supplied inputs and their appropriate dependencies. ArcPy is provided by a licensed ArcGIS installation. Running these scripts does not require an OpenAI API key.

See `docs/DATA.md`, `docs/REPRODUCIBILITY.md` and `docs/PAPER_CODE_MAP.md`. Historical GUI modules need local configuration; see `legacy/README.md`.

## Licence and citation

First-party MIT licence is preserved. PSLSA and extracted comparison code retain their original GPL terms; see `THIRD_PARTY_NOTICES.md`. Citation metadata is in `CITATION.cff`.

This upload is source-only. No training, program execution or tests were run during packaging.
