# Scientific workflow and inputs

The main numerical implementation is `research/scripts/exp_lib.py` (G2).

1. Install the relevant requirements in your own Python environment. The known
   numerical environment is recorded in requirements-paper.txt.
2. Place the processed study table at research/data/modeling_base.csv. It needs the
   fourteen raw factors, binary landslide labels, pointid, cell_id and frozen fold IDs.
3. Run `python research/scripts/03_main.py --groups G2 --models RF XGBoost CatBoost
   LogReg MLP SVM`. This explicit command excludes historical/auxiliary factor groups
   and LightGBM. Each inner training subset fits its own transformations.
4. Outputs are written under research/results. Keep them separate from the separately retained author results

Final corrected diagnostic scripts are loss_audit.py and mlp_stopping_audit.py.
They read research/data/modeling_base.csv and the selected-parameter/OOF files in
research/results. The extended 300-iteration loss is diagnostic; operational MLP
stopping uses internal validation. Training-fraction curves come from 08_diagnose.py.

07_seed.py contains fixed-configuration and pure-estimator-seed experiments. It also
performs domain mapping and therefore requires prediction_base.csv, raster inputs,
archived main outputs and map-stage configuration. The historical seed-A table is
excluded from reference because it does not match the final main configuration.
The corrected seed-B results are retained.

26_pslsa_common.py invokes the actual extracted PSLSA function through its protocol
adapter. Its shared selection criterion is mean inner-fold AUC, distinct from pooled
inner OOF AUC in the main analysis. See the script and the separately retained author results

For raw geospatial preparation, provide the raster, boundary and rainfall products
and update research/config/paths.json. Additional scripts specify their own required
inputs in their headers and arguments. Several archived exploratory stages support
auxiliary models/groups and are retained for provenance.

The archived 25_grid_budget_archived.py contains an older 540-candidate wide-grid
definition. The accepted manuscript's 16-versus-96 aggregate is in the separately retained author results
That archived script is not presented as the exact generator of the accepted
96-candidate table. This distinction is recorded rather than silently editing results
or replacing the historical source grid during packaging.

No source entry was executed or tested during preparation of this archive.


Source-only public repository: https://github.com/Eurekaxj/autoslide-llm-landslide-susceptibility. Numerical outputs and study inputs are not included.
