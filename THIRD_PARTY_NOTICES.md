# Licence scope and third-party code

The existing first-party AutoSlide code licence is retained in LICENSE (MIT).
`research/vendor/PSLSA/` is an unmodified copy of the comparison toolkit and carries
its own GNU GPL v3 licence in that directory. Its extracted methods and protocol
diff under PSLSA_extracted follow the same upstream terms. The source adapter and
PSLSA comparison scripts are comparison-specific integration code; their use with
the upstream implementation must preserve the applicable GPL terms. The top-level
MIT notice does not relicense the PSLSA subtree or its extracted code.

PSLSA citation: Guo et al. (2025), PSLSA v2.0, Environmental Modelling & Software
186, 106367. DOI: https://doi.org/10.1016/j.envsoft.2025.106367.
The matched comparison extracts and adapts model_train without importing the GUI;
the original source, extracted function and adapter diff are included for inspection.

ArcPy is supplied by a separately licensed ArcGIS installation, not by pip or this
archive. Scientific Python packages retain their respective licences. Third-party
geospatial products and original inventory data are not included in this source bundle.
