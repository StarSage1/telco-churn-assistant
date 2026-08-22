# Architecture diagram

This folder contains the required component and data-flow diagram in three formats:

- `system-architecture.png` - presentation and README image.
- `system-architecture.svg` - editable vector source.
- `system-architecture.mmd` - Mermaid source for quick regeneration.

The diagram distinguishes the language layer from the prediction layer. Ollama converts natural language into structured fields; the Logistic Regression and CatBoost ensemble produces the churn score. The bulk route reuses the same validation, feature definitions, saved models, weights, and threshold as single-customer inference.
