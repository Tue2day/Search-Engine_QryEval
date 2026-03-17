# QryEval

Course project code for query evaluation and retrieval experiments.

## Included

- Python source files for query parsing, retrieval, ranking, and query operators
- Experiment configurations in `EXP_DIR/`
- Query sets and test inputs in `QrySet/` and `TEST_DIR/`
- Reports in PDF format

## Excluded From GitHub

Large local index data under `INPUT_DIR/index-cw09/` and generated files under
`OUTPUT_DIR/` are intentionally excluded because they exceed normal GitHub file
size limits.

If you need to rerun the experiments, restore the local index files in
`INPUT_DIR/index-cw09/` before executing the project.
