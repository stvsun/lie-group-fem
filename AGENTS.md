# Research code guidance

Keep the library independent, small, and CPU-only. Read Readme.md and
docs/METHOD.md before changing numerical behavior. Use float64 and preserve
explicit rotation references, branch guards, and units. Do not silently repair
invalid input, regularize deficient projections, or imply equilibrium from group
membership. Distinguish the local implementation from the cited papers.

Verify mathematical changes with independent analytic or numerical oracles.
Run python -m pytest and reproduce all affected examples. Never relabel failed
or pending checks as passed. Preserve raw example output; use a new run directory.

Keep environments, generated outputs, data, and secrets out of Git. MIT is the
authorized license for this repository. Changing repository visibility or adding
paid services, GPU/HPC jobs, or neural training requires separate authorization.
