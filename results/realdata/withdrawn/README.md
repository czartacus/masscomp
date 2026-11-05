These three probe outputs (2026-11-04) were produced with compiled kernels that numba had cached under
the plan inventory (29 segments) and reloaded under the extended one (40): the edit DP and the lattice
read the deletion / no-insertion / start-of-form columns at index 29 instead of 40, so the likelihood
was wrong. Kept for the record; superseded by the 2026-11-05 runs in the parent directory.
