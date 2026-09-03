# Preprocessing

**Checked this cycle**: `pygeovision.preprocess.Preprocessor` has real,
confirmed functional overlap with `pygeovision.data.radiometric` —
both implement band stacking, spatial clipping to a bbox, and cloud/SCL
masking. `data.radiometric`'s versions of these operations were
extensively audited and fixed this cycle (mission-aware band aliases,
real scaling formulas — see [Architecture](../architecture.md)).
Whether `preprocess.Preprocessor`'s versions share any of the same
bugs has **not** been checked this cycle — flagging the overlap
honestly rather than assuming either that it's fine or that it shares
the same fixes.

**Verification status**: import verified. Not independently audited this cycle.

::: pygeovision.preprocess
    options:
      members: true
