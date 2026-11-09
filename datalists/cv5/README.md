# Frozen grouped five-fold split (to be restored)

Place the original fold files from the source project here, unchanged:

    fold0_train.json  fold0_val.json
    ...
    fold4_train.json  fold4_val.json
    cv5_report.md     (balance report, used for Table 2 of the paper)

Grouping: same sensor + same relative orbit + adjacent frames, union start times
within 2 h (transitive closure; 417 scenes -> 292 groups). Do not regenerate the
split: the historical fold-0 results depend on this exact assignment.

Train one fold:  python -m tools.train <config> --fold K --wandb-project ... --work-dir ...

## Status (2026-11-09)

`fold0_val.json` / `fold0_train.json` were recovered from the paper's fold-0
diagnostic logs (`docs/logs/paper_radio_f0.log`, `paper_radio_pgsg4_f0.log`,
`paper_evabl_pgsg4_f0.log`; all three list the same 83 validation scenes, whose
44-character name prefixes map one-to-one onto the archive). Train = the other
334 scenes. No datatake is shared between the two.

Folds 1-4 are still missing. Restore the original files, or recover them from the
old five-fold CM dumps with `python -m tools.data.folds_from_cm` (it refuses to
overwrite fold 0 if the dumps disagree with the files here).
