# Desktop GUI and integrated task evidence

This release retains the native GUI, final simultaneous-transition demonstration recordings, both complete integrated walking evaluations (including failures), and the preparation-speed recheck. It does not claim that the integrated walking success-rate requirement was met.

The original four final videos are archived without transcoding, frame-rate changes, or trimming. Older superseded recording videos, live shared-memory files and machine-specific audit-view symlinks are omitted. The underlying measurements and final audit results remain included. Local originals have not been deleted.

Archives larger than 48 MiB are split into ordered parts to stay below Git hosting file limits. `manifest.json` records each original file's SHA-256, each archive/part checksum, and verified file counts. Archive paths are relative to the repository root.

Verify all archive parts and original file contents:

```sh
python artifacts/release_20261114/restore.py --verify-only
```

Restore everything, or select one group:

```sh
python artifacts/release_20261114/restore.py
python artifacts/release_20261114/restore.py transitions
```

Restoration refuses to overwrite an existing file whose contents differ. The three groups are `desktop`, `transitions`, and `walking`. Unpacked outputs remain ignored by Git; the archives are the retained copies. Historical audit scripts must use the matching saved runtime source version; later default-budget changes can invalidate their input comparisons. No new physical experiment is required for restoration.

Reports available without unpacking:

- [Native GUI usage](../../for_codex/TASK_FORMAT.md)
- [Final transition demonstrations](../gui_direct_transition_20261111/README.md)
- [Integrated walking, 15 seconds](../walking_integrated_20261112/VALIDATION.md)
- [Integrated walking, 20 seconds](../walking_integrated20_20261112/VALIDATION.md)
- [Preparation speed after other jobs stopped](../walking_speed_recheck_20261113/VALIDATION.md)
