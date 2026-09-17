# Contributing

This release preserves a historical experiment. Improvements to documentation,
packaging and verification must keep the reference files intact.

## Preservation contract

- Keep every file listed in `docs/archive/reference-manifest.json` byte-identical.
- Do not regenerate the manifest merely to make a failing check pass.
- Preserve raw histories, logs, figures, datasets and the original README.
- Treat changes to model code, residuals, seeds, precision, optimizer, scheduling,
  batch sampling or simulation parameters as a separately authorized experiment.
- Record new results separately, with their code revision, command, configuration,
  environment and hardware. Never replace the historical outputs.
- Do not describe a CPU smoke test as reproduction of the original GPU run.

The manifest was captured from the supplied extraction before publication
preparation. It detects changes relative to that extraction, not changes that
may have occurred before it was supplied. It is not a signed authenticity record.

## Checks

Run these from the repository root:

```bash
python3 tools/verify_reference.py
python3 -B -m unittest discover -s tests -v
```

When PyTorch is installed, also run:

```bash
python3 -B -m unittest discover -s tests/runtime -v
```

Keep documentation changes and experimental changes separate. Describe what was
tested, on which hardware, and whether any reference bytes changed. Known issues
in the historical implementation belong in the reproduction guide until a new
experimental version is explicitly undertaken.
