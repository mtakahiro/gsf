Release v2.0.0

Changelog (auto-generated):
- Update Sphinx configuration (`docs/conf.py`): add project path, enable Napoleon, support RTD theme if available, and mock heavy imports for clean autodoc builds.
- Add module-level docstring to `gsf/__init__.py` to improve autodoc output.
- Fix and simplify class and function docstrings in `gsf/fitting.py` and `gsf/minimizer.py` for Sphinx/Napoleon compatibility (removed unsupported scipydoc roles).
- Install and use Sphinx extensions in the venv during doc builds, and rebuild HTML into `docs/_build/html`.
- Remove `__pycache__` compiled files and commit cleanup.

Notes:
- Documentation HTML was built locally into `docs/_build/html` and is included in `release_assets/gsf-docs-v2.0.0.zip`.
- Remaining doc build warnings are informational (some import-time side effects); heavy imports are mocked to avoid full runtime dependency installation during docs builds.
