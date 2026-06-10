# Attribution

Parts of this bot are adapted from the official Metaculus bot template:

- Repository: https://github.com/Metaculus/metac-bot-template
  (no LICENSE file is published in that repository; it is distributed by
  Metaculus explicitly for tournament participants to dissect and adapt —
  see its README.)

Specifically:

- `bot/cdf.py`: the `NumericDefaults`, `Percentile`, and `NumericDistribution`
  classes are copied verbatim from `main_with_no_framework.py` (which mirrors
  the `forecasting-tools` library's NumericDistribution). They implement
  Metaculus' continuous-CDF standardization rules.
- `bot/metaculus_api.py`: endpoint paths and payload shapes follow the same
  reference file.
- The prompt scaffolding ideas (status-quo emphasis, percentile output format)
  are inspired by the template prompts but substantially rewritten.

A full clone of the template is kept (gitignored) in `template_reference/`
for reference.
