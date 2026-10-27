# Licence decisions

Register: [licences-image.md](licences-image.md) and SBOM: [sbom-image.cdx.json](sbom-image.cdx.json),
both generated from the deployable image (`ras:local`) with `ras sbom`. RAS is a hosted service: the
software runs on our infrastructure and is not distributed to TalentGum or students. Copyleft
obligations triggered by *distribution* therefore do not apply today. They would apply if the image or
the code were ever handed to the client to run themselves.

Engineering assessment, not legal advice: confirm with counsel before distributing the image.

| Component | Licence | Use | Decision |
|---|---|---|---|
| Transkun code and V2 weights | MIT | the transcription model | OK, keep the MIT notice |
| FFmpeg (Debian package) | LGPL/GPL (Debian build enables GPL parts) | run as a separate program for decoding | OK for a hosted service. If the image is ever distributed, ship the source offer or switch to an LGPL-only FFmpeg build |
| psycopg / psycopg-binary | LGPL-3.0 | PostgreSQL driver, used unmodified as a library | OK: dynamic use, unmodified; keep the licence text |
| soxr | LGPL-2.1+ | resampling, unmodified library | OK, same as psycopg |
| certifi | MPL-2.0 | CA bundle, unmodified | OK: file-level copyleft only applies to modified files |
| tqdm | MPL-2.0 AND MIT | progress bars (pulled in by Transkun) | OK, unmodified |

## Optional assessment models (`ras models download`)

| Component | Licence | Use | Decision |
|---|---|---|---|
| hFT-Transformer code and MAESTRO weights (Sony) | MIT | second transcriber that withdraws doubtful error claims | OK. The model code is vendored in `ras/models/vendor/hft_model.py` with its notice. |
| Kong et al. transcription (`piano_transcription_inference`, `torchlibrosa`) and Zenodo weights | MIT | pedal source and optional third transcription | OK. Its loader is not used, because it fully unpickles and shells out to wget; RAS loads the weights with `weights_only=True`. |
| piano-dynamic-estimation code and weights (He et al.) | MIT | audio dynamics (pp..ff) | OK, with a note. The upstream encoder is "adapted from GuzhengTech99", which ships no licence file. The adapted code is the upstream authors' MIT release; ask them to confirm before distributing the image. |
| Polytune (AAAI 2025 student-mistake detector) | BSD-3 **with a non-commercial restriction** | not used | Rejected: cannot be used for a paid client service. |

Downloaded checkpoints are never unpickled freely: whole-module pickles go through an allow-list loader
(`ras/models/safe_load.py`) and are converted to plain state dicts at download time.

No AGPL, SSPL or non-commercial licences are present in what ships. Re-run `ras sbom` inside the image after any
dependency change and add new flagged rows here.
