> **DRYRUN — NOT NFHS-6 RESULTS.** Produced from the NFHS-4 → NFHS-5 leave-one-state-out predictions as a stand-in, only to test the post-run pipeline. Never cite these numbers.

# DRYRUN outputs

Every file in this folder was produced by `scripts/post_run.py --dryrun` from the NFHS-4 → NFHS-5 leave-one-state-out predictions, used as a stand-in in the NFHS-5 → NFHS-6 predictions format. They test the post-run pipeline and report and are NOT NFHS-6 results. Every file name starts with DRYRUN_, every table has run_mode = DRYRUN, and every figure is watermarked. `scripts/post_run.py` in real mode refuses these files, and the real report is checked never to refer to them.
