"""PIC Data Processing Pipeline - Master Driver
Executes the complete sequential targeted trial emulation pipeline (Steps 3-10).
Run from command line:
    python F:/test/pipeline/run_pipeline.py              # Run all steps
    python F:/test/pipeline/run_pipeline.py --step 5     # Run specific step
    python F:/test/pipeline/run_pipeline.py --from 3 --to 6  # Run range
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_DIR
from utils import log

os.makedirs(OUTPUT_DIR, exist_ok=True)
LOGFILE = os.path.join(OUTPUT_DIR, "processing_log.txt")


def main():
    parser = argparse.ArgumentParser(description="PIC Pipeline")
    parser.add_argument("--step", type=int, help="Run a single step (3-11)")
    parser.add_argument("--from", dest="from_step", type=int, default=3,
                        help="Start step (default: 3)")
    parser.add_argument("--to", dest="to_step", type=int, default=11,
                        help="End step (default: 11)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip steps where output file already exists")
    args = parser.parse_args()

    if args.step:
        steps = [args.step]
    else:
        steps = list(range(args.from_step, args.to_step + 1))

    t0 = time.time()
    log("=" * 60, LOGFILE)
    log("PIC Data Processing Pipeline", LOGFILE)
    log(f"Steps: {steps}", LOGFILE)
    log("=" * 60, LOGFILE)

    # Shared data containers
    icu_base = None
    abx_orders = None
    micro_isolates = None
    eligible_trials = None
    baseline_cov = None
    reduction_events = None
    cloned_trials = None

    for step in steps:
        step_start = time.time()

        try:
            if step == 3:
                from step3_icu_base import run_step3
                icu_base = run_step3(LOGFILE)

            elif step == 4:
                from step4_abx_clean import run_step4
                abx_orders = run_step4(LOGFILE, icu_base=icu_base)

            elif step == 5:
                from step5_micro import run_step5
                micro_isolates = run_step5(LOGFILE, icu_base=icu_base,
                                           abx_orders=abx_orders)

            elif step == 6:
                from step6_landmark import run_step6
                eligible_trials = run_step6(LOGFILE, icu_base=icu_base,
                                            abx_orders=abx_orders,
                                            micro_isolates=micro_isolates)

            elif step == 7:
                from step7_baseline import run_step7
                baseline_cov = run_step7(LOGFILE,
                                         eligible_trials=eligible_trials,
                                         icu_base=icu_base,
                                         abx_orders=abx_orders,
                                         micro_isolates=micro_isolates)

            elif step == 8:
                from step8_reduction import run_step8
                reduction_events = run_step8(LOGFILE,
                                             eligible_trials=eligible_trials,
                                             baseline_cov=baseline_cov,
                                             abx_orders=abx_orders,
                                             icu_base=icu_base)

            elif step == 9:
                from step9_clone import run_step9
                cloned_trials = run_step9(LOGFILE,
                                          reduction_events=reduction_events,
                                          eligible_trials=eligible_trials,
                                          icu_base=icu_base)

            elif step == 10:
                from step10_outcome import run_step10
                cloned_trials = run_step10(LOGFILE,
                                           cloned_trials=cloned_trials,
                                           micro_isolates=micro_isolates,
                                           icu_base=icu_base)

            elif step == 11:
                from step11_analysis import run_step11
                cloned_trials, analysis_results = run_step11(
                    LOGFILE, cloned_trials=cloned_trials,
                    baseline_cov=baseline_cov)

            else:
                log(f"Unknown step: {step}", LOGFILE)
                continue

            elapsed = time.time() - step_start
            log(f"Step {step} completed in {elapsed:.1f}s", LOGFILE)

        except Exception as e:
            log(f"ERROR in Step {step}: {e}", LOGFILE)
            import traceback
            traceback.print_exc()
            log(traceback.format_exc(), LOGFILE)
            sys.exit(1)

    total_elapsed = time.time() - t0
    log("=" * 60, LOGFILE)
    log(f"Pipeline complete! Total time: {total_elapsed:.1f}s "
        f"({total_elapsed/60:.1f} min)", LOGFILE)
    log("=" * 60, LOGFILE)


if __name__ == "__main__":
    main()
