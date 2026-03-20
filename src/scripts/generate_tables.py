"""
standalone script to generate all paper latex tables via ComprehensiveEvaluator.

this script runs the full evaluation pipeline and writes latex tables to
results/tables/ at the project root, ready for inclusion in the paper via
\\input{../../results/tables/<table_name>}.

usage:
    cd src && python3 scripts/generate_tables.py
    cd src && python3 scripts/generate_tables.py --full
    cd src && python3 scripts/generate_tables.py --output /path/to/results/tables

output tables:
    table1_attack_results.tex    — attack asr-r/a/t/isr with bootstrap ci
    table2_defense_matrix.tex    — attack × defense matrix (asr-r)
    table3_defense_tpr_fpr.tex   — tpr/fpr per defense per attack
    table4_evasion.tex           — watermark evasion results
    table5_adaptive_sad.tex      — adaptive adversary vs sad
    table6_ablation.tex          — ablation study summary

all comments are lowercase.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# add src/ to path when running from within src/ or project root
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from evaluation.comprehensive_eval import ComprehensiveEvaluator
from utils.logging import logger

# default output directory: project_root/results/tables/
_PROJECT_ROOT = _HERE.parent.parent
_DEFAULT_OUTPUT = str(_PROJECT_ROOT / "results" / "tables")


def generate_all_tables(
    output_dir: str = _DEFAULT_OUTPUT,
    full_mode: bool = False,
    corpus_size: int = 200,
    n_poison: int = 5,
    top_k: int = 5,
    n_seeds: int = 5,
    seed_base: int = 42,
) -> dict:
    """
    run comprehensive evaluation and write all paper latex tables.

    args:
        output_dir: directory to write .tex files
        full_mode: run all phases (ablations, evasion, adaptive); slower
        corpus_size: benign corpus size for evaluation
        n_poison: base poison count (per attack)
        top_k: retrieval top-k
        n_seeds: number of seeds for bootstrap ci
        seed_base: base random seed

    returns:
        dict mapping table_name → file path
    """
    t0 = time.time()
    print("generating paper tables via comprehensive evaluator")
    print(f"  output_dir  : {output_dir}")
    print(f"  corpus_size : {corpus_size}")
    print(f"  n_poison    : {n_poison}")
    print(f"  top_k       : {top_k}")
    print(f"  n_seeds     : {n_seeds}")
    print(f"  mode        : {'full' if full_mode else 'quick'}")
    print("")

    eval = ComprehensiveEvaluator(
        corpus_size=corpus_size,
        n_poison=n_poison,
        top_k=top_k,
        n_seeds=n_seeds,
        seed_base=seed_base,
        run_matrix=True,
        run_evasion=True,
        run_adaptive=True,
        run_ablations=full_mode,
    )

    print("[1/3] running comprehensive evaluation...")
    result = eval.run()
    elapsed_eval = time.time() - t0
    print(f"  evaluation complete in {elapsed_eval:.1f}s")

    # save json for reproducibility
    json_path = Path(output_dir).parent / "comprehensive_results.json"
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        result.save_json(str(json_path))
        print(f"  results json: {json_path}")
    except Exception as exc:
        logger.log_error("generate_tables", exc, {"phase": "save_json"})

    print("[2/3] generating latex tables...")
    tables = eval.generate_paper_tables(result, output_dir)
    print(f"  generated {len(tables)} table(s):")
    for name, path in sorted(tables.items()):
        print(f"    {name}: {path}")

    print("[3/3] generating summary statistics...")
    _print_summary(result)

    total_elapsed = time.time() - t0
    print(f"\ndone in {total_elapsed:.1f}s — tables written to {output_dir}")
    return tables  # type: ignore[no-any-return]


def _print_summary(result: Any) -> None:
    """print a brief summary of the evaluation results to stdout."""
    print("")
    if result.attack_summaries:
        print("  attack summary (asr-r mean [95% ci]):")
        for at, s in sorted(result.attack_summaries.items()):
            asr_r = s.get("asr_r", {})
            if isinstance(asr_r, dict):
                mean = asr_r.get("mean", 0)
                lo = asr_r.get("lower", 0)
                hi = asr_r.get("upper", 0)
                print(f"    {at:<18} {mean:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                print(f"    {at:<18} {float(asr_r):.3f}")

    if result.adaptive_sad_results:
        print("  adaptive sad results (evasion_rate → retrieval_degradation):")
        for at, r in sorted(result.adaptive_sad_results.items()):
            er = r.get("evasion_rate", 0)
            rd = r.get("retrieval_degradation", 0)
            print(f"    {at:<18} evasion={er:.3f}  retrieval_loss={rd:.3f}")


# ---------------------------------------------------------------------------
# cli entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """command-line entry for standalone table generation."""
    parser = argparse.ArgumentParser(
        description="generate all paper latex tables from comprehensive evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help="directory to write .tex table files",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run full mode including ablation sweeps (slow, ~30-60 min)",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=200,
        help="benign corpus size",
    )
    parser.add_argument(
        "--n-poison",
        type=int,
        default=5,
        help="base poison count per attack",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="number of seeds for bootstrap ci",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="base random seed",
    )

    args = parser.parse_args()

    try:
        tables = generate_all_tables(
            output_dir=args.output,
            full_mode=args.full,
            corpus_size=args.corpus_size,
            n_poison=args.n_poison,
            n_seeds=args.n_seeds,
            seed_base=args.seed,
        )
        print(f"\n{len(tables)} table(s) ready for paper inclusion.")
        sys.exit(0)
    except Exception as exc:
        import traceback

        print(f"\nerror: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
