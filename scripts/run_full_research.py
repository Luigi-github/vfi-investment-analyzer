"""Run the complete configured multi-market VFI research study."""

from argparse import ArgumentParser
from pathlib import Path
import sys

# When executed directly, Python places the scripts directory first on sys.path.
# Add the repository root before importing sibling project modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cycle_analysis import DEFAULT_UNIVERSE, main as run_cycle_analysis
from app.research.universes import MARKET_UNIVERSES


MARKET_ORDER = [
    "USA_STOCKS",
    "USA_ETFS",
    "CRYPTO",
    "COLOMBIA",
    "CHILE",
    "LATAM",
    "COMMODITIES_ETFS",
]


def parse_args(argv=None):
    parser = ArgumentParser(description="Run the full configured VFI research study.")
    parser.add_argument(
        "--skip-all",
        action="store_true",
        help="Run each market universe but skip the combined ALL report.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    for universe in MARKET_ORDER:
        if universe not in MARKET_UNIVERSES:
            raise RuntimeError(f"Configured universe missing: {universe}")
        print("\n" + "=" * 100)
        print(f"FULL RESEARCH: {universe}")
        print("=" * 100)
        run_cycle_analysis(["--universe", universe])

    if not args.skip_all:
        print("\n" + "=" * 100)
        print(f"FULL RESEARCH: {DEFAULT_UNIVERSE}")
        print("=" * 100)
        run_cycle_analysis(["--universe", DEFAULT_UNIVERSE])


if __name__ == "__main__":
    main(sys.argv[1:])
