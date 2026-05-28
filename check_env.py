"""
check_env.py — Run this once on your machine to verify your environment.

    python check_env.py           # check only
    python check_env.py --install # check + install anything missing
"""

import sys
import subprocess
import argparse

# ── Minimum Python version ────────────────────────────────────────────────────
MIN_PYTHON = (3, 10)

# ── Packages to check ─────────────────────────────────────────────────────────
# (import_name, pip_name, needed_for)
PACKAGES = [
    # Core
    ("numpy",       "numpy",          "all phases"),
    ("pandas",      "pandas",         "all phases"),
    ("scipy",       "scipy",          "Phase 4/5"),
    ("yaml",        "pyyaml",         "config loading"),
    ("openpyxl",    "openpyxl",       "Excel export"),
    ("tqdm",        "tqdm",           "progress bars"),
    # Image / CV
    ("cv2",         "opencv-python",  "Phase 3/4"),
    ("PIL",         "Pillow",         "Phase 3/4"),
    # 3D
    ("open3d",      "open3d",         "Phase 2/3/6"),
    ("trimesh",     "trimesh",        "Phase 2/4"),
    # Visualization
    ("matplotlib",  "matplotlib",     "Phase 5/7"),
    ("plotly",      "plotly",         "Phase 5/7"),
    # ML
    ("sklearn",     "scikit-learn",   "Phase 4/5"),
    # Notebooks
    ("jupyterlab",  "jupyterlab",     "exploration"),
]

PYTORCH_NOTE = (
    "torch — NOT checked here. Install manually from https://pytorch.org/get-started/locally/\n"
    "         (needed for Phase 3 segmentation — CPU or GPU depending on your machine)"
)

# ── Colors ────────────────────────────────────────────────────────────────────
G = "\033[92m"  # green
R = "\033[91m"  # red
Y = "\033[93m"  # yellow
B = "\033[94m"  # blue
X = "\033[0m"   # reset

def ok(msg):  print(f"  {G}✓{X}  {msg}")
def err(msg): print(f"  {R}✗{X}  {msg}")
def warn(msg):print(f"  {Y}!{X}  {msg}")
def info(msg):print(f"  {B}▸{X}  {msg}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_python():
    v = sys.version_info
    if v >= MIN_PYTHON:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    else:
        err(f"Python {v.major}.{v.minor} — need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+")
        return False

def check_pip():
    try:
        import pip
        ok(f"pip {pip.__version__}")
        return True
    except ImportError:
        err("pip not found")
        return False

def check_venv():
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        ok(f"Virtual environment active: {sys.prefix}")
    else:
        warn("No virtual environment active  ← recommended but not required")
    return True

def check_git():
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, text=True)
        ok(r.stdout.strip())
    except FileNotFoundError:
        err("git not found — install from https://git-scm.com")

def check_sqlite():
    try:
        import sqlite3
        ok(f"sqlite3 {sqlite3.sqlite_version}  (built-in)")
    except ImportError:
        err("sqlite3 missing — this is very unusual, check your Python install")

def try_import(import_name):
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "?")
        return True, version
    except ImportError:
        return False, None

def install_package(pip_name):
    print(f"     Installing {pip_name}...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"{G}done{X}")
        return True
    else:
        print(f"{R}failed{X}")
        print(f"     {result.stderr.strip()}")
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true",
                        help="Install missing packages automatically")
    args = parser.parse_args()

    print(f"\n{'═'*52}")
    print(f"  Burn Scar Project — Environment Check")
    print(f"{'═'*52}\n")

    # System checks
    print("System")
    print("──────")
    py_ok = check_python()
    if not py_ok:
        print(f"\n{R}Python version too old. Please upgrade and re-run.{X}\n")
        sys.exit(1)
    check_pip()
    check_venv()
    check_git()
    check_sqlite()

    # Package checks
    print("\nPackages")
    print("────────")
    missing = []
    for import_name, pip_name, used_for in PACKAGES:
        found, version = try_import(import_name)
        if found:
            ok(f"{pip_name:<20} {version:<12}  ({used_for})")
        else:
            err(f"{pip_name:<20} NOT INSTALLED     ({used_for})")
            missing.append((import_name, pip_name, used_for))

    # PyTorch (separate — needs manual choice)
    print()
    warn(PYTORCH_NOTE)

    # sqlite3 built-in reminder
    print()
    info("sqlite3 is built into Python — no install needed")

    # Summary
    print(f"\n{'─'*52}")
    if not missing:
        print(f"{G}All packages installed. Environment ready.{X}")
    else:
        print(f"{Y}{len(missing)} package(s) missing:{X}")
        for _, pip_name, used_for in missing:
            print(f"    • {pip_name}  ({used_for})")

        if args.install:
            print(f"\nInstalling missing packages...")
            failed = []
            for import_name, pip_name, _ in missing:
                success = install_package(pip_name)
                if not success:
                    failed.append(pip_name)
            if failed:
                print(f"\n{R}Failed to install: {', '.join(failed)}{X}")
                print("Try installing manually:")
                for p in failed:
                    print(f"  pip install {p}")
            else:
                print(f"\n{G}All packages installed successfully.{X}")
        else:
            print(f"\nRun with --install to install them automatically:")
            print(f"  python check_env.py --install")

    print()

if __name__ == "__main__":
    main()
