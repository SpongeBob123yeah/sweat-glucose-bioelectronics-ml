from pathlib import Path
import importlib.metadata
import sys
import subprocess

EXPECTED_PACKAGES = {
    "numpy": "2.2.4",
    "scipy": "1.15.2",
    "pandas": "2.2.3",
    "scikit-learn": "1.6.1",
    "xgboost": "3.0.2",
    "shap": "0.48.0",
    "matplotlib": "3.10.1"
}
EXPECTED_PYTHON = "3.13.3"


def check_exact_version(pkg_name, expect_ver):
    try:
        installed_ver = importlib.metadata.version(pkg_name)
        if installed_ver != expect_ver:
            print(f"\n【FATAL ERROR】Version lock violation (paper reproduction)")
            print(f"Package: {pkg_name}")
            print(f"Required == {expect_ver}, Detected = {installed_ver}")
            sys.exit(1)
        print(f"OK | {pkg_name} {installed_ver}")
        return f"{pkg_name} == {installed_ver}"
    except importlib.metadata.PackageNotFoundError:
        print(f"\n【FATAL ERROR】Missing package: {pkg_name}")
        sys.exit(1)


if __name__ == "__main__":
    print("===== v5.0 PARETO‑LOCKED Strict Environment Check =====")
    env_records = []
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if py_ver != EXPECTED_PYTHON:
        print(f"\n【FATAL ERROR】Python version mismatch")
        print(f"Required == {EXPECTED_PYTHON}, Detected = {py_ver}")
        sys.exit(1)
    print(f"OK | Python {py_ver}")
    env_records.append(f"Python == {py_ver}")

    for pkg, target_ver in EXPECTED_PACKAGES.items():
        line = check_exact_version(pkg, target_ver)
        env_records.append(line)

    print("\nAll environment checks passed. Starting workflow.\n")

    BASE = Path(__file__).resolve().parent
    RESULT_DIR = BASE.parent / "results"
    RESULT_DIR.mkdir(exist_ok=True)

    workflow_scripts = [
        "00_build_database.py",
        "01_model_analysis.py",
        "03_circular_heatmap.py",
        "02_make_figures.py"
    ]
    for script in workflow_scripts:
        print(f"===== Running {script} =====")
        subprocess.run([sys.executable, str(BASE / script)], check=True)

    snapshot_file = RESULT_DIR / "environment.txt"
    with open(snapshot_file, "w", encoding="utf-8") as f:
        f.write("# Auto-generated environment snapshot (run_all.py)\n")
        f.write("# v5.0 PARETO‑LOCKED paper frozen versions\n\n")
        f.write("\n".join(env_records))

    print(f"\nEnvironment snapshot saved -> {snapshot_file}")
    print("\n===== Reproduction finished successfully =====")
