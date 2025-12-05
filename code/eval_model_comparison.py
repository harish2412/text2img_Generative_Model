import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path


def setup_logger(name: str) -> logging.Logger:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = logs_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)

    logger.info(f"Logging to {logfile}")
    return logger


logger = setup_logger("eval_model_comparison")

VAL_CSV = "../data/manifests/coco_train1k.csv"
MODEL_A_DIR = "../sample_outputs/milestone1_output"
MODEL_B_DIR = "../sample_outputs/milestone2_output"
# UPDATED: point to your new Model C folder
MODEL_C_DIR = "../sample_outputs/modelC_finetuned_4k"
REPORT_PATH = "../sample_outputs/milestone3_report.json"


def run_eval(model_name: str, gen_dir: str, n: int = 1000):
    """
    Runs eval_fid_is.py as a subprocess and parses FID/IS from stdout.
    """
    logger.info(f"\nRunning evaluation for {model_name} (n={n})...")
    cmd = [
        "python",
        "eval_fid_is.py",
        "--val_csv",
        VAL_CSV,
        "--gen_dir",
        gen_dir,
        "--n",
        str(n),
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = process.communicate()

    logger.info(f"{model_name} output:\n{out}")
    if err:
        logger.info(f"{model_name} errors:\n{err}")

    fid = None
    is_mean = None
    is_std = None

    for line in out.splitlines():
        line = line.strip()
        if line.startswith("FID:"):
            try:
                fid = float(line.split("FID:")[1])
            except Exception:
                pass
        if line.startswith("Inception Score"):
            try:
                parts = line.split(":", 1)[1].strip().split("±")
                is_mean = float(parts[0])
                is_std = float(parts[1])
            except Exception:
                pass

    return fid, is_mean, is_std


def main():
    fid_a, is_a_mean, is_a_std = run_eval(
        "Model A (Milestone 1 baseline)", MODEL_A_DIR, n=1000
    )
    fid_b, is_b_mean, is_b_std = run_eval(
        "Model B (Milestone 2 improved)", MODEL_B_DIR, n=1000
    )

    fid_c = is_c_mean = is_c_std = None
    if os.path.isdir(MODEL_C_DIR):
        fid_c, is_c_mean, is_c_std = run_eval(
            "Model C (Attention-finetuned, 4k steps)", MODEL_C_DIR, n=1000
        )
    else:
        logger.info(f"Model C directory not found: {MODEL_C_DIR}. Skipping Model C.")

    logger.info("\n=== Evaluation Complete ===\n")

    print("\n\n================ MODEL COMPARISON ================")
    print(
        f"Model A (Milestone 1): FID = {fid_a:.4f}, "
        f"IS = {is_a_mean:.4f} ± {is_a_std:.4f}"
    )
    print(
        f"Model B (Milestone 2): FID = {fid_b:.4f}, "
        f"IS = {is_b_mean:.4f} ± {is_b_std:.4f}"
    )
    if fid_c is not None:
        print(
            f"Model C (Attention-finetuned, 4k): FID = {fid_c:.4f}, "
            f"IS = {is_c_mean:.4f} ± {is_c_std:.4f}"
        )
    else:
        print("Model C (Attention-finetuned, 4k): not evaluated (missing directory)")
    print("==================================================\n")

    report = {
        "config": {
            "n_samples": 1000,
            "val_csv": VAL_CSV,
        },
        "modelA": {
            "name": "Milestone 1 baseline",
            "gen_dir": MODEL_A_DIR,
            "FID": fid_a,
            "IS_mean": is_a_mean,
            "IS_std": is_a_std,
        },
        "modelB": {
            "name": "Milestone 2 improved",
            "gen_dir": MODEL_B_DIR,
            "FID": fid_b,
            "IS_mean": is_b_mean,
            "IS_std": is_b_std,
        },
    }

    if fid_c is not None:
        report["modelC"] = {
            "name": "Attention-finetuned, 4k steps",
            "gen_dir": MODEL_C_DIR,
            "FID": fid_c,
            "IS_mean": is_c_mean,
            "IS_std": is_c_std,
        }

    report["timestamp"] = str(datetime.now())

    out_path = Path(REPORT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Saved model comparison report to {out_path}")


if __name__ == "__main__":
    main()
