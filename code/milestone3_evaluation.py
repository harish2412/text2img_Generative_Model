import os
import json
import logging
from datetime import datetime
from pathlib import Path
import subprocess


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


logger = setup_logger("milestone3")

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
VAL_CSV = "../data/manifests/coco_train1k.csv"
MODEL_A_DIR = "../sample_outputs/milestone1_output"
MODEL_B_DIR = "../sample_outputs/milestone2_output"
REPORT_PATH = "../sample_outputs/milestone3_report.json"


def run_eval(model_name, gen_dir):
    """
    Runs eval_fid_is.py as a subprocess and returns parsed FID/IS.
    """
    logger.info(f"\nRunning evaluation for {model_name}...")
    cmd = [
        "python",
        "eval_fid_is.py",
        "--val_csv", VAL_CSV,
        "--gen_dir", gen_dir,
        "--n", "100"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
            fid = float(line.split("FID:")[1])
        if line.startswith("Inception Score"):
            try:
                parts = line.split(":")[1].strip().split("±")
                is_mean = float(parts[0])
                is_std = float(parts[1])
            except:
                pass

    return fid, is_mean, is_std


# ----------------------------------------------------
# Evaluate Model A & B
# ----------------------------------------------------
fid_a, is_a_mean, is_a_std = run_eval("Model A (Milestone 1)", MODEL_A_DIR)
fid_b, is_b_mean, is_b_std = run_eval("Model B (Milestone 2)", MODEL_B_DIR)

logger.info("\n=== Evaluation Complete ===\n")

print("\n\n================ MODEL COMPARISON ================")
print(f"Model A (Milestone 1): FID = {fid_a:.4f}, IS = {is_a_mean:.4f} ± {is_a_std:.4f}")
print(f"Model B (Milestone 2): FID = {fid_b:.4f}, IS = {is_b_mean:.4f} ± {is_b_std:.4f}")
print("==================================================\n")

report = {
    "modelA": {
        "FID": fid_a,
        "IS_mean": is_a_mean,
        "IS_std": is_a_std
    },
    "modelB": {
        "FID": fid_b,
        "IS_mean": is_b_mean,
        "IS_std": is_b_std
    },
    "timestamp": str(datetime.now())
}

os.makedirs("../sample_outputs", exist_ok=True)
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=4)

logger.info(f"Saved Milestone 3 report to {REPORT_PATH}")
