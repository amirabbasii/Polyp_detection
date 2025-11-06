import os
import pandas as pd
from ultralytics import YOLO
from src.attention_module import add_attention_to_model


# ======================================================
# تنظیمات کلی آزمایش
# ======================================================
DATASET_PATH = "/content/VOC.yaml"   # مسیر داده‌ها
PRETRAINED_SIZE = "m"                # سایز مدل (n/s/m/l/x)
EPOCHS = 50                          # برای Colab می‌تونی کمتر بگذاری
IMGSZ = 640
BATCH = 16
SEEDS = [0, 1, 2]              # برای ارزیابی آماری
RUNS_DIR = "runs/detect"
RESULTS_FILE = "attention_statistical_results.csv"
SUMMARY_FILE = "attention_statistical_summary.csv"


def create_attention_model(pretrained_size='m'):
    model = YOLO(f'yolov8{pretrained_size}.pt')
    return add_attention_to_model(model)

# ======================================================
# تابع آموزش و ذخیره امن
# ======================================================
def train_attention_with_seed(seed):
    print(f"\n🚀 Training (attention) with random seed = {seed}\n")
    model = create_attention_model(PRETRAINED_SIZE)
    model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name=f"attention_seed_{seed}",
        seed=seed,
        patience=30,
        cos_lr=True,
        warmup_epochs=2.0,
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        close_mosaic=10,
        augment=True,
        mixup=0.1,
        copy_paste=0.1
    )


def extract_metrics(run_name):
    csv_path = os.path.join(RUNS_DIR, run_name, "results.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ No results file found for {run_name}")
        return None

    df = pd.read_csv(csv_path)
    last_row = df.iloc[-1]
    return {
        "seed": run_name.replace("attention_seed_", ""),
        "precision": last_row["metrics/precision(B)"],
        "recall": last_row["metrics/recall(B)"],
        "f1": last_row["metrics/f1(B)"],
        "map50": last_row["metrics/mAP50(B)"],
        "map5095": last_row["metrics/mAP50-95(B)"]
    }


def run_statistical_reliability():
    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE)
        completed_seeds = set(existing["seed"].astype(str).tolist())
        print(f"📂 Found existing results for seeds: {completed_seeds}")
    else:
        existing = pd.DataFrame()
        completed_seeds = set()

    all_results = []

    for seed in SEEDS:
        if str(seed) in completed_seeds:
            print(f"⏭️ Skipping seed {seed} (already done)")
            continue

        try:
            train_attention_with_seed(seed)
            metrics = extract_metrics(f"attention_seed_{seed}")
            if metrics:
                print(f"✅ Seed {seed} completed successfully.")
                all_results.append(metrics)

                df_partial = pd.concat([existing, pd.DataFrame(all_results)], ignore_index=True)
                df_partial.to_csv(RESULTS_FILE, index=False)
                print(f"💾 Partial results saved ({len(df_partial)} seeds done).")
            else:
                print(f"⚠️ No metrics for seed {seed}")

        except Exception as e:
            print(f"❌ Error during seed {seed}: {e}")
            continue

    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        mean = df.mean(numeric_only=True)
        std = df.std(numeric_only=True)

        summary = pd.DataFrame({
            "metric": mean.index,
            "mean": mean.values,
            "std": std.values,
            "mean ± std": [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean.values, std.values)]
        })
        summary.to_csv(SUMMARY_FILE, index=False)
        print("\n📊 Statistical summary (mean ± std):\n")
        print(summary[["metric", "mean ± std"]])
        print(f"\n✅ Summary saved to {SUMMARY_FILE}")


# ======================================================
if __name__ == "__main__":
    run_statistical_reliability()
