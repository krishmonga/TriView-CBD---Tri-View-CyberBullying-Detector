#!/usr/bin/env python3
"""
Download and preprocess the Davidson et al. (2017) hate speech dataset.

Paper:  T. Davidson, D. Warmsley, M. Macy, and I. Weber,
        "Automated Hate Speech Detection and the Problem of Offensive Language,"
        Proc. ICWSM, vol. 11, no. 1, pp. 512-515, 2017.

Source: https://github.com/t-davidson/hate-speech-and-offensive-language

Original labels:
    0 = hate speech
    1 = offensive language
    2 = neither

Binary mapping (consistent with cyberbullying detection):
    hate speech (0)      → 1  (cyberbullying / abusive)
    offensive language (1) → 1  (cyberbullying / abusive)
    neither (2)           → 0  (non-cyberbullying)
"""

import os
import urllib.request
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/t-davidson/"
    "hate-speech-and-offensive-language/master/data/labeled_data.csv"
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_davidson")
OUT_FILE = os.path.join(OUT_DIR, "davidson_data.csv")


def download_and_prepare():
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.isfile(OUT_FILE):
        print(f"Already exists: {OUT_FILE}")
        df = pd.read_csv(OUT_FILE)
        print(f"  {len(df)} rows")
        return

    raw_path = os.path.join(OUT_DIR, "labeled_data_raw.csv")
    print(f"Downloading Davidson et al. (2017) dataset ...")
    urllib.request.urlretrieve(URL, raw_path)
    print(f"  Saved raw: {raw_path}")

    df = pd.read_csv(raw_path)
    print(f"  Raw rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Class distribution:")
    print(f"    0 (hate speech):        {(df['class'] == 0).sum()}")
    print(f"    1 (offensive language):  {(df['class'] == 1).sum()}")
    print(f"    2 (neither):             {(df['class'] == 2).sum()}")

    df["oh_label"] = df["class"].apply(lambda c: 0 if c == 2 else 1)
    df = df.rename(columns={"tweet": "Text"})

    out = df[["Text", "oh_label"]]
    out.to_csv(OUT_FILE, index=False)
    print(f"\n  Binary distribution:")
    print(f"    1 (abusive):      {(out['oh_label'] == 1).sum()}")
    print(f"    0 (non-abusive):  {(out['oh_label'] == 0).sum()}")
    print(f"\n  Saved: {OUT_FILE}  ({len(out)} rows)")


if __name__ == "__main__":
    download_and_prepare()
