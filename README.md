# 🌾 Crop Yield + Climate AI

Fine-tuning Mistral 7B to analyze agricultural yield patterns from 57 years of global crop and climate data.


---

## What This Project Does

This project takes 57 years of real UN crop yield and NASA climate data, converts it into instruction-output training pairs, and fine-tunes a Mistral 7B language model to generate structured agricultural analysis on demand.

You give the model a country, year, and climate conditions. It writes you a full agronomic report — yield benchmarking, CO2 and temperature impact, and historical context — the way a trained agricultural analyst would.

---

## Project Structure

```
crop-climate-ai/
│
├── data/
│   ├── crop_climate_raw.csv          # Raw merged dataset (9,401 rows)
│   └── crop_climate_clean.csv        # Cleaned + enriched dataset
│
├── dataset/
│   ├── crop_climate_instructions.jsonl  # All 9,401 instruction-output pairs
│   ├── train.jsonl                      # 8,460 training examples
│   └── val.jsonl                        # 941 validation examples
│
├── notebooks/
│   └── Step3_FineTune_Colab.ipynb    # Fine-tuning notebook (run on Colab T4)
│
└── README.md
```

---

## Data Sources

| Dataset | Source | What it adds |
|---|---|---|
| Crop yields (9 crops) | FAOSTAT via OWID / TidyTuesday | Wheat, rice, maize, soybeans, potatoes, barley, beans, cassava, bananas — tonnes per hectare |
| Cereal yield + mechanization | FAO | Overall cereal kg/ha, tractors per 100 sq km |
| Temperature anomaly | NASA GISS | Global surface temp anomaly vs 1951–1980 baseline |
| CO2 concentration | Mauna Loa (NOAA) | Atmospheric CO2 in ppm, annual average |

**Merged dataset:** 9,401 rows × 23 columns — 184 countries, 1961–2018

---

## Pipeline Overview

```
Step 1 — Data Collection & Merging
  4 public datasets → 1 clean CSV (9,401 rows, 184 countries, 57 years)

Step 2 — Instruction-Output Formatting
  Each CSV row → structured question-answer pair in HF chat format
  9,401 pairs → split into train (8,460) / val (941)

Step 3 — Fine-Tuning on Colab T4 (free)
  Mistral 7B Instruct v0.3 + QLoRA (4-bit) + Unsloth
  ~70 min on free Google Colab T4 GPU

Step 4 — Deployment
  LoRA adapter pushed to Hugging Face Hub
```

---

## Training Setup

```python
Base model:   mistralai/Mistral-7B-Instruct-v0.3
Method:       QLoRA (4-bit quantization via bitsandbytes)
Framework:    Unsloth (2–5x faster, 80% less VRAM than standard)
LoRA rank:    r=8
GPU:          Google Colab T4 (free tier, 15GB VRAM)
Epochs:       1
Batch size:   1 (gradient accumulation = 8 → effective batch = 8)
Seq length:   1024 tokens
LR:           2e-4 (cosine scheduler)
Optimizer:    AdamW 8-bit
Train time:   ~70 minutes
```

---

## Sample Output

**Input prompt:**
```
Country: Pakistan | Year: 2010
CO2: 390 ppm (High CO2 era) | Temp anomaly: +0.72°C (Warm year)
Crops: wheat, rice, maize, potatoes
```

**Model output:**
```
## Agricultural Analysis: Pakistan, 2010

### Crop Yield Performance
- Wheat: 2.55 t/ha — good
- Rice: 3.06 t/ha — good
- Maize: 3.81 t/ha — good
- Potatoes: 22.68 t/ha — good

### Climate Context
- CO2: 390 ppm — elevated greenhouse conditions
- Temp anomaly: +0.72°C — moderate warming year

### Key Observations
- Moderate warming may have extended growing seasons in cooler highland
  regions while adding heat stress to lowland crops.
- C3 crops like wheat and rice may show modest CO2 fertilization effects
  at this concentration level.
- Strongest performer: Potatoes at 22.68 t/ha.
```

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/AliBux/crop-climate-ai
cd crop-climate-ai
```

### 2. Reproduce the dataset (optional — files already included)
```bash
pip install pandas requests
python data/build_dataset.py
```

### 3. Run fine-tuning on Google Colab
- Open `notebooks/Step3_FineTune_Colab.ipynb` in Google Colab
- Set runtime to **GPU → T4** (free)
- Upload `train.jsonl` and `val.jsonl`
- Fill in your Hugging Face token and username in Cell 10
- Run all cells (~70 minutes)

### 4. Use the deployed model
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, "AliBux/crop-climate-mistral-7b")
```

Full inference example in the [model card](https://huggingface.co/AliBux/crop-climate-mistral-7b).

---

## Key Decisions & Why

**Why Mistral 7B?** Small enough to train on a free T4 GPU, strong enough to write coherent analytical text. Good balance of quality and cost.

**Why QLoRA?** Full fine-tuning of 7B parameters needs 40GB+ VRAM. QLoRA (4-bit) reduces that to ~5GB — making free Colab T4 viable.

**Why Unsloth?** Runs 2–5x faster than standard HuggingFace training with 80% less VRAM. Essential for staying within Colab's 90-minute session limit.

**Why 1 epoch?** Two epochs (~120 min) risked hitting Colab's free session timeout. One epoch (~70 min) trains safely and avoids overfitting on a structured dataset like this.

**Why r=8 not r=16?** Lower rank = less VRAM. For a structured analytical task like this, r=8 captures enough pattern without pushing the T4 to its limit.

---

## Limitations

- Climate signals are global averages, not country-specific local weather
- Data ends at 2018 — model is not suitable for recent yield predictions
- Outputs are qualitative text analysis, not numeric predictions
- Best results with prompts that match the training format

---

## Author

**AliBux** — AI/ML developer focused on LLM fine-tuning, RAG systems, and applied machine learning.

- 🤗 Hugging Face: [huggingface.co/AliBux](https://huggingface.co/AliBux)
- 💼 Available for freelance AI/ML projects on Upwork
