# crop-climate-mistral

Fine-tuned Mistral-7B-Instruct on a crop-climate dataset using QLoRA via Unsloth.

The model takes structured agricultural inputs — country, year, CO2 concentration,
temperature anomaly, and crop list — and generates a yield analysis with climate context.

**Model on HuggingFace:** https://huggingface.co/AliBux/crop-climate-mistral-7b

## Training
- Base: mistralai/Mistral-7B-Instruct-v0.3 (4-bit QLoRA)
- LoRA rank: 16, target modules: q/k/v/o/gate/up/down projections
- Dataset: 1,148 crop-climate examples after length filtering
- Best checkpoint: step 200, eval loss 0.100

## Files
- `crop_climate_finetune.ipynb` — full training notebook (runs on free Colab T4)
- `train.jsonl` / `val.jsonl` — dataset (add these or link your HF dataset)
