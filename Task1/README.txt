README - TASK 1: Vietnamese Food Visual Question Answering on Kaggle
====================================================================

This folder contains the notebooks for Task 1: Vietnamese Food Visual Question Answering (VQA).
The project builds a Vietnamese food VQA dataset and compares four model configurations:
A1 = ViT + PhoBERT + LSTM decoder, A2 = ViT + PhoBERT + Transformer decoder,
B1 = BLIP zero-shot with Vietnamese-English translation bridge, and B2 = BLIP fine-tuning with LoRA.


1. FILE DESCRIPTION - ONE FILE PER LINE
=======================================

generate_qa_pairs.ipynb - Generates Vietnamese QA pairs from Vietnamese food images and metadata using Gemini API, batching images by account/model quota, validating/formatting answers, and exporting VN_Food_QAs_FINAL.json.

vqa-traina-lstm.ipynb - Trains Architecture A1 on Kaggle: frozen ViT image encoder + frozen PhoBERT question encoder + 512-d fusion + custom LSTM answer decoder, then saves LSTM checkpoints and training plots.

vqa-traina-transformer-decoder.ipynb - Trains Architecture A2 on Kaggle: frozen ViT image encoder + frozen PhoBERT question encoder + 512-d fusion + custom Transformer decoder, then saves Transformer checkpoints and metric plots.

vqa-trainb-phase1-translation.ipynb - Creates the translation bridge for Approach B by translating Vietnamese questions/answers into English using Helsinki-NLP translation models and exporting translation.json / translation split files.

vqa-trainb-phase2-zeroshot-finetuning-with-lora.ipynb - Runs Approach B: B1 BLIP zero-shot evaluation and B2 BLIP fine-tuning with LoRA adapters, then saves LoRA adapter artifacts and evaluation outputs.


2. EXPECTED KAGGLE ENVIRONMENT
==============================

Recommended Kaggle settings:
- Accelerator: GPU T4 or better.
- Internet: ON, because notebooks download HuggingFace models and may install small packages.
- Python: Kaggle default Python 3.x.
- Notebook type: Kaggle Notebook.

Main models downloaded during execution:
- google/vit-base-patch16-224-in21k for image encoding in A1/A2.
- vinai/phobert-base for Vietnamese question encoding in A1/A2.
- Helsinki-NLP/opus-mt-vi-en and Helsinki-NLP/opus-mt-en-vi for translation bridge.
- Salesforce/blip-vqa-base for B1/B2.


3. REQUIRED KAGGLE DATASET STRUCTURE
====================================

For training notebooks, attach a Kaggle Dataset that contains the processed VQA data.
The notebooks are written to look for this path by default:

/kaggle/input/datasets/khangnhut/vqa-vietnameses-food-dataset

Inside the dataset folder, the expected files/folders are:

train.json
val.json
test.json
vietnamese_food_QAs_pairs.json
vietnamese_food_images_2k5/

Minimum required files for A1/A2/B1/B2 training:
- train.json
- val.json
- test.json
- vietnamese_food_images_2k5/ containing image files referenced by the JSON rows.

If your Kaggle dataset path is different, edit the DATA_DIR or CANDIDATE_DATA_DIRS cell in each notebook.
For example, change:

DATA_DIR = Path('/kaggle/input/datasets/khangnhut/vqa-vietnameses-food-dataset')

to your actual Kaggle input path, such as:

DATA_DIR = Path('/kaggle/input/your-dataset-name')


4. RECOMMENDED RUN ORDER
========================

Use this order when running the full pipeline from data generation to training:

Step 1 - QA generation, optional if processed JSON already exists
---------------------------------------------------------------
Run generate_qa_pairs.ipynb only if you need to regenerate VN_Food_QAs_FINAL.json from raw images and metadata.
This notebook was originally Colab/Drive-oriented and uses Gemini API keys, so on Kaggle you must edit:
- BASE_DIR
- CSV_PATH
- IMAGE_DIR
- OUTPUT_DIR
- Gemini API account/key configuration

If you already have train.json, val.json and test.json, you can skip this notebook.

Step 2 - Train Architecture A1 LSTM
-----------------------------------
Open vqa-traina-lstm.ipynb on Kaggle.
Attach the processed dataset.
Enable GPU and Internet.
Run all cells from top to bottom.
Main outputs are saved under:

/kaggle/working/checkpoints_A

Expected outputs include:
- best LSTM checkpoint
- tokenizer/config files if exported by the notebook
- history JSON/CSV
- training and validation plots

Step 3 - Train Architecture A2 Transformer
------------------------------------------
Open vqa-traina-transformer-decoder.ipynb on Kaggle.
Use the same dataset and same Kaggle settings.
Run all cells from top to bottom.
Main outputs are saved under:

/kaggle/working/checkpoints_A

Expected outputs include:
- best Transformer checkpoint
- history JSON/CSV
- training and validation plots

Step 4 - Create translation bridge for BLIP
-------------------------------------------
Open vqa-trainb-phase1-translation.ipynb on Kaggle.
Attach the same processed dataset.
Enable GPU and Internet.
Run all cells from top to bottom.
Main outputs are saved under:

/kaggle/working/translation.json
/kaggle/working/translation_train.json
/kaggle/working/translation_val.json
/kaggle/working/translation_test.json
/kaggle/working/translation_cache.json
/kaggle/working/translation_config.json

Important notes:
- This step can take time because it translates questions and answers.
- For quick debugging, set MAX_ROWS_PER_SPLIT to a small number such as 200.
- For final training/evaluation, set MAX_ROWS_PER_SPLIT = None.
- After finishing, save the output as a Kaggle Dataset if you want to reuse it in Phase 2 without rerunning translation.

Step 5 - Run BLIP zero-shot and BLIP + LoRA fine-tuning
-------------------------------------------------------
Open vqa-trainb-phase2-zeroshot-finetuning-with-lora.ipynb on Kaggle.
Attach:
- the processed VQA dataset, and
- the translation outputs from Step 4 if the notebook expects translation.json from a previous run/dataset.

Enable GPU and Internet.
Run all cells from top to bottom.
Main outputs are saved under a Kaggle working/checkpoint directory, typically:

/kaggle/working/checkpoints_B_translation

Expected B2 LoRA outputs include:
- adapter_model.safetensors
- adapter_config.json
- processor_config.json
- tokenizer.json
- tokenizer_config.json
- evaluation files if generated by the notebook


5. QUICK RUN OPTION IF DATA IS ALREADY PROCESSED
================================================

If the processed dataset is already available, the shortest reproducible order is:

1. vqa-traina-lstm.ipynb
2. vqa-traina-transformer-decoder.ipynb
3. vqa-trainb-phase1-translation.ipynb
4. vqa-trainb-phase2-zeroshot-finetuning-with-lora.ipynb

Skip generate_qa_pairs.ipynb unless regenerating QA pairs from the raw image dataset.


6. IMPORTANT CONFIG VALUES
==========================

Architecture A notebooks:
- IMAGE_MODEL = google/vit-base-patch16-224-in21k
- TEXT_MODEL = vinai/phobert-base
- FEATURE_DIM = 512
- MAX_Q_LEN = 64
- MAX_A_LEN = 12
- BATCH_SIZE is usually 16
- LR is usually 3e-5 or 1e-4 depending on the notebook version
- OUT_DIR = /kaggle/working/checkpoints_A

Translation notebook:
- VI_EN_MODEL = Helsinki-NLP/opus-mt-vi-en
- EN_VI_MODEL = Helsinki-NLP/opus-mt-en-vi
- BATCH_SIZE_TRANSLATE = 16
- MAX_LEN = 96
- OUT_DIR = /kaggle/working

BLIP + LoRA notebook:
- Base model = Salesforce/blip-vqa-base
- B1 = zero-shot BLIP inference
- B2 = LoRA fine-tuning
- Typical LoRA setting: r = 8, alpha = 16, dropout = 0.05
- Typical training setting: small batch size with gradient accumulation, about 3 epochs


7. HOW TO SAVE OUTPUTS FROM KAGGLE
==================================

After each notebook finishes:

1. Check the right sidebar Output section in Kaggle.
2. Download output files directly, or create a Kaggle Dataset from /kaggle/working.
3. Recommended saved outputs:
   - checkpoints_A/ for A1/A2 checkpoints and plots.
   - translation.json and translation split files for BLIP phase 2.
   - checkpoints_B_translation/ for BLIP zero-shot / LoRA results.
   - adapter_model.safetensors and adapter_config.json for B2 LoRA.


8. COMMON ISSUES AND FIXES
==========================

Issue: FileNotFoundError for train.json, val.json or test.json.
Fix: Check the Kaggle input path and edit DATA_DIR / CANDIDATE_DATA_DIRS in the config cell.

Issue: Images cannot be found.
Fix: Ensure vietnamese_food_images_2k5/ is inside the attached Kaggle Dataset and image paths in JSON are relative/compatible.

Issue: HuggingFace model download fails.
Fix: Turn Internet ON in Kaggle Notebook settings and rerun the first import/model-loading cells.

Issue: CUDA out of memory.
Fix: Reduce BATCH_SIZE, reduce num_workers, restart session, or use gradient accumulation.

Issue: Translation takes too long.
Fix: Use MAX_ROWS_PER_SPLIT = 200 for debugging; use None only for the final full run.

Issue: LoRA training is slow or unstable.
Fix: Keep batch size small, use gradient accumulation, keep learning rate around 5e-5, and avoid full fine-tuning of BLIP.


9. FINAL NOTES
==============

- All notebooks should be run top-to-bottom in a fresh Kaggle session.
- Do not assume outputs from one notebook remain available after closing a Kaggle session; save important outputs as Kaggle Datasets.
- For final submission, include trained checkpoints and LoRA adapter files, especially adapter_model.safetensors for B2.
- If only inference/demo is needed, use saved model artifacts instead of retraining from scratch.
