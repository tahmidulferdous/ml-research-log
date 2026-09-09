# Notebooks

131 Kaggle notebooks that are not tied to a logged Weights & Biases project:
exploratory work, one-off probes, coursework and practice. Notebooks that produced
the runs in `projects/` live with their project instead.

Each notebook is stored as it was pulled from Kaggle, alongside its
`kernel-metadata.json`. Notebooks that Kaggle auto-named have been given a readable
title and a one-line description after review; the directory name is that title.

31 further notebooks were left out: empty stubs, unmodified Kaggle starter
boilerplate, and one copied tutorial that carried a third party's credential.


## leakage study probes (16)

| Notebook | Last run | What it is |
| --- | --- | --- |
| [04 blocked cv](leakage-study-probes/04-blocked-cv/) | 2026-09-08 |  |
| [00 dataset audit](leakage-study-probes/00-dataset-audit/) | 2026-09-07 |  |
| [01 split controls](leakage-study-probes/01-split-controls/) | 2026-09-07 |  |
| [01b null distribution](leakage-study-probes/01b-null-distribution/) | 2026-09-07 |  |
| [02 identity probe](leakage-study-probes/02-identity-probe/) | 2026-09-07 |  |
| [11 proxy validation](leakage-study-probes/11-proxy-validation/) | 2026-09-07 |  |
| [11b leakage dose](leakage-study-probes/11b-leakage-dose/) | 2026-09-07 |  |
| [12 luna calibration](leakage-study-probes/12-luna-calibration/) | 2026-09-07 |  |
| [ccal multi seed robustness probe](leakage-study-probes/ccal-multi-seed-robustness-probe/) | 2026-05-12 | Multi-seed robustness evaluation of honest sequential 70/20/10 split on CCAL architecture (Plan B). |
| [pretrained_model_1C_train](leakage-study-probes/pretrained-model-1c-train/) | 2026-04-01 |  |
| [base_ccal_v3](leakage-study-probes/base-ccal-v3/) | 2026-03-29 |  |
| [exp_1a_paper_exact_train](leakage-study-probes/exp-1a-paper-exact-train/) | 2026-03-29 |  |
| [preprocessing](leakage-study-probes/preprocessing/) | 2026-03-26 |  |
| [train](leakage-study-probes/train/) | 2026-03-26 |  |
| [rectified_flow](leakage-study-probes/rectified-flow/) | 2026-02-27 |  |
| [dsmat-pico-nano-fixed](leakage-study-probes/dsmat-pico-nano-fixed/) | 2026-02-08 |  |

## lung cancer classification (23)

| Notebook | Last run | What it is |
| --- | --- | --- |
| [06 baselines](lung-cancer-classification/06-baselines/) | 2026-09-08 |  |
| [07 rf train clean](lung-cancer-classification/07-rf-train-clean/) | 2026-09-08 |  |
| [01c random image null](lung-cancer-classification/01c-random-image-null/) | 2026-09-07 |  |
| [rectified_flow_tpu](lung-cancer-classification/rectified-flow-tpu/) | 2026-04-28 |  |
| [lung_effnet_full_experiment](lung-cancer-classification/lung-effnet-full-experiment/) | 2026-04-22 |  |
| [aug_diff](lung-cancer-classification/aug-diff/) | 2026-02-22 |  |
| [newFresh start](lung-cancer-classification/newfresh-start/) | 2026-02-22 |  |
| [dataAug_IQ-OTH](lung-cancer-classification/dataaug-iq-oth/) | 2026-02-21 |  |
| [DSMAT_BRAIN](lung-cancer-classification/dsmat-brain/) | 2026-02-11 |  |
| [lungDsmamba](lung-cancer-classification/lungdsmamba/) | 2026-02-11 |  |
| [dsmat_walkthrough](lung-cancer-classification/dsmat-walkthrough/) | 2026-02-10 |  |
| [MambaVision_Toy_Implementation](lung-cancer-classification/mambavision-toy-implementation/) | 2026-02-10 |  |
| [newtest1](lung-cancer-classification/newtest1/) | 2026-02-10 |  |
| [lung cancer multi dataset cnn](lung-cancer-classification/lung-cancer-multi-dataset-cnn/) | 2026-02-10 | Deep CNN benchmarking and evaluation metrics across advanced imaging lung cancer datasets. |
| [DSMaT_MambaVision_Full_Family](lung-cancer-classification/dsmat-mambavision-full-family/) | 2026-02-08 |  |
| [dataset_mine](lung-cancer-classification/dataset-mine/) | 2026-01-24 |  |
| [base_ccal_architecture](lung-cancer-classification/base-ccal-architecture/) | 2025-10-31 |  |
| [ccal full experiment pipeline](lung-cancer-classification/ccal-full-experiment-pipeline/) | 2025-09-30 | Full CCAL (Contrastive Cross-Attention Learning) model training and evaluation on IQ-OTHNCCD dataset. |
| [iqothnccd class distribution audit](lung-cancer-classification/iqothnccd-class-distribution-audit/) | 2025-09-29 | Dataset audit and class imbalance inspection on IQ-OTHNCCD lung cancer CT slices. |
| [testing_base_ccal](lung-cancer-classification/testing-base-ccal/) | 2025-09-28 |  |
| [ccal_dataset](lung-cancer-classification/ccal-dataset/) | 2025-09-27 |  |
| [cnn+attention](lung-cancer-classification/cnn-attention/) | 2025-09-25 |  |
| [Vit-Base Implementation](lung-cancer-classification/vit-base-implementation/) | 2025-04-09 |  |

## generative models for ct (6)

| Notebook | Last run | What it is |
| --- | --- | --- |
| [rectified flow fid evaluation](generative-models-for-ct/rectified-flow-fid-evaluation/) | 2026-04-28 | Evaluation script computing Fréchet Inception Distance (FID) on synthetic CT images from Rectified Flow. |
| [diff_aug_inference](generative-models-for-ct/diff-aug-inference/) | 2026-02-23 |  |
| [img_gen](generative-models-for-ct/img-gen/) | 2025-05-11 |  |
| [ddpm ct image synthesis](generative-models-for-ct/ddpm-ct-image-synthesis/) | 2024-12-23 | Denoising Diffusion Probabilistic Model (DDPM) implementation in PyTorch for CT image synthesis. |
| [diffusers pipeline experiments](generative-models-for-ct/diffusers-pipeline-experiments/) | 2024-12-23 | Hugging Face Diffusers pipeline fine-tuning and synthetic image sampling. |
| [autoencoder and diffusion ct](generative-models-for-ct/autoencoder-and-diffusion-ct/) | 2024-11-10 | Deep generative modeling walkthrough comparing variational autoencoders and diffusion on medical imaging. |

## gans (6)

| Notebook | Last run | What it is |
| --- | --- | --- |
| [Comparison DCGAN](gans/comparison-dcgan/) | 2026-01-26 |  |
| [Depthwise Conv GAN](gans/depthwise-conv-gan/) | 2026-01-26 |  |
| [Introduction to CNN Keras - 0.997 (top 6%)](gans/introduction-to-cnn-keras-0-997-top-6/) | 2025-01-15 |  |
| [dcgan dogs vs cats](gans/dcgan-dogs-vs-cats/) | 2024-09-04 | PyTorch DCGAN implementation training on Dogs vs Cats dataset. |
| [Honey Bee health detection with CNN](gans/honey-bee-health-detection-with-cnn/) | 2024-08-22 |  |
| [imagenet scraping and wgan](gans/imagenet-scraping-and-wgan/) | 2024-03-13 | ImageNet synset URL scraping and WGAN image generation experiments. |

## object detection (8)

| Notebook | Last run | What it is |
| --- | --- | --- |
| [iiuc-idcard-v2=0.38map95](object-detection/iiuc-idcard-v2-0-38map95/) | 2026-07-16 |  |
| [yolov8 iiuc idcard training](object-detection/yolov8-iiuc-idcard-training/) | 2026-01-25 | Custom YOLOv8 PyTorch dataset pipeline and training for IIUC student ID card detection. |
| [idcard image2image diffusion](object-detection/idcard-image2image-diffusion/) | 2025-06-12 | Image-to-image diffusion pipeline experiment using IIUC ID card image samples. |
| [tesseract ocr idcard parsing](object-detection/tesseract-ocr-idcard-parsing/) | 2025-05-19 | Tesseract OCR and image preprocessing pipeline extracting text from student ID cards. |
| [bert_practice](object-detection/bert-practice/) | 2025-05-11 |  |
| [53BM-face](object-detection/53bm-face/) | 2024-10-30 |  |
| [iiuc-idcard-v1](object-detection/iiuc-idcard-v1/) | 2024-10-16 |  |
| [iiuc bus counter](object-detection/iiuc-bus-counter/) | 2024-10-14 |  |

## language models (13)

| Notebook | Last run | What it is |
| --- | --- | --- |
| [GTP-2](language-models/gtp-2/) | 2025-10-14 |  |
| [banglagpt inference pipeline](language-models/banglagpt-inference-pipeline/) | 2025-10-14 | Text generation inference pipeline using fine-tuned BanglaGPT model with Hugging Face transformers. |
| [Attention](language-models/attention/) | 2025-10-08 |  |
| [hw_benchmark](language-models/hw-benchmark/) | 2025-07-05 |  |
| [blip2 vision language inference](language-models/blip2-vision-language-inference/) | 2025-04-05 | Vision-language image captioning and visual question answering using Salesforce BLIP-2. |
| [transformer self attention from scratch](language-models/transformer-self-attention-from-scratch/) | 2025-04-05 | PyTorch implementation of multi-head self-attention and positional encoding from scratch. |
| [bangla text classification lstm](language-models/bangla-text-classification-lstm/) | 2025-04-05 | PyTorch LSTM network for Bangla text and document classification. |
| [bangla transformer pretraining](language-models/bangla-transformer-pretraining/) | 2025-04-05 | Custom Transformer model pretraining loop on Bangla literary corpus with custom tokenizer. |
| [GPT from scratch](language-models/gpt-from-scratch/) | 2025-03-04 |  |
| [Embeddings](language-models/embeddings/) | 2025-03-01 |  |
| [transformer_scratch](language-models/transformer-scratch/) | 2025-02-24 |  |
| [hands on ml nlp rnn transformers](language-models/hands-on-ml-nlp-rnn-transformers/) | 2024-11-10 | Hands-On Machine Learning Chapter 16 NLP tutorial on Char-RNN, sentiment analysis, and Transformer architectures. |
| [practicing classification  ](language-models/practicing-classification/) | 2023-12-10 |  |

## coursework and practice (49)

| Notebook | Last run | What it is |
| --- | --- | --- |
| [base_paper_v1](coursework-and-practice/base-paper-v1/) | 2026-04-25 |  |
| [knowledge distillation](coursework-and-practice/knowledge-distillation/) | 2025-12-10 |  |
| [random_practice](coursework-and-practice/random-practice/) | 2025-06-12 |  |
| [credit card fraud eda](coursework-and-practice/credit-card-fraud-eda/) | 2025-05-31 | Exploratory data analysis and class distribution check on Kaggle Credit Card Fraud Detection dataset. |
| [mscad medical dataset eda](coursework-and-practice/mscad-medical-dataset-eda/) | 2025-05-05 | Exploratory data analysis, shape inspection, and summary statistics of MSCAD cardiovascular dataset. |
| [tpu flower classification tensorflow](coursework-and-practice/tpu-flower-classification-tensorflow/) | 2024-09-16 | TensorFlow / Keras convolutional network training on TPU for Kaggle Flower Classification. |
| [lets Predict NaN Value](coursework-and-practice/lets-predict-nan-value/) | 2024-09-01 |  |
| [first time pretrained 97%](coursework-and-practice/first-time-pretrained-97/) | 2024-08-22 |  |
| [modernTitanic](coursework-and-practice/moderntitanic/) | 2024-08-17 |  |
| [Coffee or Beer in London - Your Choice!](coursework-and-practice/coffee-or-beer-in-london-your-choice/) | 2024-07-30 |  |
| [tabular playground random forest](coursework-and-practice/tabular-playground-random-forest/) | 2024-07-29 | Random Forest baseline with cross-validation predictions for Tabular Playground Series (Jan 2021). |
| [meta kaggle competitions analysis](coursework-and-practice/meta-kaggle-competitions-analysis/) | 2024-07-29 | Analysis of Meta-Kaggle competition metadata and evaluation metric distributions. |
| [spaceship titanic summary](coursework-and-practice/spaceship-titanic-summary/) | 2024-07-29 | Pandas describe and info check on Spaceship Titanic train and test sets. |
| [first time CNN?](coursework-and-practice/first-time-cnn/) | 2024-07-10 |  |
| [AlexNet CNN Architecture on Tensorflow (beginner)](coursework-and-practice/alexnet-cnn-architecture-on-tensorflow-beginner/) | 2024-03-13 |  |
| [Titanic - start of a journey around data world ](coursework-and-practice/titanic-start-of-a-journey-around-data-world/) | 2024-02-15 |  |
| [pro script](coursework-and-practice/pro-script/) | 2024-02-13 |  |
| [Classification Metrics for Beginners](coursework-and-practice/classification-metrics-for-beginners/) | 2024-02-10 |  |
| [first time tensorflow](coursework-and-practice/first-time-tensorflow/) | 2024-02-01 |  |
| [house prices data exploration](coursework-and-practice/house-prices-data-exploration/) | 2024-01-07 | Feature exploration and initial preprocessing for the Kaggle House Prices competition dataset. |
| [EDA on Titanic](coursework-and-practice/eda-on-titanic/) | 2023-12-28 |  |
| [spaceship titanic eda](coursework-and-practice/spaceship-titanic-eda/) | 2023-12-25 | Comprehensive Seaborn visualizations and categorical analysis for Spaceship Titanic competition. |
| [house prices regression modeling](coursework-and-practice/house-prices-regression-modeling/) | 2023-12-09 | House Prices regression workflow with missing value handling, feature preprocessing, and baseline predictions. |
| [numpy random sampling practice](coursework-and-practice/numpy-random-sampling-practice/) | 2023-11-19 | NumPy random sampling and matrix manipulation exercises. |
| [house prices minimal check](coursework-and-practice/house-prices-minimal-check/) | 2023-11-12 | Minimal loading and column inspection for Kaggle House Prices dataset. |
| [variable unique values analysis before EDA](coursework-and-practice/variable-unique-values-analysis-before-eda/) | 2023-11-12 |  |
| [NBA Trends](coursework-and-practice/nba-trends/) | 2023-11-03 |  |
| [billionaires statistics eda](coursework-and-practice/billionaires-statistics-eda/) | 2023-10-31 | Exploratory analysis and demographic breakdown of the global billionaires statistics dataset. |
| [spaceship titanic feature check](coursework-and-practice/spaceship-titanic-feature-check/) | 2023-10-29 | Initial feature loading, data types inspection, and target distribution for Spaceship Titanic. |
| [Bar Chart Race MotoGP](coursework-and-practice/bar-chart-race-motogp/) | 2023-10-26 |  |
| [spotify streams eda](coursework-and-practice/spotify-streams-eda/) | 2023-10-26 | Pandas data loading and summary analysis of most streamed songs on Spotify. |
| [Titanic - The Only Notebook You Need To See](coursework-and-practice/titanic-the-only-notebook-you-need-to-see/) | 2023-10-23 |  |
| [Basic](coursework-and-practice/basic/) | 2023-10-22 |  |
| [Flexing Pandas Skill](coursework-and-practice/flexing-pandas-skill/) | 2023-10-12 |  |
| [codecademy ex: 1 movie statistic](coursework-and-practice/codecademy-ex-1-movie-statistic/) | 2023-07-01 |  |
| [titanic.py](coursework-and-practice/titanic-py/) | 2023-04-13 |  |
| [Easiest EDA](coursework-and-practice/easiest-eda/) | 2022-04-23 |  |
| [titanic666](coursework-and-practice/titanic666/) | 2022-04-10 |  |
| [tabular data inspection](coursework-and-practice/tabular-data-inspection/) | 2022-04-08 | Basic Pandas data loading and summary info inspection for tabular data. |
| [titanic data preprocessing](coursework-and-practice/titanic-data-preprocessing/) | 2022-04-07 | Quick feature loading and null value check for the Titanic classification dataset. |
| [Short Code EDA](coursework-and-practice/short-code-eda/) | 2021-09-02 |  |
| [titanic random forest classification](coursework-and-practice/titanic-random-forest-classification/) | 2021-08-03 | Data preprocessing, feature engineering, and Random Forest classification on Kaggle Titanic dataset. |
| [california housing regression eda](coursework-and-practice/california-housing-regression-eda/) | 2021-07-10 | Exploratory data analysis, train-test splitting, and feature inspection on California Housing dataset. |
| [calculus autodiff tutorial](coursework-and-practice/calculus-autodiff-tutorial/) | 2021-06-29 | Mathematical walkthrough of numerical and symbolic automatic differentiation. |
| [geometry slope animations calculus](coursework-and-practice/geometry-slope-animations-calculus/) | 2021-06-29 | Interactive Matplotlib animations illustrating derivative secant slopes and differentiability. |
| [hands on ml ch1 landscape](coursework-and-practice/hands-on-ml-ch1-landscape/) | 2021-06-29 | Hands-On Machine Learning Chapter 1 walkthrough replicating OECD Better Life and GDP linear regression. |
| [geometry slope animations calculus](coursework-and-practice/geometry-slope-animations-calculus/) | 2021-06-29 | Interactive Matplotlib animations illustrating derivative secant slopes and differentiability. |
| [geometry slope animations calculus](coursework-and-practice/geometry-slope-animations-calculus/) | 2021-06-29 | Interactive Matplotlib animations illustrating derivative secant slopes and differentiability. |
| [geometry slope animations calculus](coursework-and-practice/geometry-slope-animations-calculus/) | 2021-06-29 | Interactive Matplotlib animations illustrating derivative secant slopes and differentiability. |

## miscellaneous (10)

| Notebook | Last run | What it is |
| --- | --- | --- |
| [wandb api run history exporter](miscellaneous/wandb-api-run-history-exporter/) | 2026-09-06 | Utility script using wandb.Api() to export run metrics history to CSV. |
| [dataset_preprocessing](miscellaneous/dataset-preprocessing/) | 2026-03-24 |  |
| [medmnist](miscellaneous/medmnist/) | 2026-02-05 |  |
| [DDP usage](miscellaneous/ddp-usage/) | 2026-01-01 |  |
| [TPU_tesing](miscellaneous/tpu-tesing/) | 2025-10-06 |  |
| [megentaRT](miscellaneous/megentart/) | 2025-06-21 |  |
| [clip zero shot image probing](miscellaneous/clip-zero-shot-image-probing/) | 2025-05-12 | OpenAI CLIP model zero-shot image-text similarity scoring probe. |
| [GP2_from_scratch](miscellaneous/gp2-from-scratch/) | 2025-04-02 |  |
| [gpu-test](miscellaneous/gpu-test/) | 2024-11-22 |  |
| [trying again](miscellaneous/trying-again/) | 2022-10-08 |  |
