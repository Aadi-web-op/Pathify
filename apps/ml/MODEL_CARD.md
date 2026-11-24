---
language: [en, hi]
license: mit
tags: [career-guidance, riasec, holland-codes, retrieval, multilingual, onnx]
datasets: [onet-31.0, openpsychometrics-riasec]
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
---

# Pathify models

Three small models behind [Pathify](https://github.com/aadi-web-op/pathify), a career-guidance
app for Indian students from Class 9 to college. They are built to run on a CPU next to the
web API, and they are evaluated against simple baselines on held-out data. All numbers
below come from `reports/` in the repository and are produced by the training scripts.

## 1. Interest Reader

**Task.** Read what a student enjoys, in their own words (English, Hinglish or Hindi), and
estimate their six Holland interest scores (Realistic, Investigative, Artistic, Social,
Enterprising, Conventional), 0-100 each.

**Model.** `paraphrase-multilingual-MiniLM-L12-v2` (118M parameters) with mean pooling and a
two-layer regression head, fine-tuned end to end. Loss: MSE on the six scores plus a small
cross-entropy term on the strongest type.

**Training data (O*NET 31.0, CC BY 4.0).** Each of 923 occupations has an expert/ML-rated
interest profile on a 1-7 scale. Every task statement and job title of an occupation is
labelled with that profile, scaled to 0-1, and each task also appears in a first-person form
("I would like to …"). O*NET's illustrative activities, the 48 Open Psychometrics items and
Pathify's own 48 items add short single-type examples. 19,690 training sentences.

**Evaluation.**
- *Held-out occupations*: 10% of occupations (2,384 sentences) never seen in training.
  Metric: Pearson correlation per type (averaged) and how often the strongest predicted type
  matches the occupation's.
- *Student statements*: 72 sentences written by hand in how students talk (42 English,
  18 Hinglish, 12 Hindi), 12 per type. Metric: is the labelled type the top (or top two)?

| | Occupations: r | Occupations: top type | Students: top-1 | Students: top-2 | English | Hinglish | Hindi |
|---|---|---|---|---|---|---|---|
| Zero-shot (similarity to type descriptions) | 0.415 | 50.8% | 80.6% | 93.1% | 81.0% | 66.7% | 100% |
| Linear probe on frozen embeddings | 0.757 | 66.5% | 72.2% | 87.5% | 85.7% | 33.3% | 83.3% |
| **Fine-tuned** | **0.770** | **69.7%** | **84.7%** | **94.4%** | **92.9%** | 61.1% | 91.7% |
| Fine-tuned, embeddings frozen (ablation) | 0.768 | 69.7% | 84.7% | 93.1% | 92.9% | 61.1% | 91.7% |
| **Shipped ONNX** (pruned, int8 embeddings) | **0.769** | — | **83.3%** | — | — | — | — |

**What this shows.** Fine-tuning is what makes the scores themselves usable (r 0.42 → 0.77
on occupations it never saw). On short student statements the multilingual base model is
already a strong zero-shot ranker, and fine-tuning on English task text adds 4 points
overall but slightly erodes Hinglish and Hindi (n = 30, so a difference of one or two
sentences). Freezing the embedding layer to protect cross-lingual alignment changed nothing,
so the Hinglish gap comes from training only on English text, not from embedding drift: the
fix is Hinglish and Hindi training data (see "Next steps").

**Use in Pathify.** A live first guess under "Things you enjoy" during onboarding, clearly
labelled as a guess. The 30-item questionnaire, not this model, produces a student's results.

## 2. Pathfinder

**Task.** From six interest scores, estimate which of 12 broad fields people with similar
interests actually studied.

**Data (Open Psychometrics RIASEC, 2015-2018).** 145,828 people took a 48-item Holland test
and typed their field of study. We drop careless respondents (29,285 people ticked one of the
test's fake vocabulary words), people without a usable answer, and majors too vague to map
("general", "idk"). Free-text majors are mapped to fields with ordered keyword rules that
include Indian degree names (B.Tech, BCA, B.Com, CA, MBBS). 58,554 people remain, split
80/10/10, stratified.

**Model.** A 19-feature input (the six scores, each relative to the person's own average,
and their spread) into a two-hidden-layer MLP (64, 32), then temperature scaling (T = 1.05)
fitted on the validation set. Exported as JSON weights and run with numpy.

| On 5,857 held-out people | Top-1 | Top-3 | Log loss | Calibration error |
|---|---|---|---|---|
| Base rates | 24.7% | 52.8% | 2.212 | — |
| Holland congruence rule | 19.3% | 43.0% | 2.382 | 0.040 |
| Logistic regression | 33.6% | 63.5% | 1.972 | 0.018 |
| Gradient boosting | 33.8% | 63.3% | 1.974 | 0.015 |
| **MLP + temperature (shipped)** | **33.7%** | **63.0%** | **1.974** | **0.007** |
| *Indian respondents only (n = 113)* | *31.0%* | *63.7%* | *2.040* | *0.053* |

**What this shows.** The learned model beats the textbook congruence rule by 20 points of
top-3 accuracy, and it is well calibrated, so its percentages can be shown as percentages.
Three very different model families tie, which says the limit is the signal (interests alone
explain only part of what people study), not model capacity. That matches the research.

**Use in Pathify.** "What people like you studied" on the results page, shown as share and as
lift over the average ("2.1× the average"). Lift cancels most of the sample's skew: the test
was hosted on a psychology site, so psychology is over-represented.

## 3. Ksarth Retriever

**Task.** Given a student's question, find the career guides Ksarth should answer from and cite.

**Model.** The same base encoder, fine-tuned with in-batch negatives
(MultipleNegativesRankingLoss, no-duplicates batches) on 1,073 question-guide pairs generated
from each career's own catalogue facts in English, Hinglish and Hindi. 3 epochs on CPU.

**Evaluation.** 75 questions written by hand (44 English, 21 Hinglish, 10 Hindi), some
deliberately vague with several correct careers. Never used for training.

| | Hit@1 | Hit@3 | MRR@10 | Hindi hit@3 | Hinglish hit@3 |
|---|---|---|---|---|---|
| BM25 with aliases (previous search) | 76.0% | 78.7% | 0.789 | 10% | 85.7% |
| Base encoder | 80.0% | 94.7% | 0.872 | 100% | 81.0% |
| **Fine-tuned** | **92.0%** | **94.7%** | **0.944** | **100%** | **95.2%** |
| Equal-weight RRF of fine-tuned + BM25 | 81.3% | 88.0% | 0.862 | 50% | 90.5% |
| **Shipped int8 ONNX** (fine-tuned, pruned, quantized) | **92.0%** | **96.0%** | **0.948** | **100%** | **90.5%** |

**What this shows.** Fine-tuning moved the right career into first place far more often
(80% → 92%). Fusing with keyword search at equal weight made things worse, because BM25
can't read Devanagari, so the shipped search ranks by the encoder and uses BM25 only as a
second way to count a career as relevant (exact names such as "NEET-UG"). Small talk scores
at most 0.27 cosine similarity against real questions' median of 0.52, so a 0.30 gate keeps
"hi" and "tell me a joke" from producing citations.

**Known misses.** "janwaron ka doctor kaise bante hain" (how to become an animals' doctor)
ranks Doctor above Veterinarian: the model latches onto "doctor".

## Serving

`scripts/export_models.py` prunes the vocabulary to Latin and Devanagari pieces
(250,002 → 106,590 pieces; text in those scripts tokenizes identically, checked), exports to ONNX
and applies dynamic int8 quantization, then re-runs the evaluations on the shipped files.

| | Size | Max output change |
|---|---|---|
| Original (fp32, full vocabulary) | ~470 MB | — |
| Pruned, fp32 ONNX | 250 MB | 0.0 (identical) |
| **Retriever: pruned, full int8 (shipped)** | **63 MB** | 0.034; eval unchanged |
| **Interest Reader: pruned, int8 embeddings only (shipped)** | **127 MB** | 0.014; student top-1 83.3% vs 84.7% fp32 |

Full int8 cost the Interest Reader 4 points on student statements (its regression head is
sensitive to quantized matrix multiplications), so only its embedding table is quantized.

The API runs them with ONNX Runtime and the `tokenizers` library; PyTorch is only needed to train.

## Limits and responsible use

- **These are hints, not verdicts.** Pathify says so on every screen where a model speaks.
  Interests are one input into a career decision, next to ability, money, family and place.
- **Sample bias.** Pathfinder's data comes from an international online sample of people who
  chose to take a personality test (80,579 from the US, 2,216 from India). It describes what
  those people studied, not what Indian students should study.
- **Occupational stereotypes.** O*NET profiles describe US occupations as rated by experts and
  models; some careers look different in India.
- **Hinglish is the weakest language** for the Interest Reader (61%), and the hand-written
  eval sets are small (72 and 75 items), so differences of a few points are within noise.
- **No personal data** was used for training. Pathify does not train on its users' answers.

## Next steps

- Multilingual training data for the Interest Reader: Hinglish and Hindi paraphrases of the
  activity sentences. (Freezing embeddings was tried and did not help.)
- A larger hand-labelled eval set, written with students.
- Distilling Ksarth's grounded answers into a small open LLM (QLoRA on a free GPU), so the
  whole guide can run without a paid API.

## Data licences

- O*NET 31.0 Database, U.S. Department of Labor, Employment and Training Administration,
  CC BY 4.0. Pathify has modified the information. USDOL/ETA has not approved, endorsed or
  tested these modifications.
- Open Psychometrics RIASEC data (openpsychometrics.org/_rawdata), released for research use.
