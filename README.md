# Ihema Team · Project 5 — Locality-Aware Agriculture AI for Nigeria

> Advice that reflects *a* farm, in *Nigeria*, in words *Nigerian Farmers* can understand and says so when it isn't sure.

This repository holds the answer-generation component of an agriculture AI model for Nigeria. This README is for anyone arriving here: it explains **who we are building for, what "good" means, how the solution is designed to deliver it, what exists today, and how to contribute to building and fine-tuning it.**

---

## 1. The problem

Nigeria's climate and topography change sharply over short distances — humid south, semi-arid north, highland plateaus. The people who depend on that land are just as varied: geologists, small-, mid- and large-scale farmers, livestock rearers, cash-crop growers and local agricultural merchants.

Most agritech tools are built on **global data** that smooths over local differences in weather, soil and terrain. The consequences are poorly timed decisions (when to plant, treat, irrigate, store or sell) and avoidable losses in yield and income. At scale, those losses threaten livelihoods and national food security.

## 2. The vision

An agriculture AI model that gives **more accurate, locality-aware insight** to everyone in that chain, built on four commitments:

| Commitment | What it means for the product |
|---|---|
| **Locality-aware** | Advice is conditioned on where the user actually is (agro-zone, soil, terrain, climate), not on a generic average. |
| **Accurate** | Recommendations (rates, timings, practices) are grounded in vetted source material, never invented. |
| **Honest about uncertainty** | When evidence is thin, conflicting or out of scope, the answer says so instead of sounding confident. |
| **Accessible** | Usable across different levels of comprehension and literacy: plain wording, short answers, no jargon wall. |

A secondary aim from the problem statement: the same locality-aware approach should help **geologists** discover the mineral prospects of a region, while farmers make better-informed decisions and raise their harvests.

## 3. From principles to engineering decisions

Every commitment should map to something concrete in the system. This table is the contract between the vision and the code.

| Commitment | How we implement it | Status |
|---|---|---|
| Locality-aware | Questions carry metadata (`topic`, `crop`, `agro_zone`). Retrieval uses these as soft signals so that same-zone, same-crop evidence is preferred; the model prompt starts with the same metadata. | **Built** (coarse: 3 zones) |
| Accurate | The model never answers from its weights alone: each prompt includes retrieved factsheet sentences and similar, already-verified Q&A. A retrieval-only path exists as a safe fallback. | **Built** |
| Accurate (no regressions) | A fine-tuned model replaces the retrieval answer only if it beats it on held-out data by a margin; any failure falls back to retrieval. | **Built** |
| Honest about uncertainty | Planned: use neighbour agreement (how much the retrieved answers disagree, and how weak the best match is) as a confidence signal; below a threshold the answer is hedged or deferred to a local extension officer. | **Planned** |
| Accessible | Planned: reading-level-controlled answer style, short outputs, and local-language / voice delivery. Short, single-sentence answers are already the output format. | **Partly built** |
| Geologist / mineral-prospect track | Needs different data (geological surveys, maps) and its own evaluation. Not started. | **Not started** |

## 4. The solution

### 4.1 Why a small language model, grounded in retrieval

- **Grounding beats memory.** General LLMs hallucinate application rates, timings and "local practices". We keep the model on a short leash: it writes from supplied evidence.
- **Small models are deployable.** A fine-tuned small LM (2–4B parameters) is feasible to adapt cheaply and to run in constrained environments; this is the realistic path to reaching users with limited connectivity or budget.
- **Retrieval carries locality.** Locality lives in the evidence that is retrieved, so improving the corpus (more Nigerian, zone-specific material) improves answers without retraining from scratch.

### 4.2 The pipeline

```
 question + locality metadata (topic, crop, agro_zone)
            │
            ▼
   ┌───────────────────┐   hybrid word + character TF-IDF, soft metadata bonuses
   │ 1. Retrieve        │──► similar solved questions  +  best factsheet sentences
   └───────────────────┘
            │
            ├──────────────► 2a. Retrieval + MBR answer (CPU, always available)
            │                    picks the candidate closest, on average, to what
            │                    trusted neighbours say (a weighted medoid)
            ▼
   ┌───────────────────┐
   │ 2b. RAG + LoRA SLM │──► generates an answer from metadata + factsheet notes
   └───────────────────┘      + 2 similar solved examples
            │
            ▼
   3. Selection: keep the LLM answer only if it clearly beats 2a on held-out data
            │
            ▼
   4. Validated output (schema, order, non-empty) — with 2a as the fallback
```

Key choices in 2b: LoRA adapters on an fp16/bf16 base model (no quantisation); loss on **answer tokens only**; identical prompt construction for training and inference (training rows use leave-one-out neighbours so the model never sees its own answer); decoding settings chosen on held-out data.

### 4.3 How we measure progress — and the limits of that measure

We use the Kaggle benchmark `agriculture-climate-slm` as a **measurable testbed** for the generation step. It scores mean Levenshtein distance to reference answers.

That metric is a useful *proxy*, but it measures **textual similarity, not truth, safety or usability**. A bland answer that resembles the reference can beat a correct one that is phrased differently. So:

- Use the benchmark to compare *generation strategies* quickly and reproducibly.
- Do **not** treat a good score as proof the advice is right for a Nigerian farmer. Section 6 lists the additional checks required before anything is put in front of users.

## 5. Where we are, and where this goes

**Done**
- Retrieval + MBR answer selection, tuned by cross-validation on the real metric.
- Retrieval-augmented LoRA fine-tuning pipeline with automatic fallback.
- Validated, reproducible notebook (`agriculture-climate-slm-optimized.ipynb`).

**Not yet verified:** the GPU fine-tuning stage has not been run end to end, and no scores on the real data have been recorded here. Treat the first run as a shakedown.

**Roadmap (proposed — to be confirmed by the team)**

| Phase | Goal | Deliverables |
|---|---|---|
| 1 — Foundation | Reliable grounded generation on the benchmark data | Real-data run, recorded CV/hold-out scores, tuned config |
| 2 — Nigerian locality | Replace coarse zones with real local context | Nigeria-specific corpus (crop calendars, pest/disease notes, soil and rainfall context by state/zone); finer `agro_zone` taxonomy; metadata for soil/terrain where available |
| 3 — Honesty | Uncertainty the user can see | Confidence score from neighbour agreement and match strength; hedged phrasing templates; "ask your extension officer" deferral; abstention rate tracked as a metric |
| 4 — Accessibility | Usable by all literacy levels | Reading-level-controlled generation; local-language (e.g. Hausa, Yoruba, Igbo, Pidgin) and voice/SMS delivery; testing with real users |
| 5 — Geology track | Mineral-prospect insight | Separate data sources and evaluation; scoped as its own workstream |

## 6. Guide for contributors: building and fine-tuning this feature

### Working principles

1. **Evidence first.** If a fact (rate, timing, practice) is not in the retrieved sources, the model should not state it. Improve the corpus before improving the model.
2. **Measure before you change.** Add every idea to the cross-validation table and compare against the retrieval baseline. Small data means small differences are noise — prefer changes with clear margins.
3. **Keep the safe path alive.** The retrieval answer must always work without a GPU. New model code must fail back to it.
4. **Same context at train and test time.** Any prompt change must be applied identically to both.
5. **Honesty is a feature with its own tests.** Add examples where the right answer is "not sure" and check the system handles them.

### Suggested workflow

1. **Data:** extend `documents.csv`-style factsheets and Q&A with locality-specific, vetted material. Keep metadata columns consistent.
2. **Retrieval:** check that the right sentences surface for hard questions before touching the model.
3. **Fine-tune:** adjust `LORA_R`, `EPOCHS`, `LR`, `MAX_SEQ_LENGTH` in the config cell; compare greedy vs beam decoding on the hold-out.
4. **Evaluate on three levels:**
   - *Automatic:* the benchmark metric (comparison of strategies).
   - *Expert review:* agronomists / extension officers rate a sample for correctness and local fit, flagging any invented numbers.
   - *User review:* people at different literacy levels rate clarity.
5. **Gate for release:** no answer type goes to users until expert review shows no unsupported dosages/timings and the uncertainty behaviour (Phase 3) is in place.

### Definition of done for a model change

- Beats or matches the retrieval baseline on held-out data by the agreed margin.
- Expert spot-check shows no fabricated specifics.
- Falls back cleanly when the model, GPU or evidence is missing.
- Reproducible from top to bottom with fixed seeds and no network access.

## 7. Repository map and quick start

| File | Purpose |
|---|---|
| `agriculture-climate-slm-optimized.ipynb` | The full pipeline: load data → CV → retrieval/MBR → optional LoRA fine-tune → validated output |
| `README.md` | This document |

**Quick start (Kaggle Notebook):** attach the competition data; for the fine-tuning stage also attach a small instruct model (e.g. Phi-3-mini, Gemma-2-2b-it or Qwen2.5-1.5B-Instruct via Kaggle Models) and enable a GPU; run all cells. Without a GPU or model the notebook runs the retrieval-only path automatically. Behaviour is controlled from the config cell at the top (`USE_LLM`, `MODEL_PATH_OVERRIDE`, `MIN_GAIN`, `REFIT_FULL`, …).

## 8. Open questions for the team

- Which regions, crops and user groups come first, and what is the minimum viable "locality" (state? LGA? coordinates)?
- Which sources are authoritative enough to enter the corpus, and who vets them?
- What confidence threshold triggers a hedge or a handoff to a human?
- Which languages and delivery channels (app, SMS, voice/WhatsApp) reach the target users first?
- Is the geologist track in scope for this release, or a follow-on?

---

*Advice generated by this system is educational and must not replace in-person agronomic or geological assessment.*