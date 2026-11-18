# Legacy code (reference only)

This folder holds the first version of Pathify. Nothing here is served or deployed.
It stays in the repo so features can be ported into `apps/` stage by stage, and it
will be deleted once that is done.

| Path | What it was | Status |
| --- | --- | --- |
| `frontend/` | 11 static HTML pages using the Tailwind Play CDN | Replaced by `apps/web` (landing, auth, onboarding, home, Ksarth). Community, certifications, gamification and roadmap pages are ported in later stages. |
| `backend_ml_main.py` | FastAPI app that tried to load the pickled model | Replaced by `apps/api` |
| `backend_rules_main.py` | Same app with a hand-written rules predictor | Replaced by `apps/api` |
| `predictionModel/` | Gradient-boosting model, encoders and training CSV | Kept for reference. The model only covers 12 IT roles and needs an old scikit-learn to unpickle. Stage 2 replaces it with RIASEC-based matching. |

Known problems in this code, all fixed or removed in `apps/`:

- Stored XSS in community chat (`innerHTML` with user content)
- Self-service certificate completion with +500 XP
- Calls to the retired `gemini-1.5-flash` model with silent hardcoded fallbacks
- JWT kept in `localStorage`; OAuth without a `state` parameter
- Hardcoded or random stats presented as real data
