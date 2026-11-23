# Pathify

Career guidance for Indian students from Class 9 to college. Pathify helps a student choose a stream after Class 10, pick entrance exams and a degree after Class 12, and get ready for a first job, with reasons they can explain to their parents.

![Landing page with a career metro map](docs/screenshots/landing.png)

![Results: a Class 10 student's top three careers drawn as metro lines from where they are now](docs/screenshots/results.png)

| Assessment | Career guide |
| --- | --- |
| ![Assessment: one activity at a time, five answers](docs/screenshots/assessment.png) | ![Career guide: routes in India with exams, degrees and durations](docs/screenshots/career-guide.png) |
| **Job tryout feedback** | **Family report in Hindi** |
| ![Tryout feedback: each task marked, with the reasoning](docs/screenshots/tryout-feedback.png) | ![Family report in Hindi for parents](docs/screenshots/family-report-hindi.png) |
| **Onboarding** | **Home** |
| ![Onboarding: Class 10 student picks class and board](docs/screenshots/onboarding.png) | ![Home: the student's route ahead and Ksarth prompts](docs/screenshots/home.png) |

## What works today

- **Accounts**: email and password, plus Google, GitHub and LinkedIn when configured. The UI only shows buttons for providers that are set up.
- **Onboarding** for four stages (Class 9–10, Class 11–12, college, graduate). Each stage asks only what applies to it, and school students confirm a parent or guardian knows they use Pathify.
- **Home** shows the decisions ahead of the student as a route, starting where they are now, plus their assessment progress or top matches.
- **Interest assessment**: 30 everyday activities (48 if the student wants a sharper result), scored on Holland's six interest types (RIASEC). Students can go back, leave and resume.
- **Results**: the student's top three careers drawn as metro lines from where they are now, through each real step (stream, entrance exam, degree) to the career. Every match explains why it fits, what to keep in mind, and whether the student's stream allows the route. The page also says how settled the result is.
- **Career library**: a public, statically generated guide to 38 careers across health, engineering, design, law, business, science, public service and skilled trades, filterable by stream. No account needed, so it can be found through search.
- **Job tryouts**: six 10-minute tasks taken from real jobs (data analyst, doctor, UX designer, lawyer, teacher, civil engineer), covering all six interest types. Students get feedback on each answer, see how a professional would approach it, then rate how much they enjoyed it. That rating shows up next to the career in their results and in the family report.
- **Ksarth**, an AI guide that knows the student's stage, stream, goals, interest results and top matches. Its answers draw on Pathify's career guides and link to the ones they used. Replies stream in, history is saved, and Ksarth can answer in Hindi.
- **Family report** in English or Hindi: what the student enjoys, their top three careers with reasons and routes, how their tryouts went, and questions to talk through at home. It can be printed or saved as a PDF, or shared through a private link that expires after 30 days and can be turned off.
- **Settings**: name, light / dark / system theme, sign out, and full account deletion.

Coming next: a full Hindi interface, tryouts for every career in the library, progress tracking along a chosen route, and a class-level view for school counsellors.

## How matching works, and how well

**Scoring.** Each answer updates a running estimate of how much the student enjoys each of the six interest types, with an uncertainty that shrinks as answers come in (a conjugate normal model). The results page shows that uncertainty as a bracket on each score.

**Fit.** Each career has a six-part interest profile. Fit is the correlation between the student's profile and the career's, so it rewards the same *shape* of interests rather than simply liking everything. Each interest type's share of that correlation becomes the explanation on the card ("You enjoy investigating and solving problems, and this career is full of it").

**Accuracy, measured by simulation.** `scripts/simulate_assessment.py` creates students with known interests, has them answer with realistic noise (some statements are more appealing than others, and people aren't perfectly consistent), and checks how many of their true top-5 careers the test finds:

| Questions | True top-5 careers found | True top interest found |
| ---: | ---: | ---: |
| 18 | 65% | 67% |
| 24 | 69% | 70% |
| **30 (default)** | **71%** | **72%** |
| 36 | 74% | 74% |
| 48 (optional) | 77% | 77% |

Even all 48 statements find only about 4 of the 5 true top careers, because neighbouring careers are close together. That is why the product says "places to start exploring, not a verdict", and why job tryouts come next.

**The adaptive test we didn't ship.** The first design picked each next question adaptively, and later tried stopping once the career list stopped moving. Measured against a balanced test of the same length, neither helped (70.6% vs 70.5% top-5 agreement over 1,000 simulated students). With statements that are interchangeable within a type and not yet calibrated on real students, accuracy depends on how many questions are answered, not which. So the shipped test is a balanced 30, with an optional 18 more. Adaptive selection becomes worthwhile once real responses let us calibrate each statement (item response theory).

**Confidence.** The result is labelled high, medium or low by nudging each score up and down by its own uncertainty and counting how many top-5 careers change. In simulation this tracks real accuracy: a list that doesn't move finds about 95% of the true top five; one that moves by two careers finds about 61%. A test checks that the labels stay in that order.

## Job tryouts and how they're graded

Objective tasks (single choice, multi-select, numbers with a tolerance) are marked exactly. Multi-select gets half credit only if every chosen option is right.

Written answers are graded by the model against the task's published criteria, with structured JSON output. The model returns a verdict per criterion (met / partly / missed) with a reason, plus two or three sentences of feedback. **The server computes the points from the verdicts**, so the model can't be talked into "full marks". The student's answer is sent inside `<answer>` tags with any fake closing tag stripped, the prompt tells the model to treat it as data, and phrases that try to instruct the grader are flagged. Answers under five words skip the model. Malformed output is retried once and otherwise left ungraded, and with no AI key, written answers are marked "not graded" instead of getting an invented score.

`evals/grader_cases.json` holds 20 hand-labelled answers: a strong, a partial and a weak one for each written task, plus two injection attempts. `scripts/eval_grader.py` runs the real model against them and fails if exact agreement is below 70%, agreement within one step is below 95%, or an injection attempt scores above 0.25. CI runs it whenever the repository has a `GEMINI_API_KEY` secret.

## How Ksarth stays grounded

Before each reply, a BM25 search over the career library (titles, the names students use such as "MBBS", "CA" or "UPSC", day-to-day work, routes, exams) picks up to three relevant guides. Class numbers and words that appear in every route ("exam", "stream", "college") are ignored, so a vague question finds nothing instead of noise. When a question names no career ("which exams fit my stream?"), the student's own top matches are used instead.

The guides go into the prompt as numbered notes, and the model is asked to cite them like [1]. After the reply, the server removes any citation number that doesn't point at a note and saves the rest as links under the reply. In offline mode Ksarth still lists the guides that look related.

## Sharing the family report safely

The report includes the student's first name only, never an email or contact details. Each student has at most one active link, made of a 32-character random token. It expires after 30 days, a new link replaces the old one, and the student can turn it off at any time. Deleting the account deletes the link. The public endpoint is rate-limited and sends `Cache-Control: no-store` and `X-Robots-Tag: noindex`, and the page is marked `noindex` with `no-referrer`.

Every heading, reason, question and note exists in English and Hindi, and a test checks that both languages have the same keys and the same placeholders. Exam and degree names stay in English in both, as they appear on official forms.

**Where the career data comes from.** `data/careers.json` is written by hand: routes, entrance exams, degrees and typical durations in India, plus the official body to check with. Interest profiles are Pathify's estimates on a 1–7 scale, guided by O*NET's ratings for the similar U.S. occupation named on each career. The catalog deliberately has no salaries or cut-offs, because we can't source and date them yet. A test fails if one slips in. Replacing the estimates with O*NET's published interest data (CC BY 4.0) is a planned improvement.

## Design: Metro

The interface is drawn like a quiet transit map. Careers are destinations, exams are the lines you ride, and your class is the station you start from.

- **Palette.** Warm paper in light mode (soft charcoal in dark), one indigo action colour, and marigold used only for the small "you are here" dot. Muted route colours (indigo, coral, teal, amber) only ever mark routes and data. Tokens live in `apps/web/src/app/globals.css`.
- **Type.** Bricolage Grotesque for display, Geist for text, JetBrains Mono for small labels, and Instrument Serif italic for the one accent word in a headline. Noto Sans Devanagari covers Hindi.
- **Moments, used sparingly.** The landing hero has a thin-line metro map that draws itself in, with one slow train per line. A departures card shows real routes from the career library, and the three big decisions are a scroll-pinned story. The feature tiles are small working demos. A light confetti burst marks your first look at your results and a finished tryout.
- **Restraint.** Effects pause off-screen and in background tabs. With `prefers-reduced-motion` they show their finished state instead. Decorative canvases are hidden from screen readers.

The confetti burst (`apps/web/src/lib/burst.ts`) is an original implementation of an idea from [sakuga-vfx](https://github.com/finktheartist/sakuga-vfx) by Justin Finkelman (MIT).

## Architecture

```mermaid
flowchart LR
  Browser -->|"pages, /api/*"| Web["Next.js 15 (apps/web)"]
  Web -->|"proxies /api/*"| API["FastAPI (apps/api)"]
  API --> Mongo[("MongoDB")]
  API --> LLM["LLM provider: Gemini or offline stub"]
```

The browser only ever talks to the web app's origin. Next.js forwards `/api/*` to FastAPI, so auth cookies are first-party and there is no CORS to get wrong.

Decisions worth knowing about:

| Area | Choice | Why |
| --- | --- | --- |
| Sessions | 15-minute access token and 30-day refresh token, both in httpOnly cookies. The refresh cookie is scoped to `/api/auth`. | JavaScript can't read either token, so an XSS bug can't steal a session. |
| Refresh | Every refresh token works once. Reusing an old one revokes the whole token family. | A stolen refresh token stops working as soon as either party uses it. |
| CSRF | SameSite=Lax cookies plus an Origin / Referer check on every write. | Two independent layers. |
| Passwords | Argon2. v1 bcrypt hashes are accepted and upgraded on the next sign-in. | Existing users keep working without a reset. |
| OAuth | `state` stored in a short-lived httpOnly cookie and compared in constant time. Only provider-verified emails can link to an existing account. | Stops login CSRF and account takeover by unverified email. |
| AI | One `LLMProvider` interface. Gemini sends the key in a header, retries transient errors, and never retries a stream that already sent text. | Model failures show an honest error. Nothing falls back to made-up answers. |
| Limits | Per-user Ksarth quota, per-email and per-IP login limits, per-network sign-up limit (30 an hour by default, since a school lab shares one IP). | Protects the AI budget and slows brute force. In-memory for one instance; Redis when there are more. |
| Types | The web client is generated from FastAPI's OpenAPI schema with `openapi-typescript`. CI fails if the two drift apart. | A renamed field breaks the build, not production. |
| Data | `User` documents stay compatible with the v1 schema. | The existing database keeps working. |
| Career data | One JSON source in `data/`, validated by the API at startup and in CI, and imported by the web app at build time for static career pages. | The API and the public pages can't disagree. |

## Repository layout

```
apps/api      FastAPI app, tests, assessment simulation and Dockerfile
apps/web      Next.js app, Playwright tests and Dockerfile
data/         Career catalog and assessment statements, shared by both apps
legacy/       The v1 HTML pages, backends and ML model, kept for reference while porting
scripts/      purge_secrets_from_history.py
docs/         Screenshots
```

## Run it

### With Docker

```bash
cp .env.example .env
```

Set `SECRET_KEY` in `.env` (and `GEMINI_API_KEY` for real Ksarth replies), then:

```bash
docker compose up --build
```

Open http://localhost:3000.

### Without Docker

You need Python 3.12+, Node 22 and a MongoDB on `localhost:27017`.

```bash
cd apps/api && python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
cd apps/api && .venv/bin/uvicorn app.main:app --reload --port 8000
```

```bash
cd apps/web && npm install && npm run dev
```

Without `GEMINI_API_KEY`, Ksarth runs in offline mode and says so on screen.

### Configuration

All settings are environment variables; see [`.env.example`](.env.example).

| Variable | Default | Notes |
| --- | --- | --- |
| `SECRET_KEY` | `change-me` | Required in production (32+ random characters). |
| `WEB_ORIGIN` | `http://localhost:3000` | Public URL of the web app. Must be https in production. |
| `DATABASE_URL`, `DATABASE_NAME` | local MongoDB, `pathify_db` | |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | empty, `gemini-2.5-flash` | Empty key means offline stub. |
| `GOOGLE_CLIENT_ID`, `GITHUB_CLIENT_*`, `LINKEDIN_CLIENT_*` | empty | OAuth redirect URIs are `{WEB_ORIGIN}/api/auth/{github,linkedin}/callback`. |
| `KSARTH_MESSAGES_PER_HOUR` | `30` | Per student. |
| `SIGNUPS_PER_HOUR_PER_NETWORK` | `30` | Raise it for a school deployment. |
| `API_INTERNAL_URL` | `http://localhost:8000` | Where the web app forwards `/api/*`. Read at build time. |

The API reads `X-Forwarded-For` for rate limiting, so keep it reachable only through the web app, as the compose file does.

## Tests

```bash
cd apps/api && .venv/bin/pytest
```

139 tests run against a real MongoDB (set `TEST_DATABASE_URL` if it isn't on localhost). They cover sign-up, sign-in, refresh-token reuse, OAuth state, origin checks, rate limits, profile validation, account deletion, Ksarth streaming, the Gemini client, the career data, scoring, matching and explanations, stream rules, the assessment flow, the simulated accuracy figures quoted above, tryout marking and the grader's contract with the model, retrieval and citations, and the family report and its share links.

```bash
cd apps/api && .venv/bin/python -m scripts.simulate_assessment 2000
```

```bash
cd apps/web && npm run lint && npm run typecheck && npm run build
```

```bash
cd apps/web && npx playwright test
```

The end-to-end suite runs against the full stack on a desktop, a 412px phone and a 360px phone. It walks through sign-up, onboarding, Ksarth, theme, sign-out, sign-in and account deletion; takes the full assessment as a Class 11 Commerce student, including going back and resuming; checks the results map, explanations and stream rules; browses the public career library; takes a job tryout and reflects on it; follows Ksarth's sources; and opens the family report in both languages, shares it, opens the link signed out, and checks it stops working once turned off. It also checks that model output can't inject HTML, that an expired access token refreshes silently, and that no page scrolls sideways.

CI (`.github/workflows/ci.yml`) runs all of the above on every pull request.

## Security notes

Real API keys were committed to this repository's history before v2. They must be rotated with each provider; rewriting history doesn't make leaked keys safe. After rotating, `scripts/purge_secrets_from_history.py` removes every committed `.env` file and redacts the leaked values from all commits. It then has to be force-pushed, and every existing clone re-cloned.
