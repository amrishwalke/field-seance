# Field Seance

**Track 3 - The Open Haunt.** Get a biomedical innovator up to speed in an unfamiliar field in minutes, not weeks.

## The problem

Engineers, clinicians and founders moving into a new clinical area spend weeks piecing together anatomy, current treatments, unmet needs and where the field is heading. Generic chatbots answer one question at a time, with no structure and no sources.

## What it does

Type a field, for example `peripheral nerve repair`. Field Seance builds a live training guide:

- **4 categories:** anatomy and physiology, current approaches, unmet needs, future directions
- **3 depth levels in each:** Orientation, Working knowledge, Expert detail
- **Citations on every factual sentence**, plus self-check questions for each level
- **Opportunity statements** ("A way to ... in ... that ..."), each tied to a cited unmet need, and a 10-day study plan

The output is a single HTML page with collapsible levels, generated live on every run. Nothing is hardcoded.

## How it works, and why each partner is needed

| Partner | Role | What breaks without it |
|---|---|---|
| **CrewAI** | Runs two agents in sequence: a field tutor (one task per category) and an opportunity scout that reads all four briefs | No structured teaching, no cross-category synthesis |
| **You.com** | Web Search API, two live searches per category. Results are numbered, and every factual sentence cites one | Every claim would be unsourced model memory |
| **Parasail** | OpenAI-compatible inference: `parasail-glm-53-flash`, with `parasail-deepseek-v4-flash` as an automatic backup if a section comes back empty | Nothing reasons over the evidence or writes the guide |

## Run it

Requires Python 3.10-3.13 (CrewAI does not support 3.14).

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then paste your two keys into .env
python main.py "peripheral nerve repair"
```

**macOS / Linux**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your two keys into .env
python main.py "peripheral nerve repair"
```

A run takes a few minutes: eight live searches, then each section is written as its own small CrewAI job, so one failure doesn't lose the whole guide. Requests are capped at 4 per minute to stay under Parasail's free-tier limit. The guide opens in your browser and is saved in `output/`.

To change the grid, edit `CATEGORIES` and `LEVELS` at the top of `main.py`.

## Limits

- Sources come from the open web. Paywalled papers are mostly out of reach.
- The model can still misread a source. Every citation is linked so you can check it.
- This is an educational overview, not clinical advice.

## Privacy

Queries go to You.com and Parasail, which are cloud services. Use it for public topics only. Don't enter confidential, client or patient information. Keys live in `.env`, which is git-ignored.
