"""
Field Seance - get a biomedical innovator up to speed in a new field, fast.

Usage:
    python main.py "peripheral nerve repair"

Partners:
    You.com  - live web search; every claim in the guide is cited to a result it returned
    CrewAI   - runs the agents: a field tutor for each category and an opportunity scout
    Parasail - the models that reason over the evidence and write
"""

import datetime
import html
import json
import os
import re
import sys
import time
import webbrowser
from pathlib import Path

import markdown
import requests
from dotenv import load_dotenv

load_dotenv()
for key in ("YDC_API_KEY", "PARASAIL_API_KEY"):
    if not os.getenv(key):
        sys.exit(f"Missing {key}. Copy .env.example to .env and add your keys.")

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402

FIELD = " ".join(sys.argv[1:]).strip() or "peripheral nerve repair"

# The training grid: each category and the searches used to ground it.
CATEGORIES = {
    "Anatomy and physiology": ["anatomy and physiology", "structure and function review"],
    "Current approaches": ["current treatment options", "standard of care clinical outcomes"],
    "Unmet needs": ["limitations and unmet clinical needs", "complications and poor outcomes"],
    "Future directions": ["emerging technologies and future directions", "recent clinical trials"],
}
LEVELS = [
    "Level 1 - Orientation (plain language, key terms)",
    "Level 2 - Working knowledge (mechanisms, key numbers, main options)",
    "Level 3 - Expert detail (specific studies, trials, devices, open debates)",
]

# Parasail models (OpenAI-compatible). If the first one returns an empty answer,
# the script automatically retries that section with the backup model.
PRIMARY_MODEL = "parasail-glm-53-flash"
BACKUP_MODEL = "parasail-deepseek-v4-flash"

# Pause between sections so separate jobs stay under Parasail's free-tier limit
# of 5 requests per minute. Set to 0 if your account has the paid limit.
PAUSE_SECONDS = 15


def make_llm(model_id: str) -> LLM:
    return LLM(
        model=f"openai/{model_id}",
        base_url="https://api.parasail.io/v1",
        api_key=os.environ["PARASAIL_API_KEY"],
        temperature=0.2,
        max_tokens=12000,  # room for the model to think and still write the answer
    )


LLMS = [(PRIMARY_MODEL, make_llm(PRIMARY_MODEL)), (BACKUP_MODEL, make_llm(BACKUP_MODEL))]


def you_search(query: str, count: int = 4) -> list[dict]:
    """Run one You.com web search and return title, URL and passage text for each result."""
    response = requests.post(
        "https://ydc-index.io/v1/search",
        headers={"X-API-Key": os.environ["YDC_API_KEY"], "Content-Type": "application/json"},
        json={"query": query, "count": count},
        timeout=45,
    )
    if response.status_code != 200:
        print(f"  Search failed ({response.status_code}) for '{query}': {response.text[:200]}")
        return []
    found = []
    for item in response.json().get("results", {}).get("web", []):
        passages = item.get("snippets") or [item.get("description", "")]
        text = " ".join(p if isinstance(p, str) else json.dumps(p) for p in passages).strip()
        if not text or "checking your browser" in text.lower():
            continue  # skip pages that returned no usable text
        found.append({"title": item.get("title") or "Untitled", "url": item.get("url", ""), "text": text[:500]})
    return found


def gather_sources(query_suffixes: list[str]) -> list[dict]:
    """Search each query for this category and merge results, dropping duplicate URLs."""
    sources, seen = [], set()
    for suffix in query_suffixes:
        query = f"{FIELD} {suffix}"
        print(f"  You.com search: {query}")
        for result in you_search(query):
            if result["url"] and result["url"] not in seen:
                seen.add(result["url"])
                sources.append(result)
    return sources


def sources_for_prompt(sources: list[dict]) -> str:
    if not sources:
        return "(No usable sources were found for this category.)"
    return "\n".join(f"[{i}] {s['title']} ({s['url']})\n{s['text']}" for i, s in enumerate(sources, 1))


def sources_markdown(sources: list[dict]) -> str:
    if not sources:
        return "### Sources\nNo usable sources were found."
    lines = []
    for i, s in enumerate(sources, 1):
        title = s["title"].replace("[", "(").replace("]", ")")
        lines.append(f"{i}. [{title}]({s['url']})")
    return "### Sources\n" + "\n".join(lines)


def make_tutor(llm: LLM) -> Agent:
    return Agent(
        role="Biomedical field tutor",
        goal=f"Teach an innovator the field of {FIELD} quickly and accurately, at increasing depth",
        backstory=(
            "You are a clinician-engineer who builds crash courses for medical-device innovators. "
            "You only state facts found in the sources you are given, and you cite every one."
        ),
        llm=llm,
        respect_context_window=False,
        max_iter=3,
        verbose=True,
    )


def make_scout(llm: LLM) -> Agent:
    return Agent(
        role="Medtech opportunity scout",
        goal="Turn unmet needs and the limits of current approaches into concrete innovation opportunities",
        backstory="You write crisp, evidence-backed need statements for medical-device innovators.",
        llm=llm,
        respect_context_window=False,
        max_iter=3,
        verbose=True,
    )


def run_section(make_agent, description: str, expected_output: str) -> str:
    """Run one section as its own small crew. If a model fails or returns nothing, try the backup."""
    for model_id, llm in LLMS:
        print(f"\n  Writing with {model_id}...")
        agent = make_agent(llm)
        task = Task(description=description, expected_output=expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, max_rpm=4, verbose=True)
        try:
            text = crew.kickoff().raw
            if text and text.strip():
                return text
            print(f"  {model_id} returned an empty answer.")
        except Exception as error:  # keep going so one failure doesn't lose the whole guide
            print(f"  {model_id} failed: {str(error)[:200]}")
        time.sleep(PAUSE_SECONDS)
    return "_This section could not be generated on this run. Run the script again to retry._"


level_list = "\n".join(f"- {level}" for level in LEVELS)


def category_description(category: str, sources: list[dict]) -> str:
    return (
        f"Field: {FIELD}\nCategory: {category}\n\n"
        f"Sources found by live You.com search:\n{sources_for_prompt(sources)}\n\n"
        f"Teach this category at these three depths:\n{level_list}\n\n"
        "Rules:\n"
        "- Start each depth with a heading line exactly like '### Level 1 - Orientation'.\n"
        "- End each level with 2 self-check questions.\n"
        "- After every factual sentence, cite the supporting source number in square "
        "brackets, like [2]. Use only the source numbers listed above.\n"
        "- Use only facts stated in the sources above. If the sources are thin for a "
        "level, say so in one sentence instead of guessing.\n"
        "- Do not write a sources list. It is added automatically.\n"
        "- Keep your private reasoning brief and spend your effort on the written answer."
    )


def scout_description(briefs: dict[str, str]) -> str:
    joined = "\n\n".join(f"=== {category} ===\n{text}" for category, text in briefs.items())
    return (
        f"Field: {FIELD}. Here are the category briefs written from live sources:\n\n{joined}\n\n"
        "Using only these briefs, write:\n"
        "1. Three opportunity statements in the form 'A way to [address a problem] in "
        "[population] that [measurable outcome]'. For each, name the unmet need and the "
        "limitation of current approaches it targets, and cite the supporting evidence as "
        "(category name [number]).\n"
        "2. A 10-day study plan that moves through the categories from Level 1 to Level 3.\n"
        "Do not add facts that are not in the briefs. Start the two parts with the headings "
        "'### Opportunities' and '### 10-day study plan'."
    )


def md_to_html(md_text: str) -> str:
    # Escape raw HTML so nothing from the web or the model can run as code in the page.
    return markdown.markdown(md_text.replace("<", "&lt;"), extensions=["extra"])


def sections_to_html(md_text: str) -> str:
    """Wrap each '### ' section in its own collapsible block, Level 1 open by default."""
    chunks = re.split(r"(?m)^### ", md_text)
    out = [md_to_html(chunks[0])] if chunks[0].strip() else []
    for chunk in chunks[1:]:
        heading, _, body = chunk.partition("\n")
        is_open = " open" if heading.strip().lower().startswith(("level 1", "opportunities")) else ""
        out.append(
            f"<details{is_open}><summary>{html.escape(heading.strip())}</summary>"
            f"{md_to_html(body)}</details>"
        )
    return "\n".join(out)


def build_page(sections: list[tuple[str, str]], minutes: float, source_count: int) -> str:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(
        f"<section><h2>{html.escape(title)}</h2>{sections_to_html(text)}</section>"
        for title, text in sections
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Field Seance: {html.escape(FIELD)}</title>
<style>
:root {{ --bg:#fafaf7; --fg:#1d1d1b; --muted:#6b6a64; --card:#ffffff; --line:#e3e1d9; --accent:#5b4bb7; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#1b1b1a; --fg:#ecebe6; --muted:#a3a29b; --card:#262624; --line:#3a3a37; --accent:#afa9ec; }}
}}
body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.65 system-ui, sans-serif; }}
main {{ max-width:880px; margin:0 auto; padding:32px 20px 60px; }}
h1 {{ font-size:28px; margin:0 0 4px; }} h2 {{ font-size:20px; margin:0 0 12px; }}
.meta {{ color:var(--muted); font-size:14px; margin-bottom:24px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; margin:16px 0; }}
details {{ border-top:1px solid var(--line); padding:10px 0; }}
summary {{ cursor:pointer; font-weight:600; color:var(--accent); }}
a {{ color:var(--accent); overflow-wrap:anywhere; }}
footer {{ color:var(--muted); font-size:13px; margin-top:28px; }}
</style></head>
<body><main>
<h1>Field Seance: {html.escape(FIELD)}</h1>
<div class="meta">Generated live {stamp} in {minutes:.1f} min from {source_count} You.com sources. Reasoning: Parasail. Agents: CrewAI.</div>
{body}
<footer>Educational overview generated from live web sources. Not clinical advice. Check every citation before relying on it.</footer>
</main></body></html>"""


if __name__ == "__main__":
    print(f"\nBuilding a field guide for: {FIELD}\n")
    start = time.time()

    sources_by_category = {}
    for category, suffixes in CATEGORIES.items():
        print(f"{category}:")
        sources_by_category[category] = gather_sources(suffixes)
        print(f"  {len(sources_by_category[category])} usable sources\n")

    briefs = {}
    for category, sources in sources_by_category.items():
        print(f"\n=== Writing: {category} ===")
        briefs[category] = run_section(
            make_tutor,
            category_description(category, sources),
            "Markdown with three level sections, inline citations and self-check questions.",
        )
        time.sleep(PAUSE_SECONDS)

    print("\n=== Writing: Opportunities and study plan ===")
    opportunities = run_section(
        make_scout,
        scout_description(briefs),
        "Markdown with '### Opportunities' and '### 10-day study plan' sections.",
    )
    minutes = (time.time() - start) / 60

    sections = [
        (category, f"{briefs[category]}\n\n{sources_markdown(sources)}")
        for category, sources in sources_by_category.items()
    ]
    sections.append(("Opportunities and study plan", opportunities))
    total_sources = sum(len(s) for s in sources_by_category.values())

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", FIELD.lower()).strip("-") or "field"
    page = out_dir / f"field_guide_{slug}.html"
    page.write_text(build_page(sections, minutes, total_sources), encoding="utf-8")
    print(f"\nDone in {minutes:.1f} minutes. Saved {page.resolve()}")
    webbrowser.open(page.resolve().as_uri())
