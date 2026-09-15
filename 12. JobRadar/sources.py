"""
sources.py — job board adapters.

ALL LEGITIMATE PUBLIC APIs. No scraping, no CAPTCHAs, no ban risk.

WHY NOT LINKEDIN: their partner API is enterprise-only (incorporated
companies, months of approval, thousands/month) and a personal job-search
tool competes with their own product — an automatic rejection. Scraping
violates their ToS and risks your own profile, which is itself a
job-hunting asset.

WHY THAT'S FINE: most LinkedIn postings REDIRECT to Greenhouse, Lever,
Ashby or Workday. Pulling from those is getting the job AT ITS SOURCE,
usually before LinkedIn syndicates it. For anything you find manually,
there's a paste-URL box that runs the identical pipeline.

COMPANY LIST IS SCOPED TO YOUR BAR: ₹40L+ base at 6 YOE, or USD/EUR
remote. No point ingesting companies that structurally can't clear it.
"""

import asyncio
from dataclasses import dataclass, field

import httpx

UA = {"User-Agent": "JobRadar/1.0 (personal job search)"}
#   Identifying yourself honestly is basic good-citizen behaviour on a
#   free public API. Anonymous hammering is what gets IP ranges blocked.


@dataclass
class RawJob:
    source: str
    company: str
    title: str
    location: str = ""
    description: str = ""
    apply_url: str = ""
    salary_raw: str = ""
    tags: list[str] = field(default_factory=list)


# Companies that realistically pay ₹40L+ base at 6 YOE, or pay in USD/EUR
GREENHOUSE = ["stripe", "databricks", "anthropic", "figma", "airbnb",
              "razorpay", "postman", "browserstack", "vercel", "supabase"]
LEVER = ["netflix", "spotify", "plaid", "brex", "cred", "swiggy"]
ASHBY = ["openai", "linear", "replit", "modal", "clickhouse", "deel", "posthog"]


def _clean(html: str) -> str:
    import html as h
    import re
    t = re.sub(r"<[^>]+>", " ", h.unescape(html or ""))
    return re.sub(r"\s+", " ", t).strip()


async def fetch_greenhouse() -> list[RawJob]:
    jobs = []
    async with httpx.AsyncClient(timeout=20, headers=UA) as c:
        for token in GREENHOUSE:
            try:
                r = await c.get(
                    f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs",
                    params={"content": "true"})
                if r.status_code != 200:
                    continue
                for j in r.json().get("jobs", []):
                    jobs.append(RawJob(
                        source="greenhouse", company=token.replace("-", " ").title(),
                        title=j.get("title", ""),
                        location=(j.get("location") or {}).get("name", ""),
                        description=_clean(j.get("content", ""))[:6000],
                        apply_url=j.get("absolute_url", "")))
            except (httpx.HTTPError, ValueError):
                continue    # one dead board must not stop the other nine
    return jobs


async def fetch_lever() -> list[RawJob]:
    jobs = []
    async with httpx.AsyncClient(timeout=20, headers=UA) as c:
        for token in LEVER:
            try:
                r = await c.get(f"https://api.lever.co/v0/postings/{token}",
                                params={"mode": "json"})
                if r.status_code != 200:
                    continue
                for p in r.json():
                    cat = p.get("categories") or {}
                    jobs.append(RawJob(
                        source="lever", company=token.title(),
                        title=p.get("text", ""), location=cat.get("location", ""),
                        description=(p.get("descriptionPlain") or "")[:6000],
                        apply_url=p.get("hostedUrl", ""),
                        tags=[cat.get("commitment", "")]))
                        #   ^ Lever often states "Full-time" here directly —
                        #   a rare case where the SOURCE gives employment type
            except (httpx.HTTPError, ValueError):
                continue
    return jobs


async def fetch_ashby() -> list[RawJob]:
    """Ashby is where recent YC / a16z-backed startups cluster — often the
    most willing to hire remotely across borders and sponsor visas."""
    jobs = []
    async with httpx.AsyncClient(timeout=20, headers=UA) as c:
        for token in ASHBY:
            try:
                r = await c.get(
                    f"https://api.ashbyhq.com/posting-api/job-board/{token}",
                    params={"includeCompensation": "true"})
                    #   Ashby optionally returns comp — free signal that
                    #   lets the salary agent skip expensive web research
                if r.status_code != 200:
                    continue
                for j in r.json().get("jobs", []):
                    comp = (j.get("compensation") or {}).get("compensationTierSummary", "")
                    jobs.append(RawJob(
                        source="ashby",
                        company=j.get("organizationName") or token.title(),
                        title=j.get("title", ""), location=j.get("location", ""),
                        description=(j.get("descriptionPlain") or "")[:6000],
                        apply_url=j.get("jobUrl", ""), salary_raw=comp,
                        tags=[j.get("employmentType", "")]))
            except (httpx.HTTPError, ValueError):
                continue
    return jobs


async def fetch_remoteok() -> list[RawJob]:
    jobs = []
    async with httpx.AsyncClient(timeout=20, headers=UA) as c:
        try:
            r = await c.get("https://remoteok.com/api")
            data = r.json() if r.status_code == 200 else []
        except (httpx.HTTPError, ValueError):
            return jobs
    for j in data[1:] if len(data) > 1 else []:   # [0] is a legal notice
        if not isinstance(j, dict):
            continue
        lo, hi = j.get("salary_min"), j.get("salary_max")
        jobs.append(RawJob(
            source="remoteok", company=j.get("company", ""),
            title=j.get("position", ""), location=j.get("location") or "Remote",
            description=(j.get("description") or "")[:6000],
            apply_url=j.get("url", ""),
            salary_raw=f"${lo:,}-${hi:,} USD" if lo and hi else "",
            tags=j.get("tags", []) or []))
    return jobs


async def fetch_arbeitnow() -> list[RawJob]:
    """German/EU focused — notably surfaces visa-sponsorship roles."""
    jobs = []
    async with httpx.AsyncClient(timeout=20, headers=UA) as c:
        try:
            r = await c.get("https://www.arbeitnow.com/api/job-board-api")
            data = r.json().get("data", []) if r.status_code == 200 else []
        except (httpx.HTTPError, ValueError):
            return jobs
    for j in data:
        jobs.append(RawJob(
            source="arbeitnow", company=j.get("company_name", ""),
            title=j.get("title", ""), location=j.get("location", ""),
            description=(j.get("description") or "")[:6000],
            apply_url=j.get("url", ""), tags=j.get("tags", []) or []))
    return jobs


async def fetch_remotive() -> list[RawJob]:
    jobs = []
    async with httpx.AsyncClient(timeout=20, headers=UA) as c:
        try:
            r = await c.get("https://remotive.com/api/remote-jobs",
                            params={"limit": 150})
            data = r.json().get("jobs", []) if r.status_code == 200 else []
        except (httpx.HTTPError, ValueError):
            return jobs
    for j in data:
        jobs.append(RawJob(
            source="remotive", company=j.get("company_name", ""),
            title=j.get("title", ""),
            location=j.get("candidate_required_location", ""),
            #   ^ the most useful field here: it states WHERE candidates
            #   may live ("Worldwide", "Europe", "USA Only") — maps almost
            #   directly onto the relocation tiers
            description=_clean(j.get("description", ""))[:6000],
            apply_url=j.get("url", ""), salary_raw=j.get("salary", "") or "",
            tags=[j.get("job_type", "")]))
    return jobs


ALL_SOURCES = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "ashby": fetch_ashby,
    "remoteok": fetch_remoteok,
    "arbeitnow": fetch_arbeitnow,
    "remotive": fetch_remotive,
}


async def fetch_all() -> tuple[list[RawJob], dict]:
    """
    All sources CONCURRENTLY.

    return_exceptions=True is doing real work: without it, the first
    failing source cancels the rest and you lose five good result sets
    because one board changed its URL. (Project 4's lesson, applied.)
    """
    names = list(ALL_SOURCES)
    results = await asyncio.gather(*[ALL_SOURCES[n]() for n in names],
                                   return_exceptions=True)
    jobs, stats = [], {}
    for name, res in zip(names, results):
        if isinstance(res, Exception):
            stats[name] = f"failed: {str(res)[:60]}"
        else:
            stats[name] = len(res)
            jobs.extend(res)
    return jobs, stats