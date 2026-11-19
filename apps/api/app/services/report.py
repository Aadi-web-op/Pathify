"""Build the family report from the student's profile, interest results and tryouts.

The report reuses the same matching and explanations as the results page, then
words them for parents, in English or Hindi. It carries the student's first
name only: no email or other contact details, since the report can be shared.
"""

from app.catalog import catalog, tryout_bank
from app.models import User
from app.models.assessment import latest_result
from app.models.tryout import latest_attempts
from app.models.user import Stage, utcnow
from app.schemas.report import ReportCareer, ReportInterest, ReportOut, ReportStep, ReportTryout
from app.services.matching import fit_label, match
from app.services.report_text import (
    ACTIVITY,
    DIMENSIONS,
    FIT_LABELS,
    QUESTIONS,
    REASONS,
    STAGE,
    STREAMS,
    TEXT,
    Lang,
)

TOP = 3


def _years(years: float) -> str:
    return f"{round(years * 2) / 2:g}"


def stage_text(user: User, lang: Lang) -> str:
    p = user.profile
    if p is None:
        return ""
    if p.stage in (Stage.SCHOOL_9_10, Stage.SCHOOL_11_12):
        text = STAGE[lang]["school"].format(grade=p.grade)
        if p.stream:
            text += f", {STREAMS[lang][p.stream.value]}"
        return text
    key = "college" if p.stage == Stage.COLLEGE else "graduate"
    return STAGE[lang][key].format(degree=p.degree)


def _reflection_text(enjoyment: int, see_myself: str, lang: Lang) -> str:
    t = TEXT[lang]
    return f"{t['enjoyment'].format(n=enjoyment)}. {t['see_' + see_myself]}."


async def build_report(user: User, lang: Lang) -> ReportOut:
    t = TEXT[lang]
    name = user.name.split(" ")[0]

    def fill(key: str, **kw) -> str:
        return t[key].format(name=name, **kw)

    result = await latest_result(user.id)
    attempts = await latest_attempts(user.id)

    interests: list[ReportInterest] = []
    careers: list[ReportCareer] = []
    confidence = None
    if result:
        order = sorted(result.scores, key=lambda k: (-result.scores[k], "RIASEC".index(k)))
        for key in order[:3]:
            label, description = DIMENSIONS[lang][key]
            interests.append(ReportInterest(key=key, name=label, description=description, score=result.scores[key]))
        confidence = t[f"confidence_{result.confidence}"]

        for m in match({d: float(v) for d, v in result.scores.items()}, catalog(), user.profile)[:TOP]:
            c = m.career
            route = m.open_route or m.routes[0]
            tryout = tryout_bank().for_career(c.slug)
            attempt = attempts.get(tryout.slug) if tryout else None
            tryout_text = None
            if attempt and attempt.reflection:
                tryout_text = (
                    f"{t['tried_this'].format(title=tryout.title)}. "
                    f"{_reflection_text(attempt.reflection.enjoyment, attempt.reflection.see_myself, lang)}"
                )

            def reason(r) -> str:
                return REASONS[lang][r.kind].format(name=name, activity=ACTIVITY[lang][r.dimension])

            careers.append(
                ReportCareer(
                    slug=c.slug,
                    title=c.title_hi if lang == "hi" else c.title,
                    summary=c.summary_hi if lang == "hi" else c.summary,
                    fit_label=FIT_LABELS[lang][fit_label(m.fit)],
                    reasons=[reason(r) for r in m.reasons],
                    watch_outs=[reason(r) for r in m.watch_outs],
                    route_label=route.label,
                    route_open=route.open,
                    route_note=None if route.open else fill("not_open"),
                    steps=[ReportStep(label=s.label, years=s.years) for s in route.steps],
                    years_text=t["years"].format(years=_years(route.years)) if route.years else None,
                    check_with=[b.name for b in c.check_with],
                    tryout=tryout_text,
                )
            )

    tryouts: list[ReportTryout] = []
    for slug, attempt in attempts.items():
        tryout = tryout_bank().get(slug)
        career = catalog().get(attempt.career_slug)
        if not tryout or not career or not attempt.reflection:
            continue
        tryouts.append(
            ReportTryout(
                title=tryout.title,
                career_title=career.title_hi if lang == "hi" else career.title,
                summary=_reflection_text(attempt.reflection.enjoyment, attempt.reflection.see_myself, lang),
            )
        )

    stage = user.profile.stage if user.profile else Stage.SCHOOL_9_10
    question_set = (
        "school_9_10" if stage == Stage.SCHOOL_9_10 else "school_11_12" if stage == Stage.SCHOOL_11_12 else "college"
    )
    generated = utcnow()
    text = {
        key: fill(key)
        for key in (
            "title",
            "intro",
            "interests",
            "interests_none",
            "careers",
            "careers_none",
            "why",
            "keep_in_mind",
            "route",
            "check_with",
            "tryouts",
            "tryouts_none",
            "questions",
            "about_heading",
            "about",
            "names_note",
        )
    }
    text["generated"] = t["generated"].format(date=generated.strftime("%d %b %Y"))
    return ReportOut(
        language=lang,
        student_name=name,
        stage=stage_text(user, lang),
        generated_at=generated,
        text=text,
        interests=interests,
        confidence=confidence,
        careers=careers,
        tryouts=tryouts,
        questions=QUESTIONS[lang][question_set],
    )
