from app.catalog import Career, item_bank
from app.models import KsarthMessage, User
from app.models.assessment import StoredResult
from app.models.user import Goal, Stage, Stream, StudentProfile
from app.services.llm import ChatMessage
from app.services.retrieval import career_note

HISTORY_TURNS = 12

STAGE_TEXT = {
    Stage.SCHOOL_9_10: "in Class {grade}",
    Stage.SCHOOL_11_12: "in Class {grade}",
    Stage.COLLEGE: "a college student",
    Stage.GRADUATE: "a recent graduate",
}
STREAM_TEXT = {
    Stream.SCIENCE_PCM: "Science (PCM)",
    Stream.SCIENCE_PCB: "Science (PCB)",
    Stream.SCIENCE_PCMB: "Science (PCMB)",
    Stream.COMMERCE: "Commerce",
    Stream.HUMANITIES: "Humanities / Arts",
    Stream.VOCATIONAL: "a vocational stream",
    Stream.UNDECIDED: "an undecided stream",
}
GOAL_TEXT = {
    Goal.CHOOSE_STREAM: "choosing a stream after Class 10",
    Goal.CHOOSE_EXAMS_DEGREE: "choosing entrance exams and a degree",
    Goal.FIRST_JOB: "getting a first job or internship",
    Goal.SWITCH_FIELD: "switching to a different field",
    Goal.EXPLORE: "exploring options without a fixed goal",
}

BASE_PROMPT = """You are Ksarth, the career guide inside Pathify, an app that helps Indian students \
from Class 9 to college make education and career decisions.

How you work:
- Be warm, direct and practical. Short paragraphs, plain words. Use lists only for steps or options.
- Ground advice in the Indian system: boards (CBSE, ICSE, state), streams, entrance exams such as \
JEE, NEET, CUET, CLAT, NID DAT and NATA, degrees, and how hiring works in India.
- Never invent statistics, cut-offs, fees, salaries or dates. If you are not sure of a figure, say so \
and tell the student where to check (the official exam or college website).
- Show more than one route when there is one. Many careers can be reached from several streams.
- Respect family pressure without dismissing it. Suggest how to talk it through with parents.
- Do not make the decision for the student. Help them reason about it.
- If a student mentions self-harm, hopelessness or abuse, respond with care, encourage them to talk \
to a trusted adult, and share India's Tele-MANAS helpline: 14416 (free, 24x7).
- Stay on education, careers, skills and study planning. Politely steer back if asked about other topics.
"""


def describe_student(user: User) -> str:
    profile: StudentProfile | None = user.profile
    first_name = user.name.split(" ")[0]
    if profile is None:
        return f"The student's name is {first_name}. They have not completed onboarding yet."
    parts = [f"The student's name is {first_name}. They are {STAGE_TEXT[profile.stage].format(grade=profile.grade)}"]
    if profile.stream:
        parts.append(f" studying {STREAM_TEXT[profile.stream]}")
    if profile.board:
        parts.append(f" ({profile.board.value.upper()} board)")
    if profile.degree:
        year = f", year {profile.year_of_study}" if profile.year_of_study else ""
        parts.append(f" doing {profile.degree}{year}")
    if profile.state:
        parts.append(f", based in {profile.state}")
    parts.append(".")
    goals = ", ".join(GOAL_TEXT[g] for g in profile.goals)
    parts.append(f" They want help with: {goals}.")
    if profile.interests:
        parts.append(f" Interests they listed: {', '.join(profile.interests)}.")
    if profile.language == "hi":
        parts.append(" They prefer Hindi. Reply in simple Hindi (Devanagari) unless they write in English.")
    return "".join(parts)


def describe_interests(result: StoredResult | None, top_careers: list[str]) -> str:
    if result is None:
        return "They haven't taken the interest assessment yet; you can suggest it when it would help."
    bank = item_bank()
    names = [bank.dimension(k).name for k in result.code]
    text = (
        f"Their interest assessment (Holland RIASEC) is strongest in {names[0]}, then {names[1]} and {names[2]} "
        f"(confidence: {result.confidence})."
    )
    if top_careers:
        text += f" Their closest career matches are {', '.join(top_careers)}."
    return text + " Treat these as starting points to explore, not verdicts."


CITATION_RULES = """Reference notes from Pathify's career library follow. Use them for facts about routes, \
exams, degrees and durations. When a sentence relies on a note, cite it with its number in square brackets, \
like [1]. Only cite notes you actually used, and never invent a number. If the notes don't cover what was \
asked, say so plainly and suggest the official body to check with, instead of guessing."""


def reference_block(notes: list[Career]) -> str:
    if not notes:
        return ""
    lines = "\n".join(f"[{i}] {career_note(c)}" for i, c in enumerate(notes, start=1))
    return f"\n\n{CITATION_RULES}\n{lines}"


def system_prompt(
    user: User,
    result: StoredResult | None = None,
    top_careers: list[str] | None = None,
    notes: list[Career] | None = None,
) -> str:
    return (
        f"{BASE_PROMPT}\nAbout this student:\n{describe_student(user)}\n"
        f"{describe_interests(result, top_careers or [])}"
        f"{reference_block(notes or [])}"
    )


async def recent_history(user: User) -> list[ChatMessage]:
    rows = (
        await KsarthMessage.find(KsarthMessage.user_id == user.id)
        .sort(-KsarthMessage.created_at)
        .limit(HISTORY_TURNS)
        .to_list()
    )
    messages = [ChatMessage(role=r.role, content=r.content) for r in reversed(rows)]
    # Gemini expects the conversation to start with a user turn.
    while messages and messages[0].role != "user":
        messages.pop(0)
    return messages
