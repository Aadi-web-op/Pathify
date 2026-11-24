from datetime import datetime

from beanie import Document, Indexed, PydanticObjectId
from pymongo import ASCENDING, IndexModel


class Classroom(Document):
    """A counsellor's class. Students join with its code and can leave at any time."""

    counsellor_id: Indexed(PydanticObjectId)  # type: ignore[valid-type]
    name: str
    school: str
    code: Indexed(str, unique=True)  # type: ignore[valid-type]
    created_at: datetime
    archived_at: datetime | None = None

    class Settings:
        name = "classrooms"


class Membership(Document):
    """A student in a class. Exists only because the student entered the code and agreed to share a summary."""

    classroom_id: Indexed(PydanticObjectId)  # type: ignore[valid-type]
    student_id: Indexed(PydanticObjectId)  # type: ignore[valid-type]
    joined_at: datetime

    class Settings:
        name = "memberships"
        indexes = [IndexModel([("classroom_id", ASCENDING), ("student_id", ASCENDING)], unique=True)]
