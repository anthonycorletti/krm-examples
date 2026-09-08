from app.kit.db.models.base import RecordModel


class Verification(RecordModel, table=True):
    __tablename__ = "verification"
    environment: str = "local"
    component: str = "postgres"
    subject: str
    evidence: str
