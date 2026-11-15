"""Database connection, collection names, indexes, and lifecycle management."""

import logging
from typing import Optional, Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel
from app.config import settings

logger = logging.getLogger(__name__)

# Canonical Collection Names
USERS_COLLECTION = "users"
EXAM_SESSIONS_COLLECTION = "exam_sessions"
EVENTS_COLLECTION = "events"
EVIDENCE_COLLECTION = "evidence"
REPORTS_COLLECTION = "reports"


class InMemoryAsyncCollection:
    """Lightweight in-memory async collection for testing or degraded offline operation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._data: List[Dict[str, Any]] = []

    async def insert_one(self, document: Dict[str, Any]) -> Any:
        doc = dict(document)
        self._data.append(doc)
        class InsertResult:
            inserted_id = doc.get("_id", doc.get("id", doc.get("session_id")))
        return InsertResult()

    def _match_doc(self, doc: Dict[str, Any], filter_query: Dict[str, Any]) -> bool:
        """Check if document satisfies MongoDB filter expression."""
        for k, v in filter_query.items():
            doc_val = doc.get(k)
            if isinstance(v, dict):
                for op, val in v.items():
                    if op == "$in":
                        if isinstance(doc_val, list):
                            if not any(item in val for item in doc_val):
                                return False
                        elif doc_val not in val:
                            return False
                    elif op == "$eq":
                        if doc_val != val:
                            return False
                    elif op == "$ne":
                        if doc_val == val:
                            return False
                    elif op == "$gte":
                        if doc_val is None or doc_val < val:
                            return False
                    elif op == "$gt":
                        if doc_val is None or doc_val <= val:
                            return False
                    elif op == "$lte":
                        if doc_val is None or doc_val > val:
                            return False
                    elif op == "$lt":
                        if doc_val is None or doc_val >= val:
                            return False
            elif isinstance(doc_val, list):
                if v not in doc_val:
                    return False
            elif doc_val != v:
                return False
        return True

    async def count_documents(self, filter_query: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching filter expression."""
        filter_query = filter_query or {}
        return sum(1 for doc in self._data if self._match_doc(doc, filter_query))

    async def find_one(self, filter_query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        for doc in self._data:
            if self._match_doc(doc, filter_query):
                return dict(doc)
        return None

    def find(self, filter_query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None) -> "InMemoryAsyncCursor":
        filter_query = filter_query or {}
        matching = [dict(doc) for doc in self._data if self._match_doc(doc, filter_query)]
        return InMemoryAsyncCursor(matching)

    async def update_one(self, filter_query: Dict[str, Any], update_query: Dict[str, Any]) -> Any:
        doc = await self.find_one(filter_query)
        if doc:
            for item in self._data:
                match = True
                for k, v in filter_query.items():
                    if item.get(k) != v:
                        match = False
                        break
                if match:
                    if "$set" in update_query:
                        item.update(update_query["$set"])
                    class UpdateResult:
                        modified_count = 1
                    return UpdateResult()
        class EmptyUpdateResult:
            modified_count = 0
        return EmptyUpdateResult()

    async def delete_one(self, filter_query: Dict[str, Any]) -> Any:
        idx = -1
        for i, doc in enumerate(self._data):
            match = True
            for k, v in filter_query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                idx = i
                break
        if idx != -1:
            self._data.pop(idx)
            class DeleteResult:
                deleted_count = 1
            return DeleteResult()
        class EmptyDeleteResult:
            deleted_count = 0
        return EmptyDeleteResult()

    async def create_indexes(self, indexes: List[IndexModel]) -> List[str]:
        return [idx.document.get("name", "idx") for idx in indexes]


class InMemoryAsyncCursor:
    """Cursor wrapper for InMemoryAsyncCollection query results."""

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self._data = data
        self._sort_key = None
        self._sort_dir = 1
        self._skip = 0
        self._limit = None

    def sort(self, key_or_list: Any, direction: int = 1) -> "InMemoryAsyncCursor":
        if isinstance(key_or_list, list):
            self._sort_key, self._sort_dir = key_or_list[0]
        else:
            self._sort_key = key_or_list
            self._sort_dir = direction
        return self

    def skip(self, count: int) -> "InMemoryAsyncCursor":
        self._skip = count
        return self

    def limit(self, count: int) -> "InMemoryAsyncCursor":
        self._limit = count
        return self

    async def to_list(self, length: Optional[int] = None) -> List[Dict[str, Any]]:
        results = list(self._data)
        if self._sort_key:
            reverse = self._sort_dir == DESCENDING
            results.sort(key=lambda x: str(x.get(self._sort_key, "")), reverse=reverse)
        if self._skip:
            results = results[self._skip:]
        if self._limit is not None:
            results = results[:self._limit]
        if length is not None:
            results = results[:length]
        return results


class InMemoryAsyncDatabase:
    """In-memory MongoDB substitute maintaining collection instances."""

    def __init__(self) -> None:
        self._collections: Dict[str, InMemoryAsyncCollection] = {}

    def __getitem__(self, name: str) -> InMemoryAsyncCollection:
        if name not in self._collections:
            self._collections[name] = InMemoryAsyncCollection(name)
        return self._collections[name]

    def get_collection(self, name: str) -> InMemoryAsyncCollection:
        return self[name]


class DatabaseManager:
    """Manages MongoDB async client, in-memory fallback, and index creation."""

    client: Optional[AsyncIOMotorClient] = None
    db: Optional[Any] = None
    is_mock: bool = False

    @classmethod
    async def connect(cls) -> None:
        """Connect to MongoDB or initialize mock storage if unreachable."""
        if cls.is_mock and cls.db is not None:
            await cls.create_indexes()
            return

        try:
            import asyncio
            logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}...")
            cls.client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=500,
            )
            test_db = cls.client[settings.MONGODB_DB_NAME]
            # Ping database with strict 0.5s timeout to confirm availability
            await asyncio.wait_for(cls.client.admin.command("ping"), timeout=0.5)
            cls.db = test_db
            cls.is_mock = False
            logger.info("Successfully connected to MongoDB.")
            await cls.create_indexes()
        except Exception as exc:
            logger.warning(
                f"MongoDB unavailable ({exc}). Initializing high-fidelity in-memory database store."
            )
            cls.db = InMemoryAsyncDatabase()
            cls.is_mock = True
            await cls.create_indexes()

    @classmethod
    async def create_indexes(cls) -> None:
        """Create indexes on session_id, timestamp, email, etc."""
        if cls.db is None:
            return

        try:
            # 1. users: unique index on email
            user_indexes = [
                IndexModel([("email", ASCENDING)], unique=True, name="idx_users_email_unique"),
                IndexModel([("role", ASCENDING)], name="idx_users_role"),
            ]
            await cls.db[USERS_COLLECTION].create_indexes(user_indexes)

            # 2. exam_sessions: unique index on session_id, index on start_time, index on assigned_proctor
            session_indexes = [
                IndexModel([("session_id", ASCENDING)], unique=True, name="idx_sessions_id_unique"),
                IndexModel([("start_time", ASCENDING)], name="idx_sessions_start_time"),
                IndexModel([("status", ASCENDING)], name="idx_sessions_status"),
                IndexModel([("assigned_proctor", ASCENDING)], name="idx_sessions_assigned_proctor"),
            ]
            await cls.db[EXAM_SESSIONS_COLLECTION].create_indexes(session_indexes)

            # 3. events: indexes on session_id and timestamp
            event_indexes = [
                IndexModel([("session_id", ASCENDING), ("timestamp", DESCENDING)], name="idx_events_session_timestamp"),
                IndexModel([("session_id", ASCENDING)], name="idx_events_session_id"),
                IndexModel([("timestamp", DESCENDING)], name="idx_events_timestamp"),
                IndexModel([("modality", ASCENDING)], name="idx_events_modality"),
            ]
            await cls.db[EVENTS_COLLECTION].create_indexes(event_indexes)

            # 4. evidence: indexes on session_id, event_id, timestamp
            evidence_indexes = [
                IndexModel([("session_id", ASCENDING)], name="idx_evidence_session_id"),
                IndexModel([("event_id", ASCENDING)], name="idx_evidence_event_id"),
                IndexModel([("timestamp", DESCENDING)], name="idx_evidence_timestamp"),
            ]
            await cls.db[EVIDENCE_COLLECTION].create_indexes(evidence_indexes)

            # 5. reports: unique on session_id, index on created_at
            report_indexes = [
                IndexModel([("session_id", ASCENDING)], unique=True, name="idx_reports_session_id_unique"),
                IndexModel([("created_at", DESCENDING)], name="idx_reports_created_at"),
            ]
            await cls.db[REPORTS_COLLECTION].create_indexes(report_indexes)

            logger.info("Successfully established database indexes on session_id, timestamp, and constraints.")
        except Exception as exc:
            logger.error(f"Error creating database indexes: {exc}")

    @classmethod
    async def disconnect(cls) -> None:
        """Close MongoDB connection gracefully."""
        if cls.client:
            logger.info("Closing MongoDB connection...")
            cls.client.close()
            cls.client = None
        if not cls.is_mock:
            cls.db = None
            cls.is_mock = False
        logger.info("Database connection closed.")

    @classmethod
    def get_db(cls) -> Any:
        """Get the current active database instance."""
        if cls.db is None:
            # Lazy initialize in-memory fallback if accessed prior to lifespan
            cls.db = InMemoryAsyncDatabase()
            cls.is_mock = True
        return cls.db


async def get_database() -> Any:
    """FastAPI dependency yielding the database instance."""
    return DatabaseManager.get_db()
