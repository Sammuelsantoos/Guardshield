"""Modelos relacionais do GuardShield."""
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Index, String

from src.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), unique=True, nullable=False, index=True)
    customer_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    merchant_category = Column(String(16))
    merchant_country = Column(String(4))
    customer_country = Column(String(4))
    device_id = Column(String(64))
    ip_address = Column(String(64))
    is_fraud = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Index("ix_transactions_customer_ts", Transaction.customer_id, Transaction.timestamp)
