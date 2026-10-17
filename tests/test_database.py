import os
import pytest
from src.database import healthcheck

@pytest.mark.skipif(
    os.getenv("SKIP_DB_TESTS") == "1",
    reason="Testes de banco desativados via SKIP_DB_TESTS=1",
)
def test_database_is_reachable():
    assert healthcheck() is True
