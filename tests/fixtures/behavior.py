import pytest
from agent_torch.models import covid
from agent_torch.populations import astoria
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.environment import envs
from tests.mocks.llm import MockLLMBackend

@pytest.fixture
def mock_llm():
    """Returns a simple mock LLM instance"""
    return MockLLMBackend()

@pytest.fixture
def archetype():
    """Returns a base archetype instance"""
    return Archetype(n_arch=7)

@pytest.fixture
def test_population():
    """Returns a population instance"""
    return astoria

@pytest.fixture
def isolation_archetype(archetype, mock_llm):
    """Returns an archetype configured with mock LLM"""
    test_prompt = "Test prompt for {age} {gender} isolation behavior decision making"
    return archetype.llm(llm=mock_llm, user_prompt=test_prompt)

@pytest.fixture
def env_with_behavior(isolation_archetype):
    """Returns environment configured with isolation behavior"""
    return envs.create(
        model=covid,
        population=astoria,
        archetypes={'make_isolation_decision': isolation_archetype}
    )