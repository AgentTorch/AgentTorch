import pytest
from contextlib import contextmanager
from agent_torch.core.decorators import with_behavior
from agent_torch.core.llm.behavior import Behavior
from fixtures.behavior import (
    archetype,
    mock_llm,
    env_with_behavior,
    isolation_archetype,
    test_population,
)


@contextmanager
def not_raises():
    try:
        yield
    except Exception as exception:
        raise pytest.fail(f"Raised {exception}.")


def test_mock_llm_response(mock_llm):
    """Test that mock LLM returns expected values"""
    assert mock_llm.prompt("test") == "0.5"
    assert mock_llm.prompt(["test1", "test2"]) == ["0.5", "0.5"]


def test_behavior_decorator(isolation_archetype, test_population):
    """Test the behavior decorator functionality"""

    @with_behavior
    class TestClass:
        def __init__(self):
            pass

    test_instance = TestClass()
    assert test_instance.behavior is None

    test_behavior = Behavior(archetype=isolation_archetype, region=test_population)
    TestClass.set_behavior(test_behavior)

    assert TestClass._class_behavior == test_behavior
    new_instance = TestClass()
    assert new_instance.behavior == test_behavior


def test_environment_behavior_setup(env_with_behavior):
    """Test environment creation and behavior setup"""
    with not_raises():
        env_with_behavior.init()
        registry = env_with_behavior.registry.helpers

        # Find and verify isolation decision behavior
        isolation_decision = None
        for category in registry.values():
            for name, obj in category.items():
                if name == "make_isolation_decision":
                    isolation_decision = obj
                    break

        assert isolation_decision is not None
        assert isolation_decision.behavior is not None


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
