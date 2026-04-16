from __future__ import annotations

from types import SimpleNamespace


def test_ensure_vector_autoreset_mode_populates_missing_export() -> None:
    from rlinf.envs.gymnasium_compat import ensure_vector_autoreset_mode

    class _FallbackAutoresetMode:
        NEXT_STEP = "NEXT_STEP"

    fake_vector = SimpleNamespace()
    fake_vector_env_module = SimpleNamespace(AutoresetMode=_FallbackAutoresetMode)
    fake_gym = SimpleNamespace(vector=fake_vector)

    ensure_vector_autoreset_mode(fake_gym, vector_env_module=fake_vector_env_module)

    assert hasattr(fake_gym.vector, "AutoresetMode")
    assert fake_gym.vector.AutoresetMode is _FallbackAutoresetMode


def test_ensure_vector_autoreset_mode_keeps_existing_export() -> None:
    from rlinf.envs.gymnasium_compat import ensure_vector_autoreset_mode

    class _ExistingAutoresetMode:
        SAME_STEP = "SAME_STEP"

    fake_vector = SimpleNamespace(AutoresetMode=_ExistingAutoresetMode)
    fake_gym = SimpleNamespace(vector=fake_vector)

    ensure_vector_autoreset_mode(fake_gym)

    assert fake_gym.vector.AutoresetMode is _ExistingAutoresetMode


def test_ensure_vector_autoreset_mode_creates_fallback_when_unavailable() -> None:
    from rlinf.envs.gymnasium_compat import ensure_vector_autoreset_mode

    fake_vector = SimpleNamespace()
    fake_gym = SimpleNamespace(vector=fake_vector)
    fake_vector_env_module = SimpleNamespace()

    ensure_vector_autoreset_mode(fake_gym, vector_env_module=fake_vector_env_module)

    assert hasattr(fake_gym.vector, "AutoresetMode")
    assert fake_gym.vector.AutoresetMode.SAME_STEP.name == "SAME_STEP"
