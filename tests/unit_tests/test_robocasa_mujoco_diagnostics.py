import numpy as np

from rlinf.envs.robocasa.venv import build_mujoco_diagnostics_snapshot


class _FakeContact:
    def __init__(self, dist, geom1, geom2):
        self.dist = dist
        self.geom1 = geom1
        self.geom2 = geom2


class _FakeNamedItem:
    def __init__(self, name):
        self.name = name


class _FakeModel:
    nbody = 2
    ngeom = 3

    def body(self, idx):
        return _FakeNamedItem(f"body_{idx}")

    def geom(self, idx):
        return _FakeNamedItem(f"geom_{idx}")


class _FakeData:
    ncon = 2
    contact = [_FakeContact(-0.1, 0, 1), _FakeContact(0.2, 1, 2)]
    qvel = np.array([1.0, 2.0, 3.0])
    xpos = np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]])
    xquat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    subtree_linvel = np.array([[0.3, 0.4, 0.5], [1.3, 1.4, 1.5]])
    energy = np.array([12.0, 34.0])


def test_build_mujoco_diagnostics_snapshot_serializes_arrays_and_names():
    model = _FakeModel()
    data = _FakeData()

    snapshot = build_mujoco_diagnostics_snapshot(
        model,
        data,
        max_contacts=8,
        include_model_names=True,
        contact_force_fn=lambda model, data, idx: [idx, idx + 1, 0, 0, 0, 0],
    )

    assert snapshot["ncon"] == 2
    assert snapshot["contacts"][0] == {
        "dist": -0.1,
        "geom1": 0,
        "geom2": 1,
        "geom1_name": "geom_0",
        "geom2_name": "geom_1",
        "force": [0, 1, 0, 0, 0, 0],
    }
    assert snapshot["qvel"] == [1.0, 2.0, 3.0]
    assert snapshot["xpos"] == [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]
    assert snapshot["xquat"] == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert snapshot["subtree_linvel"] == [[0.3, 0.4, 0.5], [1.3, 1.4, 1.5]]
    assert snapshot["energy"] == [12.0, 34.0]
    assert snapshot["kinetic_energy"] == 12.0
    assert snapshot["potential_energy"] == 34.0
    assert snapshot["body_names"] == ["body_0", "body_1"]
    assert snapshot["geom_names"] == ["geom_0", "geom_1", "geom_2"]


def test_build_mujoco_diagnostics_snapshot_truncates_contacts():
    model = _FakeModel()
    data = _FakeData()

    snapshot = build_mujoco_diagnostics_snapshot(
        model,
        data,
        max_contacts=1,
        include_model_names=False,
        contact_force_fn=lambda model, data, idx: [idx, idx + 1, 0, 0, 0, 0],
    )

    assert snapshot["ncon"] == 2
    assert len(snapshot["contacts"]) == 1
    assert snapshot["contacts"][0]["geom1_name"] == ""
    assert snapshot["body_names"] == []
    assert snapshot["geom_names"] == []


def test_build_mujoco_diagnostics_snapshot_keeps_contact_when_force_fails():
    model = _FakeModel()
    data = _FakeData()

    def _raise_force_error(model, data, idx):
        raise RuntimeError("force unavailable")

    snapshot = build_mujoco_diagnostics_snapshot(
        model,
        data,
        max_contacts=1,
        include_model_names=True,
        contact_force_fn=_raise_force_error,
    )

    assert snapshot["contacts"][0]["force"] is None
    assert snapshot["contacts"][0]["force_error"] == "force unavailable"
    assert snapshot["contacts"][0]["dist"] == -0.1


def test_robocasa_subproc_env_get_mujoco_diagnostics_calls_workers() -> None:
    from rlinf.envs.robocasa.venv import RobocasaSubprocEnv

    env = RobocasaSubprocEnv.__new__(RobocasaSubprocEnv)

    class _Worker:
        def __init__(self, value):
            self.value = value

        def get_mujoco_diagnostics(self, max_contacts, include_model_names):
            return {
                "value": self.value,
                "max_contacts": max_contacts,
                "include_model_names": include_model_names,
            }

    env.workers = [_Worker(1), _Worker(2)]

    assert env.get_mujoco_diagnostics(max_contacts=4, include_model_names=False) == [
        {"value": 1, "max_contacts": 4, "include_model_names": False},
        {"value": 2, "max_contacts": 4, "include_model_names": False},
    ]
