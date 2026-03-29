import torch

from rlinf.models.embodiment.feature_cache import (
    FeatureCache,
    FeatureCacheConfig,
    _compute_cosine_similarity,
    _compute_ssim,
    _recursive_to_cpu_pinned,
    _recursive_to_device,
)


def _make_obs(offset: int = 0) -> dict[str, torch.Tensor]:
    img = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    return {"main_images": img + offset}


def test_put_get_naive_mode():
    cache = FeatureCache(FeatureCacheConfig(enabled=True, mode="naive"))
    features = {"x": torch.randn(1, 4), "kv": (torch.randn(1, 2), torch.randn(1, 2))}
    obs = _make_obs()

    cache.put(seed=1, step=0, features=features, obs=obs)
    loaded, hit = cache.get(seed=1, step=0, current_obs=obs)

    assert hit is True
    assert loaded is not None
    assert torch.allclose(loaded["x"], features["x"])
    stats = cache.get_stats()
    assert stats.hits == 1
    assert stats.misses == 0


def test_similarity_gated_mode():
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_gated",
            similarity_metric="obs_cosine",
            similarity_threshold=0.99,
        )
    )
    features = {"x": torch.ones(1, 2)}
    obs_a = _make_obs(0)
    obs_b = _make_obs(0)
    obs_c = _make_obs(1000)

    cache.put(seed=7, step=3, features=features, obs=obs_a)
    _, hit_b = cache.get(seed=7, step=3, current_obs=obs_b)
    _, hit_c = cache.get(seed=7, step=3, current_obs=obs_c)

    assert hit_b is True
    assert hit_c is False


def test_recursive_transfer_for_nested_kv_like_data():
    nested = {
        "past_key_values": (
            (torch.randn(1, 2, 3), torch.randn(1, 2, 3)),
            (torch.randn(1, 2, 3), torch.randn(1, 2, 3)),
        ),
        "list_part": [torch.randn(1, 2), {"a": torch.randn(1, 1)}],
    }
    cpu_pinned = _recursive_to_cpu_pinned(nested)
    restored = _recursive_to_device(cpu_pinned, torch.device("cpu"))

    assert isinstance(restored["past_key_values"], tuple)
    assert isinstance(restored["past_key_values"][0], tuple)
    assert restored["past_key_values"][0][0].device.type == "cpu"
    assert restored["list_part"][1]["a"].shape == (1, 1)


def test_invalidation_and_stats():
    cache = FeatureCache(FeatureCacheConfig(enabled=True, mode="naive"))
    cache.put(seed=1, step=0, features={"x": torch.ones(1, 1)}, obs=_make_obs())
    cache.put(seed=2, step=0, features={"x": torch.ones(1, 1)}, obs=_make_obs())

    cache.invalidate(seed=1)
    _, hit = cache.get(seed=1, step=0, current_obs=_make_obs())
    assert hit is False

    cache.invalidate_all()
    _, hit = cache.get(seed=2, step=0, current_obs=_make_obs())
    assert hit is False
    assert cache.get_stats().invalidations >= 2


def test_max_cache_seeds_eviction():
    cache = FeatureCache(
        FeatureCacheConfig(enabled=True, mode="naive", max_cache_seeds=1)
    )
    cache.put(seed=10, step=0, features={"x": torch.ones(1, 1)}, obs=_make_obs())
    cache.put(seed=11, step=0, features={"x": torch.ones(1, 1)}, obs=_make_obs())

    _, hit_old = cache.get(seed=10, step=0, current_obs=_make_obs())
    _, hit_new = cache.get(seed=11, step=0, current_obs=_make_obs())
    assert hit_old is False
    assert hit_new is True


def test_similarity_functions():
    a = torch.ones(1, 3, 16, 16)
    b = torch.ones(1, 3, 16, 16)
    c = torch.zeros(1, 3, 16, 16)
    assert _compute_ssim(a, b) > 0.99
    assert _compute_ssim(a, c) < 0.1

    x = torch.tensor([1.0, 0.0, 0.0])
    y = torch.tensor([1.0, 0.0, 0.0])
    z = torch.tensor([0.0, 1.0, 0.0])
    assert _compute_cosine_similarity(x, y) > 0.99
    assert _compute_cosine_similarity(x, z) < 0.1
