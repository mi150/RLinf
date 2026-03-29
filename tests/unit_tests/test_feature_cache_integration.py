import torch

from rlinf.models.embodiment.feature_cache import FeatureCache, FeatureCacheConfig


class DummyBackboneSplitModel:
    def __init__(self, cache: FeatureCache | None = None):
        self.feature_cache = cache
        self.backbone_calls = 0
        self.action_head_calls = 0

    def _run_backbone(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        self.backbone_calls += 1
        return obs["main_images"].float().mean(dim=(-1, -2), keepdim=False)

    def _run_action_head(self, feat: torch.Tensor) -> torch.Tensor:
        self.action_head_calls += 1
        return feat.sum(dim=-1, keepdim=True)

    def predict_action_batch(
        self,
        env_obs: dict[str, torch.Tensor],
        env_seeds: torch.Tensor,
        step_indices: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for b in range(env_seeds.shape[0]):
            seed = int(env_seeds[b].item())
            step = int(step_indices[b].item())
            obs_b = {"main_images": env_obs["main_images"][b : b + 1]}
            hit = False
            feat = None
            if self.feature_cache is not None and self.feature_cache.config.enabled:
                cached, hit = self.feature_cache.get(seed, step, current_obs=obs_b)
                if hit:
                    feat = cached["feat"]
            if not hit:
                feat = self._run_backbone(obs_b)
                if self.feature_cache is not None and self.feature_cache.config.enabled:
                    self.feature_cache.put(seed, step, features={"feat": feat}, obs=obs_b)
            outputs.append(self._run_action_head(feat))
        return torch.cat(outputs, dim=0)


def _run_epoch(
    model: DummyBackboneSplitModel,
    env_seeds: torch.Tensor,
    step_counter: dict[int, int],
    done_mask_by_step: list[torch.Tensor] | None = None,
):
    for step_id in range(3):
        if done_mask_by_step is not None:
            done_mask = done_mask_by_step[step_id]
            for i, done in enumerate(done_mask.tolist()):
                if done:
                    step_counter[int(env_seeds[i].item())] = 0

        step_indices = torch.as_tensor(
            [step_counter.get(int(seed.item()), 0) for seed in env_seeds],
            dtype=torch.int64,
        )
        obs = {"main_images": torch.ones(env_seeds.shape[0], 3, 8, 8) * (step_id + 1)}
        _ = model.predict_action_batch(obs, env_seeds, step_indices)
        for seed in env_seeds.tolist():
            seed_int = int(seed)
            step_counter[seed_int] = step_counter.get(seed_int, 0) + 1


def test_two_epoch_rollout_hits_with_fixed_seeds():
    cache = FeatureCache(FeatureCacheConfig(enabled=True, mode="naive"))
    model = DummyBackboneSplitModel(cache)
    env_seeds = torch.tensor([100, 101], dtype=torch.int64)
    step_counter: dict[int, int] = {}

    _run_epoch(model, env_seeds, step_counter)
    first_epoch_backbone_calls = model.backbone_calls
    assert first_epoch_backbone_calls == 6

    # Fixed seeds + same step patterns should fully hit.
    for seed in env_seeds.tolist():
        step_counter[int(seed)] = 0
    _run_epoch(model, env_seeds, step_counter)
    assert model.backbone_calls == first_epoch_backbone_calls
    stats = cache.get_stats()
    assert stats.hits >= 6


def test_similarity_gated_mode_with_obs_difference():
    cache = FeatureCache(
        FeatureCacheConfig(
            enabled=True,
            mode="similarity_gated",
            similarity_metric="obs_cosine",
            similarity_threshold=0.999,
        )
    )
    model = DummyBackboneSplitModel(cache)
    seed = torch.tensor([200], dtype=torch.int64)
    step_idx = torch.tensor([0], dtype=torch.int64)

    obs_a = {"main_images": torch.ones(1, 3, 8, 8)}
    _ = model.predict_action_batch(obs_a, seed, step_idx)
    calls_after_a = model.backbone_calls

    obs_b = {"main_images": torch.zeros(1, 3, 8, 8)}
    _ = model.predict_action_batch(obs_b, seed, step_idx)
    assert model.backbone_calls == calls_after_a + 1


def test_cache_invalidation_on_weight_update():
    cache = FeatureCache(FeatureCacheConfig(enabled=True, mode="naive"))
    model = DummyBackboneSplitModel(cache)
    seed = torch.tensor([1], dtype=torch.int64)
    step = torch.tensor([0], dtype=torch.int64)
    obs = {"main_images": torch.ones(1, 3, 8, 8)}

    _ = model.predict_action_batch(obs, seed, step)
    _ = model.predict_action_batch(obs, seed, step)
    assert model.backbone_calls == 1

    cache.invalidate_all()
    _ = model.predict_action_batch(obs, seed, step)
    assert model.backbone_calls == 2


def test_step_counter_reset_on_done():
    cache = FeatureCache(FeatureCacheConfig(enabled=True, mode="naive"))
    model = DummyBackboneSplitModel(cache)
    env_seeds = torch.tensor([9, 10], dtype=torch.int64)
    step_counter: dict[int, int] = {}
    done_masks = [
        torch.tensor([False, False]),
        torch.tensor([True, False]),
        torch.tensor([False, False]),
    ]

    _run_epoch(model, env_seeds, step_counter, done_masks)
    # seed 9 was reset at step 1, seed 10 kept increasing.
    assert step_counter[9] == 2
    assert step_counter[10] == 3
