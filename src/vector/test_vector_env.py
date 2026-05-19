import unittest

import numpy as np
import torch

from src.vector import VectorizedMultiAgentEnv, select_device


class VectorizedEnvTests(unittest.TestCase):
    def test_crossing_orders_fill_and_update_cash_inventory(self):
        env = VectorizedMultiAgentEnv(n_agents=2, n_envs=4, max_steps=4, device="cpu", return_tensors=True)
        env.reset(seed=1)
        obs, rewards, dones, truncs, infos = env.step(
            {
                0: torch.tensor([[0.0, 1.0, 10.0]]).expand(4, -1),
                1: torch.tensor([[1.0, -1.0, 10.0]]).expand(4, -1),
            }
        )

        self.assertEqual(obs[0].shape, (4, 24))
        self.assertTrue((env.last_execution["taker_fill_qty"][:, 0] > 0).all())
        self.assertTrue(torch.isfinite(env.last_execution["taker_avg_price"]).all())
        self.assertTrue(torch.isfinite(env.cash).all())
        self.assertTrue(torch.isfinite(env.inventory).all())
        self.assertTrue(torch.isfinite(rewards[0]).all())

    def test_passive_external_flow_can_fill_agent_quote(self):
        env = VectorizedMultiAgentEnv(n_agents=1, n_envs=4, max_steps=8, device="cpu", return_tensors=True)
        env.reset(seed=2)
        env.step({0: torch.tensor([[0.0, -1.0, 50.0]]).expand(4, -1)})

        mid = env.book.mid_indices()
        result = env.book.execute_batch(
            torch.ones(4, dtype=torch.long),
            (mid - 100).clamp(0, env.book.n_price_levels - 1),
            torch.full((4,), 25.0),
            taker_agent=None,
        )

        self.assertTrue((result.passive_agent_fill_qty[:, 0] > 0).any())
        self.assertTrue((result.passive_agent_cash[:, 0] < 0).any())

    def test_self_match_prevention_excludes_own_quote(self):
        env = VectorizedMultiAgentEnv(n_agents=1, n_envs=3, max_steps=4, device="cpu", return_tensors=True)
        env.reset(seed=5)
        env.book.background_bids.zero_()
        env.book.background_asks.zero_()
        env.book.agent_asks[:, 0, env.book.center_idx] = 10.0

        result = env.book.execute_batch(
            torch.zeros(3, dtype=torch.long),
            torch.full((3,), env.book.center_idx, dtype=torch.long),
            torch.full((3,), 10.0),
            taker_agent=0,
        )

        self.assertTrue((result.fill_qty == 0).all())
        self.assertTrue((env.book.agent_asks[:, 0, env.book.center_idx] == 10.0).all())

    def test_cancel_replace_keeps_one_quote_per_agent(self):
        env = VectorizedMultiAgentEnv(n_agents=1, n_envs=2, max_steps=4, device="cpu", return_tensors=True)
        env.reset(seed=3)
        env.step({0: torch.tensor([[0.0, -1.0, 20.0], [0.0, -1.0, 20.0]])})
        first_count = (env.book.agent_bids[:, 0, :] > 0).sum(dim=1)
        env.step({0: torch.tensor([[0.0, -2.0, 20.0], [0.0, -2.0, 20.0]])})
        second_count = (env.book.agent_bids[:, 0, :] > 0).sum(dim=1)

        self.assertTrue((first_count <= 1).all())
        self.assertTrue((second_count <= 1).all())

    def test_seed_is_deterministic_on_cpu(self):
        env_a = VectorizedMultiAgentEnv(n_agents=2, n_envs=3, max_steps=3, device="cpu", return_tensors=False)
        env_b = VectorizedMultiAgentEnv(n_agents=2, n_envs=3, max_steps=3, device="cpu", return_tensors=False)
        obs_a, _ = env_a.reset(seed=4)
        obs_b, _ = env_b.reset(seed=4)

        np.testing.assert_array_equal(obs_a[0], obs_b[0])

    def test_device_selection_falls_back(self):
        device = select_device("mps")
        self.assertIn(device.type, {"mps", "cpu"})


if __name__ == "__main__":
    unittest.main()
