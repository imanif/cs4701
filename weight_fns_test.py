from absl.testing import absltest
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import weight_fns


class WeightFnTest(absltest.TestCase):
    def test_hat_normalize(self):
        blank = torch.tensor([2.0, 7.0])
        lexical = torch.tensor([[0.0, 1.0], [3.0, 5.0]])
        expect_blank = torch.tensor([-0.126928, -0.000912])
        expect_lexical = torch.tensor([[-3.44019, -2.44019], [-9.12784, -7.12784]])
        actual_blank, actual_lexical = weight_fns.hat_normalize(blank, lexical)
        torch.testing.assert_close(actual_blank, expect_blank, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(actual_lexical, expect_lexical, rtol=1e-6, atol=1e-6)

    def test_log_softmax_normalize(self):
        blank = torch.tensor([2.0, 7.0])
        lexical = torch.tensor([[0.0, 1.0], [3.0, 5.0]])
        expect_blank = torch.tensor([-0.407606, -0.142932])
        expect_lexical = torch.tensor([[-2.407606, -1.407606], [-4.142932, -2.142932]])
        actual_blank, actual_lexical = weight_fns.log_softmax_normalize(blank, lexical)
        torch.testing.assert_close(actual_blank, expect_blank, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(actual_lexical, expect_lexical, rtol=1e-6, atol=1e-6)

    def test_NullCacher(self):
        # Get cache and inspect for trainable paramaters
        cacher = weight_fns.NullCacher()
        num_params: int = sum(p.numel() for p in cacher.parameters() if p.requires_grad)
        self.assertEqual(num_params, 0)
        self.assertIsNone(cacher())

    def test_TableWeightFn(self):
        with self.subTest("batch ndim = 0"):
            table = torch.arange(5 * 4 * 3).reshape([5, 4, 3])
            weight_fn = weight_fns.TableWeightFn(table)

            frame = torch.tensor([1.0, 2.0])
            (blank, lexical) = weight_fn(None, frame)
            num_params: int = sum(
                p.numel() for p in weight_fn.parameters() if p.requires_grad
            )
            self.assertEqual(num_params, 0)
            torch.testing.assert_close(blank, table[1, :, 0])
            torch.testing.assert_close(lexical, table[1, :, 1:])

            state = torch.tensor(3)
            blank, lexical = weight_fn(None, frame, state)
            torch.testing.assert_close(blank, table[1, 3, 0])
            torch.testing.assert_close(lexical, table[1, 3, 1:])

            with self.assertRaisesRegex(
                ValueError,
                r"frame should have batch_dims=\(\) but got torch.Size\(\[1\]\)",
            ):
                weight_fn(None, frame[None])

        with self.subTest("batch ndim = 1"):
            table = torch.arange(2 * 5 * 4 * 3).reshape([2, 5, 4, 3])
            weight_fn = weight_fns.TableWeightFn(table)

            frame = torch.tensor([[1.0, 2.0], [4.0, 3.0]])
            (blank, lexical) = weight_fn(None, frame)
            num_params: int = sum(
                p.numel() for p in weight_fn.parameters() if p.requires_grad
            )
            self.assertEqual(num_params, 0)
            torch.testing.assert_close(
                blank, torch.stack((table[0, 1, :, 0], table[1, 4, :, 0]))
            )
            torch.testing.assert_close(
                lexical, torch.stack((table[0, 1, :, 1:], table[1, 4, :, 1:]))
            )

            state = torch.tensor([3, 2])
            blank, lexical = weight_fn(None, frame, state)
            torch.testing.assert_close(
                blank, torch.stack((table[0, 1, 3, 0], table[1, 4, 2, 0]))
            )
            torch.testing.assert_close(
                lexical, torch.stack((table[0, 1, 3, 1:], table[1, 4, 2, 1:]))
            )

            with self.assertRaisesRegex(
                ValueError,
                r"frame should have batch_dims=\(2,\) but got torch.Size\(\[1, 2\]\)",
            ):
                weight_fn(None, frame[None])


class LocallyNormalizedWeightFnTest(absltest.TestCase):
    def test_call(self):
        weight_fn = weight_fns.LocallyNormalizedWeightFn(
            weight_fns.JointWeightFn(vocab_size=3, hidden_size=8)
        )
        rngs = npr.default_rng(0).integers(low=0, high=1e9, size=5)
        frame = torch.from_numpy(
            npr.default_rng(rngs[0]).uniform(high=1, size=(2, 4)).astype(np.float32)
        )
        cache = torch.from_numpy(
            npr.default_rng(rngs[1]).uniform(high=1, size=(6, 5)).astype(np.float32)
        )  # context embeddings.

        with self.subTest("all context states"):
            blank, lexical = weight_fn(cache, frame)
            torch.testing.assert_close(blank.shape, torch.Size([2, 6]))
            torch.testing.assert_close(lexical.shape, torch.Size([2, 6, 3]))
            torch.testing.assert_close(
                torch.exp(blank) + torch.sum(torch.exp(lexical), dim=-1),
                torch.ones_like(blank),
                rtol=1e-4,
                atol=1e-4,
            )

        with self.subTest("per-state"):
            state = torch.tensor([2, 4])
            blank_per_state, lexical_per_state = weight_fn(cache, frame, state)
            torch.testing.assert_close(
                blank_per_state,
                blank[torch.tensor([0, 1]), state],
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                lexical_per_state,
                lexical[torch.tensor([0, 1]), state],
                rtol=1e-6,
                atol=1e-6,
            )


class JointWeightFnTest(absltest.TestCase):
    def test_call(self):
        weight_fn = weight_fns.JointWeightFn(vocab_size=3, hidden_size=8)
        rngs = npr.default_rng(0).integers(low=0, high=1e9, size=5)
        frame = torch.from_numpy(
            npr.default_rng(rngs[0]).uniform(high=1, size=(2, 4)).astype(np.float32)
        )
        cache = torch.from_numpy(
            npr.default_rng(rngs[1]).uniform(high=1, size=(6, 5)).astype(np.float32)
        )  # context embeddings.

        with self.subTest("all context states"):
            blank, lexical = weight_fn(cache, frame)
            torch.testing.assert_close(blank.shape, torch.Size([2, 6]))
            torch.testing.assert_close(lexical.shape, torch.Size([2, 6, 3]))

        with self.subTest("per-state"):
            state = torch.tensor([2, 4])
            blank_per_state, lexical_per_state = weight_fn(cache, frame, state)
            torch.testing.assert_close(
                blank_per_state,
                blank[torch.tensor([0, 1]), state],
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                lexical_per_state,
                lexical[torch.tensor([0, 1]), state],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_SharedEmbCacher(self):
        cacher = weight_fns.SharedEmbCacher(num_context_states=4, embedding_size=5)
        cacher()
        param = cacher.get_parameter("context_embeddings")
        torch.testing.assert_close(param.size(), torch.Size([4, 5]))
        torch.testing.assert_close(param, cacher())

    def test_SharedRNNCacher(self):
        pad = -2.0
        start = -1.0

        class FakeRNNCell(nn.RNNCellBase):
            """Test RNN cell that remembers past inputs."""

            def __init__(self, features: int) -> None:
                self.features = features

            @property
            def num_features(self):
                return 1

            def __call__(self, carry, inputs):
                carry = torch.cat([carry[..., 1:], inputs[..., :1]], dim=-1)
                return carry, carry

            @staticmethod
            def initialize_carry(self, size):
                batch_dims = size[:1]
                return torch.full((*batch_dims, self.features), pad)

        params = nn.Parameter(
            torch.broadcast_to(torch.tensor([start, 1.0, 2.0, 3.0])[:, None], (4, 6))
        )

        with self.subTest("context_size=2"):
            cacher = weight_fns.SharedRNNCacher(
                vocab_size=3,
                context_size=2,
                rnn_size=4,
                rnn_embedding_size=6,
                rnn_cell=FakeRNNCell(features=4),
            )
            cacher.embed.weight = params
            torch.testing.assert_close(
                cacher(),
                torch.tensor(
                    [
                        # Start.
                        [pad, pad, pad, start],
                        # Unigrams.
                        [pad, pad, start, 1],
                        [pad, pad, start, 2],
                        [pad, pad, start, 3],
                        # Bigrams.
                        [pad, start, 1, 1],
                        [pad, start, 1, 2],
                        [pad, start, 1, 3],
                        [pad, start, 2, 1],
                        [pad, start, 2, 2],
                        [pad, start, 2, 3],
                        [pad, start, 3, 1],
                        [pad, start, 3, 2],
                        [pad, start, 3, 3],
                    ]
                ),
            )

        with self.subTest("context_size=0"):
            cacher = weight_fns.SharedRNNCacher(
                vocab_size=3,
                context_size=0,
                rnn_size=4,
                rnn_embedding_size=6,
                rnn_cell=FakeRNNCell(features=4),
            )
            cacher.embed.weight = params
            torch.testing.assert_close(
                cacher(),
                torch.tensor([[pad, pad, pad, start]]),
            )


if __name__ == "__main__":
    absltest.main()
