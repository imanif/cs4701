from absl.testing import absltest
import semirings
import contexts
import torch


class FullNGramTest(absltest.TestCase):
    def test_invalid_args(self):
        with self.assertRaisesRegex(ValueError, "vocab_size should be > 0"):
            contexts.FullNGram(vocab_size=0, context_size=1)
        with self.assertRaisesRegex(ValueError, "context_size should be >= 0"):
            contexts.FullNGram(vocab_size=1, context_size=-1)

    def test_invalid_inputs(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        with self.assertRaisesRegex(
            ValueError, r"weights\.shape\[-2:\] should be \(3, 2\)"
        ):
            context.forward_reduce(torch.zeros([3, 4]), semirings.Real)
        with self.assertRaisesRegex(ValueError, r"weights\.shape\[-1\] should be 3"):
            context.backward_broadcast(torch.zeros([4]))

    def test_context_size_0_basics(self):
        context = contexts.FullNGram(vocab_size=3, context_size=0)
        self.assertEqual(context.num_states(), 1)
        self.assertEqual(context.shape(), (1, 3))
        self.assertEqual(context.start(), 0)

    def test_context_size_0_next_state(self):
        context = contexts.FullNGram(vocab_size=3, context_size=0)
        torch.testing.assert_close(
            context.next_state(torch.tensor(0), torch.tensor(1)), torch.tensor(0)
        )
        torch.testing.assert_close(
            context.next_state(torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2])),
            torch.tensor([0, 0, 0]),
        )
        torch.testing.assert_close(
            context.next_state(torch.tensor([[0, 0, 0]]), torch.tensor([[0, 1, 2]])),
            torch.tensor([[0, 0, 0]]),
        )
        # Epsilon transitions.
        torch.testing.assert_close(
            context.next_state(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0])),
            torch.tensor([0, 1, 2]),
        )

    def test_context_size_0_forward_reduce(self):
        context = contexts.FullNGram(vocab_size=3, context_size=0)
        torch.testing.assert_close(
            context.forward_reduce(torch.tensor([[1, 2, 3]]), semirings.Real),
            torch.tensor([6]),
        )
        torch.testing.assert_close(
            context.forward_reduce(torch.arange(6).reshape((2, 1, 3)), semirings.Real),
            torch.tensor([[3], [12]]),
        )
        torch.testing.assert_close(
            context.forward_reduce(
                torch.arange(6).reshape((1, 2, 1, 3)), semirings.Real
            ),
            torch.tensor([[[3], [12]]]),
        )

    def test_context_size_0_backward_broadcast(self):
        context = contexts.FullNGram(vocab_size=3, context_size=0)
        torch.testing.assert_close(
            context.backward_broadcast(torch.tensor([1])), torch.tensor([[1, 1, 1]])
        )
        torch.testing.assert_close(
            context.backward_broadcast(torch.tensor([[1], [2]])),
            torch.tensor([[[1, 1, 1]], [[2, 2, 2]]]),
        )
        torch.testing.assert_close(
            context.backward_broadcast(torch.tensor([[[1], [2]]])),
            torch.tensor([[[[1, 1, 1]], [[2, 2, 2]]]]),
        )

    def test_context_size_1_basics(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        self.assertEqual(context.num_states(), 3)
        self.assertEqual(context.shape(), (3, 2))
        self.assertEqual(context.start(), 0)

    def test_context_size_1_next_state(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        torch.testing.assert_close(
            context.next_state(torch.tensor(0), torch.tensor(1)), torch.tensor(1)
        )
        torch.testing.assert_close(
            context.next_state(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 1])),
            torch.tensor([1, 2, 1]),
        )
        torch.testing.assert_close(
            context.next_state(torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 1]])),
            torch.tensor([[1, 2, 1]]),
        )
        # Epsilon transitions.
        torch.testing.assert_close(
            context.next_state(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0])),
            torch.tensor([0, 1, 2]),
        )

    def test_context_size_1_forward_reduce(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        torch.testing.assert_close(
            context.forward_reduce(torch.arange(6).reshape((3, 2)), semirings.Real),
            torch.tensor([0, 0 + 2 + 4, 1 + 3 + 5]),
        )
        torch.testing.assert_close(
            context.forward_reduce(torch.arange(6).reshape((1, 3, 2)), semirings.Real),
            torch.tensor([[0, 0 + 2 + 4, 1 + 3 + 5]]),
        )
        torch.testing.assert_close(
            context.forward_reduce(
                torch.arange(6).reshape((1, 1, 3, 2)), semirings.Real
            ),
            torch.tensor([[[0, 0 + 2 + 4, 1 + 3 + 5]]]),
        )

    def test_context_size_1_backward_broadcast(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        torch.testing.assert_close(
            context.backward_broadcast(torch.arange(3)),
            torch.tensor([[1, 2], [1, 2], [1, 2]]),
        )
        torch.testing.assert_close(
            context.backward_broadcast(torch.arange(3).reshape((1, 3))),
            torch.tensor([[[1, 2], [1, 2], [1, 2]]]),
        )
        torch.testing.assert_close(
            context.backward_broadcast(torch.arange(3).reshape((1, 1, 3))),
            torch.tensor([[[[1, 2], [1, 2], [1, 2]]]]),
        )

    def test_context_size_2_basics(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        self.assertEqual(context.num_states(), 13)
        self.assertEqual(context.shape(), (13, 3))
        self.assertEqual(context.start(), 0)

    def test_context_size_2_next_state(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        torch.testing.assert_close(
            context.next_state(
                torch.tensor([0, 1, 3, 4, 12]), torch.tensor([1, 2, 3, 1, 2])
            ),
            torch.tensor([1, 5, 12, 4, 11]),
        )
        # Epsilon transitions.
        torch.testing.assert_close(
            context.next_state(
                torch.tensor([0, 1, 3, 4, 12]), torch.tensor([0, 0, 0, 0, 0])
            ),
            torch.tensor([0, 1, 3, 4, 12]),
        )

    def test_context_size_2_forward_reduce(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        torch.testing.assert_close(
            context.forward_reduce(
                torch.arange(39).reshape((1, 13, 3)), semirings.Real
            ),
            torch.tensor(
                [
                    [
                        0,
                        0,
                        1,
                        2,
                        3 * 4 + 54,
                        4 * 4 + 54,
                        5 * 4 + 54,
                        6 * 4 + 54,
                        7 * 4 + 54,
                        8 * 4 + 54,
                        9 * 4 + 54,
                        10 * 4 + 54,
                        11 * 4 + 54,
                    ]
                ]
            ),
        )

    def test_context_size_2_backward_broadcast(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        torch.testing.assert_close(
            context.backward_broadcast(torch.arange(13).reshape((1, 13))),
            torch.tensor([[[1, 2, 3]] + [[4, 5, 6], [7, 8, 9], [10, 11, 12]] * 4]),
        )

    def test_walk_states(self):
        context = contexts.FullNGram(vocab_size=3, context_size=2)
        self.assertEqual(
            context.walk_states(torch.zeros([2, 3, 4], dtype=torch.int32)).shape,
            (2, 3, 5),
        )
        torch.testing.assert_close(
            context.walk_states(torch.tensor([2, 3, 1])), torch.tensor([0, 2, 9, 10])
        )
        # Epsilon transitions.
        torch.testing.assert_close(
            context.walk_states(torch.tensor([2, 0, 0, 3, 1])),
            torch.tensor([0, 2, 2, 2, 9, 10]),
        )


if __name__ == "__main__":
    absltest.main()
