from absl.testing import absltest
import torch
import numpy.random as npr
import alignments
import contexts
import semirings


class AlignmentsTest(absltest.TestCase):
    def test_shift_down(self):
        torch.testing.assert_close(
            alignments.shift_down(torch.tensor([1, 2, 3]), semirings.Real),
            torch.tensor([0, 1, 2]),
        )
        torch.testing.assert_close(
            alignments.shift_down(torch.tensor([[1, 2, 3], [4, 5, 6]]), semirings.Real),
            torch.tensor([[0, 1, 2], [0, 4, 5]]),
        )
        torch.testing.assert_close(
            alignments.shift_down(
                torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32), semirings.Log
            ),
            torch.tensor([[-torch.inf, 1, 2], [-torch.inf, 4, 5]]),
        )


class FrameDependentTest(absltest.TestCase):
    def test_topology(self):
        alignment = alignments.FrameDependent()
        self.assertEqual(alignment.num_states(), 1)
        self.assertEqual(alignment.start(), 0)
        self.assertEqual(alignment.blank_next(0), 0)
        self.assertEqual(alignment.lexical_next(0), 0)
        self.assertListEqual(alignment.topological_visit(), [0])

    def test_forward(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        alignment = alignments.FrameDependent()
        rngs = npr.default_rng(0).integers(low=0, high=1e9, size=(3, 2))
        alpha = torch.from_numpy(npr.default_rng(rngs[0]).uniform(high=1, size=3))
        blank = torch.from_numpy(npr.default_rng(rngs[1]).uniform(high=1, size=3))
        lexical = torch.from_numpy(
            npr.default_rng(rngs[2]).uniform(high=1, size=(3, 2))
        )

        # Single.
        next_alpha = alignment.forward(
            alpha=alpha,
            blank=[blank],
            lexical=[lexical],
            context=context,
            semiring=semirings.Real,
        )
        torch.testing.assert_close(
            next_alpha,
            torch.tensor(
                [
                    alpha[0] * blank[0],
                    alpha[1] * blank[1] + torch.sum(alpha * lexical[:, 0]),
                    alpha[2] * blank[2] + torch.sum(alpha * lexical[:, 1]),
                ]
            ),
        )

        # Batched.
        batched_next_alpha = alignment.forward(
            alpha=alpha[None],
            blank=[blank[None]],
            lexical=[lexical[None]],
            context=context,
            semiring=semirings.Real,
        )
        torch.testing.assert_close(batched_next_alpha, next_alpha[None])

        # Wrong number of weights.
        with self.assertRaisesRegex(ValueError, "blank should be"):
            alignment.forward(
                alpha=alpha,
                blank=[blank, blank],
                lexical=[lexical],
                context=context,
                semiring=semirings.Real,
            )
        with self.assertRaisesRegex(ValueError, "lexical should be"):
            alignment.forward(
                alpha=alpha,
                blank=[blank],
                lexical=[lexical, lexical],
                context=context,
                semiring=semirings.Real,
            )

    def test_backward(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        alignment = alignments.FrameDependent()
        rngs = npr.default_rng(0).integers(low=0, high=1e9, size=(5, 2))
        alpha = torch.from_numpy(npr.default_rng(rngs[0]).uniform(high=1, size=3))
        blank = torch.from_numpy(npr.default_rng(rngs[1]).uniform(high=1, size=3))
        lexical = torch.from_numpy(
            npr.default_rng(rngs[2]).uniform(high=1, size=(3, 2))
        )
        beta = torch.from_numpy(npr.default_rng(rngs[3]).uniform(high=1, size=3))
        z = torch.from_numpy(npr.default_rng(rngs[4]).uniform(high=1, size=()))

        # backward() always uses the log semiring.

        # Single.
        log_next_beta, [blank_marginal], [lexical_marginal] = alignment.backward(
            alpha=torch.log(alpha),
            blank=[torch.log(blank)],
            lexical=[torch.log(lexical)],
            beta=torch.log(beta),
            log_z=torch.log(z),
            context=context,
        )
        next_beta = torch.exp(log_next_beta)
        torch.testing.assert_close(
            next_beta,
            torch.tensor(
                [
                    blank[0] * beta[0]
                    + lexical[0, 0] * beta[1]
                    + lexical[0, 1] * beta[2],
                    blank[1] * beta[1]
                    + lexical[1, 0] * beta[1]
                    + lexical[1, 1] * beta[2],
                    blank[2] * beta[2]
                    + lexical[2, 0] * beta[1]
                    + lexical[2, 1] * beta[2],
                ]
            ),
            rtol=1e-4,
            atol=1e-7,
        )
        torch.testing.assert_close(
            blank_marginal, alpha * blank * beta / z, rtol=1e-4, atol=1e-7
        )
        torch.testing.assert_close(
            lexical_marginal,
            torch.tensor(
                [
                    [
                        alpha[0] * lexical[0, 0] * beta[1] / z,
                        alpha[0] * lexical[0, 1] * beta[2] / z,
                    ],
                    [
                        alpha[1] * lexical[1, 0] * beta[1] / z,
                        alpha[1] * lexical[1, 1] * beta[2] / z,
                    ],
                    [
                        alpha[2] * lexical[2, 0] * beta[1] / z,
                        alpha[2] * lexical[2, 1] * beta[2] / z,
                    ],
                ]
            ),
            rtol=1e-4,
            atol=1e-7,
        )

        # Batched.
        batched_log_next_beta, _, _ = alignment.backward(
            alpha=torch.log(alpha)[None],
            blank=[torch.log(blank)[None]],
            lexical=[torch.log(lexical)[None]],
            beta=torch.log(beta)[None],
            log_z=torch.log(z)[None],
            context=context,
        )
        torch.testing.assert_close(batched_log_next_beta, log_next_beta[None])

        # Wrong number of weights.
        with self.assertRaisesRegex(ValueError, "blank should be"):
            alignment.backward(
                alpha=alpha,
                blank=[blank, blank],
                lexical=[lexical],
                beta=beta,
                log_z=z,
                context=context,
            )
        with self.assertRaisesRegex(ValueError, "lexical should be"):
            alignment.backward(
                alpha=alpha,
                blank=[blank],
                lexical=[lexical, lexical],
                beta=beta,
                log_z=z,
                context=context,
            )

    def test_string_forward(self):
        alignment = alignments.FrameDependent()
        rngs = npr.default_rng(0).integers(low=0, high=1e90, size=(3, 2))
        alpha = torch.from_numpy(npr.default_rng(rngs[0]).uniform(high=1, size=4))
        blank = torch.from_numpy(npr.default_rng(rngs[1]).uniform(high=1, size=4))
        lexical = torch.from_numpy(npr.default_rng(rngs[2]).uniform(high=1, size=4))

        # Single.
        next_alpha = alignment.string_forward(
            alpha=alpha, blank=[blank], lexical=[lexical], semiring=semirings.Real
        )
        torch.testing.assert_close(
            next_alpha,
            torch.tensor(
                [
                    alpha[0] * blank[0],
                    alpha[1] * blank[1] + alpha[0] * lexical[0],
                    alpha[2] * blank[2] + alpha[1] * lexical[1],
                    alpha[3] * blank[3] + alpha[2] * lexical[2],
                ]
            ),
        )

        # Batched.
        batched_next_alpha = alignment.string_forward(
            alpha=alpha[None],
            blank=[blank[None]],
            lexical=[lexical[None]],
            semiring=semirings.Real,
        )
        torch.testing.assert_close(batched_next_alpha, next_alpha[None])

        # Wrong number of weights.
        with self.assertRaisesRegex(ValueError, "blank should be"):
            alignment.string_forward(
                alpha=alpha,
                blank=[blank, blank],
                lexical=[lexical],
                semiring=semirings.Real,
            )
        with self.assertRaisesRegex(ValueError, "lexical should be"):
            alignment.string_forward(
                alpha=alpha,
                blank=[blank],
                lexical=[lexical, lexical],
                semiring=semirings.Real,
            )


class FrameLabelDependentTest(absltest.TestCase):
    def test_topology(self):
        alignment = alignments.FrameLabelDependent(max_expansions=2)
        self.assertEqual(alignment.num_states(), 3)
        self.assertEqual(alignment.start(), 0)
        self.assertEqual(alignment.blank_next(0), 0)
        self.assertEqual(alignment.blank_next(1), 0)
        self.assertEqual(alignment.blank_next(2), 0)
        self.assertEqual(alignment.lexical_next(0), 1)
        self.assertEqual(alignment.lexical_next(1), 2)
        self.assertIsNone(alignment.lexical_next(2))
        self.assertListEqual(alignment.topological_visit(), [0, 1, 2])

    # All possible paths. Useful for creating unit tests.
    #
    # alpha[0] * blank[0][0] * beta[0]
    # alpha[0] * lexical[0][0, 0] * blank[1][1] * beta[1]
    # alpha[0] * lexical[0][0, 0] * lexical[1][1, 0] * blank[2][1] * beta[1]
    # alpha[0] * lexical[0][0, 0] * lexical[1][1, 1] * blank[2][2] * beta[2]
    # alpha[0] * lexical[0][0, 1] * blank[1][2] * beta[2]
    # alpha[0] * lexical[0][0, 1] * lexical[1][2, 0] * blank[2][1] * beta[1]
    # alpha[0] * lexical[0][0, 1] * lexical[1][2, 1] * blank[2][2] * beta[2]

    # alpha[1] * blank[0][1] * beta[1]
    # alpha[1] * lexical[0][1, 0] * blank[1][1] * beta[1]
    # alpha[1] * lexical[0][1, 0] * lexical[1][1, 0] * blank[2][1] * beta[1]
    # alpha[1] * lexical[0][1, 0] * lexical[1][1, 1] * blank[2][2] * beta[2]
    # alpha[1] * lexical[0][1, 1] * blank[1][2] * beta[2]
    # alpha[1] * lexical[0][1, 1] * lexical[1][2, 0] * blank[2][1] * beta[1]
    # alpha[1] * lexical[0][1, 1] * lexical[1][2, 1] * blank[2][2] * beta[2]

    # alpha[2] * blank[0][2] * beta[2]
    # alpha[2] * lexical[0][2, 0] * blank[1][1] * beta[1]
    # alpha[2] * lexical[0][2, 0] * lexical[1][1, 0] * blank[2][1] * beta[1]
    # alpha[2] * lexical[0][2, 0] * lexical[1][1, 1] * blank[2][2] * beta[2]
    # alpha[2] * lexical[0][2, 1] * blank[1][2] * beta[2]
    # alpha[2] * lexical[0][2, 1] * lexical[1][2, 0] * blank[2][1] * beta[1]
    # alpha[2] * lexical[0][2, 1] * lexical[1][2, 1] * blank[2][2] * beta[2]

    def test_forward(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        alignment = alignments.FrameLabelDependent(max_expansions=2)
        rngs = npr.default_rng(0).integers(low=0, high=1e9, size=4)
        alpha = torch.from_numpy(npr.default_rng(rngs[0]).uniform(high=1, size=3))
        blank = list(
            npr.default_rng(
                rngs[1],
            ).uniform(high=1, size=(3, 3))
        )
        lexical = list(npr.default_rng(rngs[2]).uniform(high=1, size=(3, 3, 2)))

        # Single.
        next_alpha = alignment.forward(
            alpha=alpha,
            blank=blank,
            lexical=lexical,
            context=context,
            semiring=semirings.Real,
        )
        torch.testing.assert_close(
            next_alpha,
            torch.tensor(
                [
                    alpha[0] * blank[0][0],
                    alpha[0] * lexical[0][0, 0] * blank[1][1]
                    + alpha[0] * lexical[0][0, 0] * lexical[1][1, 0] * blank[2][1]
                    + alpha[0] * lexical[0][0, 1] * lexical[1][2, 0] * blank[2][1]
                    + alpha[1] * blank[0][1]
                    + alpha[1] * lexical[0][1, 0] * blank[1][1]
                    + alpha[1] * lexical[0][1, 0] * lexical[1][1, 0] * blank[2][1]
                    + alpha[1] * lexical[0][1, 1] * lexical[1][2, 0] * blank[2][1]
                    + alpha[2] * lexical[0][2, 0] * blank[1][1]
                    + alpha[2] * lexical[0][2, 0] * lexical[1][1, 0] * blank[2][1]
                    + alpha[2] * lexical[0][2, 1] * lexical[1][2, 0] * blank[2][1],
                    alpha[0] * lexical[0][0, 0] * lexical[1][1, 1] * blank[2][2]
                    + alpha[0] * lexical[0][0, 1] * blank[1][2]
                    + alpha[0] * lexical[0][0, 1] * lexical[1][2, 1] * blank[2][2]
                    + alpha[1] * lexical[0][1, 0] * lexical[1][1, 1] * blank[2][2]
                    + alpha[1] * lexical[0][1, 1] * blank[1][2]
                    + alpha[1] * lexical[0][1, 1] * lexical[1][2, 1] * blank[2][2]
                    + alpha[2] * blank[0][2]
                    + alpha[2] * lexical[0][2, 0] * lexical[1][1, 1] * blank[2][2]
                    + alpha[2] * lexical[0][2, 1] * blank[1][2]
                    + alpha[2] * lexical[0][2, 1] * lexical[1][2, 1] * blank[2][2],
                ]
            ),
        )

        # Batched.
        batched_next_alpha = alignment.forward(
            alpha=alpha[None],
            blank=[i[None] for i in blank],
            lexical=[i[None] for i in lexical],
            context=context,
            semiring=semirings.Real,
        )
        torch.testing.assert_close(batched_next_alpha, next_alpha[None])

        # Wrong number of weights.
        with self.assertRaisesRegex(ValueError, "blank should be"):
            alignment.forward(
                alpha=alpha,
                blank=blank + blank,
                lexical=lexical,
                context=context,
                semiring=semirings.Real,
            )
        with self.assertRaisesRegex(ValueError, "lexical should be"):
            alignment.forward(
                alpha=alpha,
                blank=blank,
                lexical=lexical + lexical,
                context=context,
                semiring=semirings.Real,
            )

    def test_backward(self):
        context = contexts.FullNGram(vocab_size=2, context_size=1)
        alignment = alignments.FrameLabelDependent(max_expansions=2)
        rngs = npr.default_rng(0).integers(low=0, high=1e9, size=5)
        alpha = torch.from_numpy(npr.default_rng(rngs[0]).uniform(high=1, size=3))
        blank = list(torch.from_numpy(npr.default_rng(0).uniform(high=1, size=(3, 3))))
        lexical = list(
            torch.from_numpy(npr.default_rng(0).uniform(high=1, size=(3, 3, 2)))
        )
        beta = torch.from_numpy(npr.default_rng(rngs[3]).uniform(high=1, size=3))
        z = torch.from_numpy(npr.default_rng(rngs[4]).uniform(high=1, size=()))

        # backward() always uses the log semiring.

        # Single.
        log_next_beta, blank_marginals, lexical_marginals = alignment.backward(
            alpha=torch.log(alpha),
            blank=[torch.log(i) for i in blank],
            lexical=[torch.log(i) for i in lexical],
            beta=torch.log(beta),
            log_z=torch.log(z),
            context=context,
        )
        next_beta = torch.exp(log_next_beta)
        torch.testing.assert_close(
            next_beta,
            torch.tensor(
                [
                    blank[0][0] * beta[0]
                    + lexical[0][0, 0] * blank[1][1] * beta[1]
                    + lexical[0][0, 0] * lexical[1][1, 0] * blank[2][1] * beta[1]
                    + lexical[0][0, 0] * lexical[1][1, 1] * blank[2][2] * beta[2]
                    + lexical[0][0, 1] * blank[1][2] * beta[2]
                    + lexical[0][0, 1] * lexical[1][2, 0] * blank[2][1] * beta[1]
                    + lexical[0][0, 1] * lexical[1][2, 1] * blank[2][2] * beta[2],
                    blank[0][1] * beta[1]
                    + lexical[0][1, 0] * blank[1][1] * beta[1]
                    + lexical[0][1, 0] * lexical[1][1, 0] * blank[2][1] * beta[1]
                    + lexical[0][1, 0] * lexical[1][1, 1] * blank[2][2] * beta[2]
                    + lexical[0][1, 1] * blank[1][2] * beta[2]
                    + lexical[0][1, 1] * lexical[1][2, 0] * blank[2][1] * beta[1]
                    + lexical[0][1, 1] * lexical[1][2, 1] * blank[2][2] * beta[2],
                    blank[0][2] * beta[2]
                    + lexical[0][2, 0] * blank[1][1] * beta[1]
                    + lexical[0][2, 0] * lexical[1][1, 0] * blank[2][1] * beta[1]
                    + lexical[0][2, 0] * lexical[1][1, 1] * blank[2][2] * beta[2]
                    + lexical[0][2, 1] * blank[1][2] * beta[2]
                    + lexical[0][2, 1] * lexical[1][2, 0] * blank[2][1] * beta[1]
                    + lexical[0][2, 1] * lexical[1][2, 1] * blank[2][2] * beta[2],
                ]
            ),
            rtol=1e-4,
            atol=1e-6,
        )
        torch.testing.assert_close(
            torch.stack(blank_marginals),
            torch.tensor(
                [
                    [
                        alpha[0] * blank[0][0] * beta[0],
                        alpha[1] * blank[0][1] * beta[1],
                        alpha[2] * blank[0][2] * beta[2],
                    ],
                    [
                        0,
                        alpha[0] * lexical[0][0, 0] * blank[1][1] * beta[1]
                        + alpha[1] * lexical[0][1, 0] * blank[1][1] * beta[1]
                        + alpha[2] * lexical[0][2, 0] * blank[1][1] * beta[1],
                        alpha[0] * lexical[0][0, 1] * blank[1][2] * beta[2]
                        + alpha[1] * lexical[0][1, 1] * blank[1][2] * beta[2]
                        + alpha[2] * lexical[0][2, 1] * blank[1][2] * beta[2],
                    ],
                    [
                        0,
                        alpha[0]
                        * lexical[0][0, 0]
                        * lexical[1][1, 0]
                        * blank[2][1]
                        * beta[1]
                        + alpha[0]
                        * lexical[0][0, 1]
                        * lexical[1][2, 0]
                        * blank[2][1]
                        * beta[1]
                        + alpha[1]
                        * lexical[0][1, 0]
                        * lexical[1][1, 0]
                        * blank[2][1]
                        * beta[1]
                        + alpha[1]
                        * lexical[0][1, 1]
                        * lexical[1][2, 0]
                        * blank[2][1]
                        * beta[1]
                        + alpha[2]
                        * lexical[0][2, 0]
                        * lexical[1][1, 0]
                        * blank[2][1]
                        * beta[1]
                        + alpha[2]
                        * lexical[0][2, 1]
                        * lexical[1][2, 0]
                        * blank[2][1]
                        * beta[1],
                        alpha[0]
                        * lexical[0][0, 0]
                        * lexical[1][1, 1]
                        * blank[2][2]
                        * beta[2]
                        + alpha[0]
                        * lexical[0][0, 1]
                        * lexical[1][2, 1]
                        * blank[2][2]
                        * beta[2]
                        + alpha[1]
                        * lexical[0][1, 0]
                        * lexical[1][1, 1]
                        * blank[2][2]
                        * beta[2]
                        + alpha[1]
                        * lexical[0][1, 1]
                        * lexical[1][2, 1]
                        * blank[2][2]
                        * beta[2]
                        + alpha[2]
                        * lexical[0][2, 0]
                        * lexical[1][1, 1]
                        * blank[2][2]
                        * beta[2]
                        + alpha[2]
                        * lexical[0][2, 1]
                        * lexical[1][2, 1]
                        * blank[2][2]
                        * beta[2],
                    ],
                ]
            )
            / z,
            rtol=1e-4,
            atol=1e-6,
        )
        torch.testing.assert_close(
            torch.stack(lexical_marginals),
            torch.tensor(
                [
                    [
                        [
                            alpha[0] * lexical[0][0, 0] * blank[1][1] * beta[1]
                            + alpha[0]
                            * lexical[0][0, 0]
                            * lexical[1][1, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[0]
                            * lexical[0][0, 0]
                            * lexical[1][1, 1]
                            * blank[2][2]
                            * beta[2],
                            alpha[0] * lexical[0][0, 1] * blank[1][2] * beta[2]
                            + alpha[0]
                            * lexical[0][0, 1]
                            * lexical[1][2, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[0]
                            * lexical[0][0, 1]
                            * lexical[1][2, 1]
                            * blank[2][2]
                            * beta[2],
                        ],
                        [
                            alpha[1] * lexical[0][1, 0] * blank[1][1] * beta[1]
                            + alpha[1]
                            * lexical[0][1, 0]
                            * lexical[1][1, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[1]
                            * lexical[0][1, 0]
                            * lexical[1][1, 1]
                            * blank[2][2]
                            * beta[2],
                            alpha[1] * lexical[0][1, 1] * blank[1][2] * beta[2]
                            + alpha[1]
                            * lexical[0][1, 1]
                            * lexical[1][2, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[1]
                            * lexical[0][1, 1]
                            * lexical[1][2, 1]
                            * blank[2][2]
                            * beta[2],
                        ],
                        [
                            alpha[2] * lexical[0][2, 0] * blank[1][1] * beta[1]
                            + alpha[2]
                            * lexical[0][2, 0]
                            * lexical[1][1, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[2]
                            * lexical[0][2, 0]
                            * lexical[1][1, 1]
                            * blank[2][2]
                            * beta[2],
                            alpha[2] * lexical[0][2, 1] * blank[1][2] * beta[2]
                            + alpha[2]
                            * lexical[0][2, 1]
                            * lexical[1][2, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[2]
                            * lexical[0][2, 1]
                            * lexical[1][2, 1]
                            * blank[2][2]
                            * beta[2],
                        ],
                    ],
                    [
                        [0, 0],
                        [
                            alpha[0]
                            * lexical[0][0, 0]
                            * lexical[1][1, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[1]
                            * lexical[0][1, 0]
                            * lexical[1][1, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[2]
                            * lexical[0][2, 0]
                            * lexical[1][1, 0]
                            * blank[2][1]
                            * beta[1],
                            alpha[0]
                            * lexical[0][0, 0]
                            * lexical[1][1, 1]
                            * blank[2][2]
                            * beta[2]
                            + alpha[1]
                            * lexical[0][1, 0]
                            * lexical[1][1, 1]
                            * blank[2][2]
                            * beta[2]
                            + alpha[2]
                            * lexical[0][2, 0]
                            * lexical[1][1, 1]
                            * blank[2][2]
                            * beta[2],
                        ],
                        [
                            alpha[0]
                            * lexical[0][0, 1]
                            * lexical[1][2, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[1]
                            * lexical[0][1, 1]
                            * lexical[1][2, 0]
                            * blank[2][1]
                            * beta[1]
                            + alpha[2]
                            * lexical[0][2, 1]
                            * lexical[1][2, 0]
                            * blank[2][1]
                            * beta[1],
                            alpha[0]
                            * lexical[0][0, 1]
                            * lexical[1][2, 1]
                            * blank[2][2]
                            * beta[2]
                            + alpha[1]
                            * lexical[0][1, 1]
                            * lexical[1][2, 1]
                            * blank[2][2]
                            * beta[2]
                            + alpha[2]
                            * lexical[0][2, 1]
                            * lexical[1][2, 1]
                            * blank[2][2]
                            * beta[2],
                        ],
                    ],
                    [
                        [0, 0],
                        [0, 0],
                        [0, 0],
                    ],
                ]
            )
            / z,
            rtol=1e-4,
            atol=1e-6,
        )

        # Batched.
        batched_log_next_beta, _, _ = alignment.backward(
            alpha=torch.log(alpha)[None],
            blank=[torch.log(i)[None] for i in blank],
            lexical=[torch.log(i)[None] for i in lexical],
            beta=torch.log(beta)[None],
            log_z=torch.log(z)[None],
            context=context,
        )
        torch.testing.assert_close(batched_log_next_beta, log_next_beta[None])

        # Wrong number of weights.
        with self.assertRaisesRegex(ValueError, "blank should be"):
            alignment.backward(
                alpha=alpha,
                blank=blank + blank,
                lexical=lexical,
                beta=beta,
                log_z=z,
                context=context,
            )
        with self.assertRaisesRegex(ValueError, "lexical should be"):
            alignment.backward(
                alpha=alpha,
                blank=blank,
                lexical=lexical + lexical,
                beta=beta,
                log_z=z,
                context=context,
            )

    def test_string_forward(self):
        alignment = alignments.FrameLabelDependent(max_expansions=2)
        rngs = npr.default_rng(0).integers(0, 1e9, size=3)
        alpha = torch.from_numpy(npr.default_rng(rngs[0]).uniform(1, size=4))
        blank = list(npr.default_rng(rngs[1]).uniform(1, size=(3, 4)))
        lexical = list(npr.default_rng(rngs[2]).uniform(1, size=(3, 4)))

        # Single.
        next_alpha = alignment.string_forward(
            alpha=alpha, blank=blank, lexical=lexical, semiring=semirings.Real
        )
        torch.testing.assert_close(
            next_alpha,
            torch.tensor(
                [
                    alpha[0] * blank[0][0],
                    alpha[1] * blank[0][1] + alpha[0] * lexical[0][0] * blank[1][1],
                    alpha[2] * blank[0][2]
                    + alpha[1] * lexical[0][1] * blank[1][2]
                    + alpha[0] * lexical[0][0] * lexical[1][1] * blank[2][2],
                    alpha[3] * blank[0][3]
                    + alpha[2] * lexical[0][2] * blank[1][3]
                    + alpha[1] * lexical[0][1] * lexical[1][2] * blank[2][3],
                ]
            ),
        )

        # Batched.
        batched_next_alpha = alignment.string_forward(
            alpha=alpha[None],
            blank=[i[None] for i in blank],
            lexical=[i[None] for i in lexical],
            semiring=semirings.Real,
        )
        torch.testing.assert_close(batched_next_alpha, next_alpha[None])

        # Wrong number of weights.
        with self.assertRaisesRegex(ValueError, "blank should be"):
            alignment.string_forward(
                alpha=alpha,
                blank=blank + blank,
                lexical=lexical,
                semiring=semirings.Real,
            )
        with self.assertRaisesRegex(ValueError, "lexical should be"):
            alignment.string_forward(
                alpha=alpha,
                blank=blank,
                lexical=lexical + lexical,
                semiring=semirings.Real,
            )


if __name__ == "__main__":
    absltest.main()
