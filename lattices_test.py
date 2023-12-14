import functools

from absl.testing import absltest
import numpy.random as npr
import torch
import torch.nn.functional as F
import numpy as np

import alignments
import contexts
import lattices
import semirings
import weight_fns

# TODO: PRNGKey,vjp


def weight_fn_cacher_factory(context: contexts.FullNGram):
    return weight_fns.SharedRNNCacher(
        vocab_size=context.vocab_size,
        context_size=context.context_size,
        rnn_size=24,
        rnn_embedding_size=24,
    )


def weight_fn_factory(context: contexts.ContextDependency):
    _, vocab_size = context.shape()
    return weight_fns.JointWeightFn(vocab_size=vocab_size, hidden_size=16)


class RecognitionLatticeBasicsTest(absltest.TestCase):
    def test_build_cache(self):
        vocab_size = 2
        context_size = 1
        lattice = lattices.RecognitionLattice(
            context=contexts.FullNGram(
                vocab_size=vocab_size, context_size=context_size
            ),
            alignment=alignments.FrameDependent(),
            weight_fn_cacher_factory=weight_fn_cacher_factory,
            weight_fn_factory=weight_fn_factory,
        )
        frames = torch.from_numpy(
            npr.default_rng(0).normal(0, 0.01, size=(4, 6, 8)).astype(np.float32)
        )
        num_frames = torch.tensor([6, 3, 2, 0])
        labels = torch.tensor(
            [[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2], [2, 1, 2, 1]],
            dtype=torch.float32,
        )
        num_labels = torch.tensor([4, 3, 1, 0])
        params = None  # TODO
        loss = lattice(
            frames=frames,
            num_frames=num_frames,
            labels=labels,
            num_labels=num_labels,
        )
        with self.subTest("same cache"):
            cache = lattice.apply(lattice.build_cache())
            loss_with_same_cache = lattice(
                frames=frames,
                num_frames=num_frames,
                labels=labels,
                num_labels=num_labels,
                cache=cache,
            )  # params
            torch.testing.assert_close(loss_with_same_cache, loss)
        with self.subTest("different cache"):
            # This makes sure we are using the cache when supplied.
            loss_with_different_cache = lattice(
                frames=frames,
                num_frames=num_frames,
                labels=labels,
                num_labels=num_labels,
                cache=map(lambda x: x + 1, cache),
            )  # params
            self.assertTrue(
                (loss_with_different_cache != loss).any(),
                msg=f"Should be not equal: loss={loss!r}, "
                f"loss_with_different_cache={loss_with_different_cache!r}",
            )

    def test_call(self):
        vocab_size = 2
        context_size = 1
        lattice = lattices.RecognitionLattice(
            context=contexts.FullNGram(
                vocab_size=vocab_size, context_size=context_size
            ),
            alignment=alignments.FrameDependent(),
            weight_fn_cacher_factory=weight_fn_cacher_factory,
            weight_fn_factory=weight_fn_factory,
        )
        frames = torch.from_numpy(
            npr.default_rng(0).normal(0, 0.01, size=(4, 6, 8)).astype(np.float32)
        )
        num_frames = torch.tensor([6, 3, 2, 1])
        labels = torch.tensor(
            [[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2], [2, 1, 2, 1]],
            dtype=torch.float32,
        )
        num_labels = torch.tensor([4, 3, 1, 2])
        loss = lattice(
            frames=frames,
            num_frames=num_frames,
            labels=labels,
            num_labels=num_labels,
        )
        params = None  # loss weight?
        torch.testing.assert_close(torch.isfinite(loss), [True, True, True, False])

        with self.subTest("padded inputs"):
            loss_with_padded_inputs = lattice(
                frames=F.pad(frames, [(0, 0), (0, 1), (0, 0)]),
                num_frames=num_frames,
                labels=F.pad(labels, [(0, 0), (0, 2)]),
                num_labels=num_labels,
            )
            torch.testing.assert_close(loss_with_padded_inputs, loss)

        with self.subTest("invalid shapes"):
            with self.assertRaisesRegex(
                ValueError, "frames and num_frames have different batch_dims"
            ):
                lattice(
                    frames=frames[:1],
                    num_frames=num_frames,
                    labels=labels,
                    num_labels=num_labels,
                )
            with self.assertRaisesRegex(
                ValueError, "labels and num_frames have different batch_dims"
            ):
                lattice(
                    frames=frames,
                    num_frames=num_frames,
                    labels=labels[:1],
                    num_labels=num_labels,
                )
            with self.assertRaisesRegex(
                ValueError, "num_labels and num_frames have different batch_dims"
            ):
                lattice(
                    frames=frames,
                    num_frames=num_frames,
                    labels=labels,
                    num_labels=num_labels[:1],
                )


#     def test_shortest_path(self):
#         vocab_size = 2
#         context_size = 1
#         lattice = lattices.RecognitionLattice(
#             context=contexts.FullNGram(
#                 vocab_size=vocab_size, context_size=context_size
#             ),
#             alignment=alignments.FrameDependent(),
#             weight_fn_cacher_factory=weight_fn_cacher_factory,
#             weight_fn_factory=weight_fn_factory,
#         )
#         frames = torch.from_numpy(npr.default_rng(0).normal(0, 0.01, size=(4, 6, 8)))
#         # frames = jax.random.normal(jax.random.PRNGKey(0), [4, 6, 8])
#         num_frames = torch.tensor([6, 3, 2, 0])
#         (
#             alignment_labels,
#             num_alignment_labels,
#             path_weights,
#         ), params = lattice.apply(frames, num_frames, method=lattice.shortest_path)
#         with self.subTest("reasonable outputs"):
#             torch.testing.assert_close(num_alignment_labels, [6, 3, 2, 0])
#             is_padding = torch.arange(6) >= num_frames[:, None]
#             torch.testing.assert_close(
#                 torch.where(is_padding, alignment_labels, -1),
#                 [
#                     [-1, -1, -1, -1, -1, -1],
#                     [-1, -1, -1, 0, 0, 0],
#                     [-1, -1, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0],
#                 ],
#             )
#             torch.testing.assert_close(
#                 alignment_labels >= 0,
#                 torch.ones([4, 6], dtype=bool),
#                 err_msg=f"alignment_labels={alignment_labels!r}",
#             )
#             torch.testing.assert_close(
#                 alignment_labels <= vocab_size,
#                 torch.ones([4, 6], dtype=bool),
#                 err_msg=f"alignment_labels={alignment_labels!r}",
#             )
#             torch.testing.assert_close(
#                 torch.isfinite(path_weights),
#                 [True, True, True, True],
#                 err_msg=f"path_weights={path_weights!r}",
#             )
#             torch.testing.assert_close(
#                 path_weights == 0,
#                 [False, False, False, True],
#                 err_msg=f"path_weights={path_weights!r}",
#             )

#         with self.subTest("padded inputs"):
#             (_, _, path_weights_with_padded_inputs) = lattice(
#                 params,
#                 F.pad(frames, [(0, 0), (0, 1), (0, 0)]),
#                 num_frames,
#                 method=lattice.shortest_path,
#             )
#             torch.testing.assert_close(path_weights_with_padded_inputs, path_weights)

#         with self.subTest("invalid shapes"):
#             with self.assertRaisesRegex(
#                 ValueError, "frames and num_frames have different batch_dims"
#             ):
#                 lattice(
#                     params, frames[:1], num_frames, method=lattice.shortest_path
#                 )

#     def test_frame_label_dependent(self):
#         vocab_size = 2
#         context_size = 1
#         lattice = lattices.RecognitionLattice(
#             context=contexts.FullNGram(
#                 vocab_size=vocab_size, context_size=context_size
#             ),
#             alignment=alignments.FrameLabelDependent(max_expansions=2),
#             weight_fn_cacher_factory=weight_fn_cacher_factory,
#             weight_fn_factory=weight_fn_factory,
#         )
#         frames = torch.from_numpy(npr.default_rng(0).normal(0, 0.1, size=(4, 6, 8)))
#         # frames = jax.random.normal(jax.random.PRNGKey(0), [4, 6, 8])
#         num_frames = torch.tensor([6, 3, 2, 1])
#         labels = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 1, 2], [2, 1, 2, 1]])
#         num_labels = torch.tensor([4, 3, 4, 3])
#         with self.subTest("loss"):
#             loss, params = lattice(
#                 frames=frames,
#                 num_frames=num_frames,
#                 labels=labels,
#                 num_labels=num_labels,
#             )
#             torch.testing.assert_close(torch.isfinite(loss), [True, True, True, False])
#         with self.subTest("shortest_path"):
#             alignment_labels, num_alignment_labels, path_weights = lattice(
#                 params, frames, num_frames, method=lattice.shortest_path
#             )
#             torch.testing.assert_close(num_alignment_labels, 3 * num_frames)
#             is_padding = torch.arange(18) >= num_alignment_labels[:, None]
#             torch.testing.assert_close(
#                 torch.where(is_padding, alignment_labels, -1),
#                 [
#                     [-1] * 18,
#                     [-1] * 9 + [0] * 9,
#                     [-1] * 6 + [0] * 12,
#                     [-1] * 3 + [0] * 15,
#                 ],
#             )
#             # Every third label is 0.
#             torch.testing.assert_close(
#                 alignment_labels.reshape([4, 6, 3])[..., -1], torch.zeros([4, 6])
#             )
#             torch.testing.assert_close(
#                 alignment_labels >= 0,
#                 torch.ones([4, 18], dtype=bool),
#                 err_msg=f"alignment_labels={alignment_labels!r}",
#             )
#             torch.testing.assert_close(
#                 alignment_labels <= vocab_size,
#                 torch.ones([4, 18], dtype=bool),
#                 err_msg=f"alignment_labels={alignment_labels!r}",
#             )
#             torch.testing.assert_close(
#                 torch.isfinite(path_weights),
#                 [True, True, True, True],
#                 err_msg=f"path_weights={path_weights!r}",
#             )


# class RecognitionLatticeCorrectnessTest(absltest.TestCase):
#     """Tests the correctness of various RecognitionLattice operations."""

#     def test_frame_dependent(self):
#         batch_size = 3
#         max_num_frames = 2
#         vocab_size = 2
#         context_size = 1
#         num_context_states = 3

#         frames = torch.broadcast_to(
#             torch.unsqueeze(
#                 torch.arange(max_num_frames, dtype=torch.float32), dim=0
#             ).unsqueeze(dim=2),
#             [batch_size, max_num_frames, 1],
#         )
#         num_frames = torch.tensor([2, 1, 0])

#         weight_table = 1 + torch.arange(
#             batch_size * max_num_frames * num_context_states * (1 + vocab_size),
#             dtype=torch.float32,
#         ).reshape([batch_size, max_num_frames, num_context_states, 1 + vocab_size])
#         # Alternate the signs over the frame time dimension so that we get some
#         # interesting shortest paths.
#         weight_table *= torch.unsqueeze(
#             torch.tensor([[-1, 1], [1, -1], [1, 1]]), dim=2
#         ).unsqueeze(dim=3)

#         lattice = lattices.RecognitionLattice(
#             context=contexts.FullNGram(
#                 vocab_size=vocab_size, context_size=context_size
#             ),
#             alignment=alignments.FrameDependent(),
#             weight_fn_factory=lambda _: weight_fns.TableWeightFn(weight_table),
#             weight_fn_cacher_factory=lambda _: weight_fns.NullCacher(),
#         )
#         # For easier application of methods.
#         # lattice = lattice.bind({})

#         # Forward, i.e. shortest distance.
#         for semiring_name, expected in [
#             ("MaxTropical", [-3 + 18, 21, 0]),
#             (
#                 "Real",
#                 [
#                     (-1) * (10 + 11 + 12)
#                     + (-2) * (13 + 14 + 15)
#                     + (-3) * (16 + 17 + 18),
#                     19 + 20 + 21,
#                     1,
#                 ],
#             ),
#             (
#                 "Log",
#                 [
#                     torch.logsumexp(
#                         torch.tensor(
#                             [
#                                 -1 + 10,
#                                 -1 + 11,
#                                 -1 + 12,
#                                 -2 + 13,
#                                 -2 + 14,
#                                 -2 + 15,
#                                 -3 + 16,
#                                 -3 + 17,
#                                 -3 + 18,
#                             ]
#                         ),
#                         dim=0,
#                     ),
#                     torch.logsumexp(torch.tensor([19, 20, 21]), dim=0),
#                     0.0,
#                 ],
#             ),
#         ]:
#             semiring = getattr(semirings, semiring_name)
#             with self.subTest(f"forward/{semiring_name}"):
#                 torch.testing.assert_close(
#                     lattice._forward(
#                         cache=None,
#                         frames=frames,
#                         num_frames=num_frames,
#                         semiring=semiring,
#                     )[0],
#                     expected,
#                 )

#         with self.subTest("shortest_path"):
#             (
#                 alignment_labels,
#                 num_alignment_labels,
#                 path_weights,
#             ) = lattice.shortest_path(frames=frames, num_frames=num_frames, cache=None)
#             torch.testing.assert_close(num_alignment_labels, num_frames)
#             torch.testing.assert_close(path_weights, [-3 + 18, 21, 0])
#             torch.testing.assert_close(
#                 alignment_labels,
#                 [
#                     [2, 2],
#                     [2, 0],
#                     [0, 0],
#                 ],
#             )

#         # String forward, i.e. shortest distance after intersection with a string.
#         labels = torch.tensor([[1, 2, 0], [2, 1, 0], [1, 2, 0]])
#         num_labels = torch.tensor([1, 1, 0])
#         for semiring_name, expected in [
#             ("MaxTropical", [-2 + 13, 21, 0]),
#             ("Real", [(-1) * 11 + (-2) * 13, 21, 1]),
#             (
#                 "Log",
#                 [torch.logsumexp(torch.tensor([-1 + 11, -2 + 13]), dim=0), 21.0, 0.0],
#             ),
#         ]:
#             semiring = getattr(semirings, semiring_name)
#             with self.subTest(f"string_forward/{semiring_name}"):
#                 torch.testing.assert_close(
#                     lattice._string_forward(
#                         cache=None,
#                         frames=frames,
#                         num_frames=num_frames,
#                         labels=labels,
#                         num_labels=num_labels,
#                         semiring=semiring,
#                     ),
#                     expected,
#                 )
#             with self.subTest(f"string_forward non-reachable/{semiring_name}"):
#                 torch.testing.assert_close(
#                     lattice._string_forward(
#                         cache=None,
#                         frames=frames,
#                         num_frames=num_frames,
#                         labels=labels,
#                         num_labels=torch.tensor([3, 2, 1]),
#                         semiring=semiring,
#                     ),
#                     semiring.zeros([3]),
#                 )

#         with self.subTest("call"):
#             log_loss = lattice(
#                 frames=frames,
#                 num_frames=num_frames,
#                 labels=labels,
#                 num_labels=num_labels,
#                 cache=None,
#             )
#             torch.testing.assert_close(
#                 log_loss,
#                 [
#                     torch.logsumexp(
#                         torch.tensor(
#                             [
#                                 -1 + 10,
#                                 -1 + 11,
#                                 -1 + 12,
#                                 -2 + 13,
#                                 -2 + 14,
#                                 -2 + 15,
#                                 -3 + 16,
#                                 -3 + 17,
#                                 -3 + 18,
#                             ]
#                         )
#                     )
#                     - torch.logsumexp(torch.tensor([-1 + 11, -2 + 13])),
#                     torch.logsumexp(torch.tensor([19, 20, 21])) - 21.0,
#                     0.0,
#                 ],
#                 rtol=1e-6,
#             )

#     # Tests for _backward().

#     def test_arc_marginals(self):
#         # Test _backward() by computing arc marginals. This is a bit easier to debug
#         # than the full-on forward-backward.
#         vocab_size = 2
#         context_size = 1
#         lattice = lattices.RecognitionLattice(
#             context=contexts.FullNGram(
#                 vocab_size=vocab_size, context_size=context_size
#             ),
#             alignment=alignments.FrameDependent(),
#             weight_fn_cacher_factory=weight_fn_cacher_factory,
#             weight_fn_factory=weight_fn_factory,
#         )
#         frames = jax.random.uniform(jax.random.PRNGKey(0), [4, 6, 8])
#         num_frames = torch.tensor([6, 3, 2, 0])
#         params = lattice(frames, num_frames, method=lattice.shortest_path)
#         # For easier application of methods.
#         lattice = lattice.bind(params)
#         del params
#         cache = lattice.build_cache()

#         # Compute expected marginals using autodiff.
#         def forward(masks):
#             blank_mask, lexical_mask = masks
#             log_z, _ = lattice._forward(
#                 cache=cache,
#                 frames=frames,
#                 num_frames=num_frames,
#                 semiring=semirings.Log,
#                 blank_mask=[blank_mask],
#                 lexical_mask=[lexical_mask],
#             )
#             return torch.sum(log_z)

#         num_context_states, _ = lattice.context.shape()
#         blank_mask = torch.zeros([*frames.shape[:-1], num_context_states])
#         lexical_mask = torch.zeros([*frames.shape[:-1], num_context_states, vocab_size])
#         expected_marginals = torch.autograd.grad(forward((blank_mask, lexical_mask)))

#         # Compute marginals using _backward().
#         def arc_marginals(frames, num_frames):
#             def arc_marginals_callback(
#                 weight_vjp_fn, carry, blank_marginal, lexical_marginals
#             ):
#                 del weight_vjp_fn
#                 del carry
#                 next_carry = None
#                 outputs = (blank_marginal, lexical_marginals)
#                 return next_carry, outputs

#             (
#                 log_z,
#                 alpha_0_to_T_minus_1,
#             ) = lattice._forward(  # pylint: disable=invalid-name
#                 cache=cache,
#                 frames=frames,
#                 num_frames=num_frames,
#                 semiring=semirings.Log,
#             )
#             _, (blank_marginal, lexical_marginals) = lattice._backward(
#                 cache=cache,
#                 frames=frames,
#                 num_frames=num_frames,
#                 log_z=log_z,
#                 alpha_0_to_T_minus_1=alpha_0_to_T_minus_1,
#                 init_callback_carry=None,
#                 callback=arc_marginals_callback,
#             )
#             return blank_marginal, lexical_marginals

#         actual_marginals = arc_marginals(frames, num_frames)
#         map(
#             functools.partial(torch.testing.assert_close, rtol=1e-3),
#             actual_marginals,
#             expected_marginals,
#         )

#     def test_forward_backward(self):
#         vocab_size = 2
#         context_size = 1
#         lattice = lattices.RecognitionLattice(
#             context=contexts.FullNGram(
#                 vocab_size=vocab_size, context_size=context_size
#             ),
#             alignment=alignments.FrameDependent(),
#             weight_fn_cacher_factory=weight_fn_cacher_factory,
#             weight_fn_factory=weight_fn_factory,
#         )
#         frames = jax.random.uniform(jax.random.PRNGKey(0), [4, 6, 8])
#         num_frames = torch.tensor([6, 3, 2, 0])
#         params = lattice(frames, num_frames, method=lattice.shortest_path)

#         def forward(params, frames):
#             cache = lattice(params, method=lattice.build_cache)
#             log_z, _ = lattice(
#                 params,
#                 cache=cache,
#                 frames=frames,
#                 num_frames=num_frames,
#                 semiring=semirings.Log,
#                 method=lattice._forward,
#             )
#             return log_z

#         expected_log_z, expected_vjp_fn = jax.vjp(forward, params, frames)

#         def forward_backward(params, frames):
#             cache = lattice(params, method=lattice.build_cache)
#             return lattice(
#                 params,
#                 cache=cache,
#                 frames=frames,
#                 num_frames=num_frames,
#                 method=lattice._forward_backward,
#             )

#         actual_log_z, actual_vjp_fn = jax.vjp(forward_backward, params, frames)
#         torch.testing.assert_close(actual_log_z, expected_log_z)

#         for g in [
#             torch.ones_like(expected_log_z),
#             jax.random.uniform(jax.random.PRNGKey(0), expected_log_z.shape),
#         ]:
#             expected_grads = expected_vjp_fn(g)
#             actual_grads = actual_vjp_fn(g)
#             map(
#                 functools.partial(torch.testing.assert_close, rtol=1e-3, atol=1e-6),
#                 actual_grads,
#                 expected_grads,
#             )


if __name__ == "__main__":
    absltest.main()
