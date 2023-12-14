import abc
from typing import Callable, Generic, Optional, TypeVar

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


# Weight functions are the only components in GNAT with trainable parameters. We
# implement weight functions in two parts: WeightFn and WeightFnCacher.
#
# A WeightFn is a neural network that computes the arc weights for a given
# frame. Sometimes it requires static data that doesn't depend on the frames but
# is expensive to compute (e.g. the context embeddings of the shared-rnn weight
# function). We avoid unnecessarily recomputing such static data by off-loading
# the computation of static data to a separate WeightFnCacher (e.g.
# SharedRNNCacher).
#
# This way, whenever we know the static data doesn't change (e.g. when the
# underlying model parameters don't change such as during inference), we can
# reuse the result from WeightFnCacher as cache.

T = TypeVar("T")


class WeightFn(nn.Module, Generic[T]):
    """Interface for weight functions.

    A weight function is a neural network that computes the arc weights from all
    or some context states for a given frame. A WeightFn is used in pair with a
    WeightFnCacher that produces the static data cache, e.g. JointWeightFn can be
    used with SharedEmbCacher or SharedRNNCacher.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, cache: T, frame: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes arc weights for a given frame.

        Args:
          cache: Cached data from the corresponding WeightFnCacher.
          frame: [batch_dims..., feature_size] input frame.
          state: None or int32 array broadcastable to [batch_dims...]. If None,
            compute arc weights for all context states. Otherwise, compute arc
            weights for the specified context state.

        Returns:
          (blank, lexical) tuple.

          If state is None:
          - blank: [batch_dims..., num_context_states] weights for blank arcs.
            blank[..., p] is the weight of producing blank from context state p.
          - lexical: [batch_dims..., num_context_states, vocab_size] weights for
            lexical arcs. lexical[..., p, y] is the weight of producing label y from
            context state p.

          If state is not None:
          - blank: [batch_dims...] weights for blank arcs from the corresponding
            `state`.
          - lexical: [batch_dims..., vocab_size] weights for lexical arcs.
            lexical[..., y] is the weight of producing label y from the
            corresponding `state`.
        """
        raise NotImplementedError


class WeightFnCacher(nn.Module, Generic[T]):
    """Interface for weight function cachers.

    A weight function cacher prepares static data that may require expensive
    computational work. For example: the context state embeddings used by
    JointWeightFn can be from running an RNN on n-gram label sequences
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self) -> T:
        """Builds the cached data."""


def hat_normalize(
    blank: torch.Tensor, lexical: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local normalization used in the Hybrid Autoregressive Transducer (HAT) paper.

    The sigmoid of the blank weight is directly interpreted as the probability of
    blank. The lexical probability is then normalized with a log-softmax.

    Args:
      blank: [batch_dims...] blank weight.
      lexical: [batch_dims..., vocab_size] lexical weights.

    Returns:
      Normalized (blank, lexical) weights.
    """
    # Outside normalizer.
    z = torch.log(1 + torch.exp(blank))
    normalized_blank = blank - z
    normalized_lexical = F.log_softmax(lexical, dim=-1) - z[..., None]
    return normalized_blank, normalized_lexical


def log_softmax_normalize(
    blank: torch.Tensor, lexical: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard log-softmax local normalization.

    Weights are concatenated and then normalized together.

    Args:
      blank: [batch_dims...] blank weight.
      lexical: [batch_dims..., vocab_size] lexical weights.

    Returns:
      Normalized (blank, lexical) weights.
    """
    all_weights = torch.cat([blank[..., None], lexical], axis=-1)
    all_weights = F.log_softmax(all_weights)
    return all_weights[..., 0], all_weights[..., 1:]


class LocallyNormalizedWeightFn(WeightFn[T]):
    """Wrapper for turning any weight function into a locally normalized one.

    This is the recommended way of obtaining a locally normalized weight function.
    Algorithms such as those that computes the sequence log-loss may rely on a
    weight function being of this type to eliminate unnecessary denominator
    computation.

    It is thus also important for the normalize function to be mathematically
    correct: let (blank, lexical) be the pair of weights produced by the normalize
    function, then `torch.exp(blank) + torch.sum(torch.exp(lexical), dim=-1)` should be
    approximately equal to 1.

    Attributes:
      weight_fn: Underlying weight function.
      normalize: Callable that produces normalized log-probabilities from (blank,
        lexical) weights, e.g. hat_normalize() or log_softmax_normalize().
    """

    def __init__(self, weight_fn: WeightFn[T]) -> None:
        super().__init__()
        normalize: Callable[
            [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ] = hat_normalize
        self.normalize = normalize
        self.weight_fn = weight_fn

    def forward(
        self, cache: T, frame: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blank, lexical = self.weight_fn(cache, frame, state)
        # pylint: disable=too-many-function-args
        return self.normalize(blank, lexical)


class MyLazyLinear(nn.LazyLinear):
    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.ones_(self.bias)


class JointWeightFn(WeightFn[torch.Tensor]):
    r"""Common implementation of both the shared-emb and shared-rnn weight functions.

    To use shared-emb weight functions, pair this with a SharedEmbCacher. To use
    shared-rnn weight functions, pair this with a SharedRNNCacher. More generally,
    this weight function works with any WeightFnCacher that produces a
    [num_context_states, embedding_size] context embedding table.

    Attributes:
      vocab_size: Size of the lexical output vocabulary (not including the blank),
        i.e. $|\Sigma|$.
      hidden_size: Hidden layer size.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.project_context_embeddings = MyLazyLinear(hidden_size, bias=False)
        self.project_frame = MyLazyLinear(hidden_size, bias=False)
        self.joint_bias = nn.Parameter(torch.zeros(hidden_size))
        self.lin1 = nn.Linear(hidden_size, 1)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        with torch.no_grad():
            nn.init.ones_(self.lin1.weight)
            nn.init.zeros_(self.lin1.bias)
            nn.init.ones_(self.lin2.weight)
            nn.init.zeros_(self.lin2.bias)

    def forward(
        self,
        cache: torch.Tensor,
        frame: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_embeddings = cache
        if state is None:
            frame = frame[..., None, :]
        else:
            context_embeddings = context_embeddings[state.long()]
        projected_context_embeddings = self.project_context_embeddings(
            context_embeddings
        )
        projected_frame = self.project_frame(frame)
        joint = nn.functional.tanh(
            self.joint_bias + projected_context_embeddings + projected_frame
        )
        blank = torch.squeeze(self.lin1(joint), dim=-1)
        lexical = self.lin2(joint)
        return blank, lexical


class SharedEmbCacher(WeightFnCacher[torch.Tensor]):
    """A randomly initialized, independent context embedding table.

    The result context embedding table can be used with JointWeightFn.
    """

    def __init__(self, num_context_states: int, embedding_size: int) -> None:
        super().__init__()
        self.num_context_states = num_context_states
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(self.num_context_states, self.embedding_size)

    def forward(self) -> torch.Tensor:
        self.embed.weight = nn.Parameter(nn.init.uniform_(self.embed.weight, -1.0, 0.0))
        self.register_parameter("context_embeddings", self.embed.weight)
        return self.context_embeddings


class SharedRNNCacher(WeightFnCacher[torch.Tensor]):
    """Builds a context embedding table by running n-gram context labels through an RNN.

    This is usually used with last.contexts.FullNGram, where num_context_states =
    sum(vocab_size**i for i in range(context_size + 1)). The result context
    embedding table can be used with JointWeightFn.
    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        rnn_size: int,
        rnn_embedding_size: int,
        rnn_cell: Optional[nn.RNNCellBase] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.rnn_size = rnn_size
        self.rnn_embedding_size = rnn_embedding_size
        if rnn_cell is None:
            self.rnn_cell = nn.LSTMCell(rnn_embedding_size, rnn_size)
        else:
            self.rnn_cell = rnn_cell
        self.embed = nn.Embedding(self.vocab_size + 1, self.rnn_embedding_size)

    # running one step of rnn w init zeros tensor
    def forward(self) -> torch.Tensor:
        def tile_rnn_state(state):
            return einops.repeat(state, "n ... -> (n v) ...", v=self.vocab_size)

        embed = self.embed
        if isinstance(self.rnn_cell, nn.LSTMCell):
            rnn_states, start_embedding = self.rnn_cell(embed(torch.LongTensor([0])))
        else:
            rnn_states, start_embedding = self.rnn_cell(
                self.rnn_cell.initialize_carry(self.rnn_cell, (1, self.rnn_size)),
                embed(torch.LongTensor([0])),
            )
        parts = [start_embedding]
        for i in range(self.context_size):
            if i == 0:
                inputs = embed(torch.arange(1, self.vocab_size + 1))
            else:
                inputs = einops.repeat(inputs, "n ... -> (v n) ...", v=self.vocab_size)
            if isinstance(self.rnn_cell, nn.LSTMCell):
                rnn_states, embeddings = self.rnn_cell(inputs)
            else:
                rnn_states, embeddings = self.rnn_cell(
                    tile_rnn_state(rnn_states), inputs
                )
            parts.append(embeddings)
        return torch.cat(parts, dim=0)


class NullCacher(WeightFnCacher[type(None)]):
    """A cacher that simply returns None.

    Mainly used with TableWeightFn for unit testing.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self) -> None:
        return None


class TableWeightFn(WeightFn[type(None)]):
    """Weight function that looks up a fixed table, useful for testing.

    Attributes:
      table: [batch_dims..., input_vocab_size, num_context_states, 1 + vocab_size]
        arc weight table. For each input frame, we simply cast the 0-th element
        into an integer "input label" and look up the corresponding weights. The
        weights of blank arcs are stored at `table[..., 0]`, and the weights of
        lexical arcs at `table[..., 1:]`.
    """

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = table
        self.params = None

    def forward(
        self, cache: None, frame: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache

        *batch_dims, input_vocab_size, num_context_states, _ = self.table.shape
        if frame.shape[:-1] != torch.Size(batch_dims):
            raise ValueError(
                f"frame should have batch_dims={tuple(batch_dims)} but "
                f"got {frame.shape[:-1]}"
            )

        frame_mask = F.one_hot(frame[..., 0].to(torch.int64), input_vocab_size)
        weights = torch.einsum("...xcy,...x->...cy", self.table, frame_mask)

        if state is not None:
            state = torch.broadcast_to(state, batch_dims)
            state_mask = F.one_hot(state, num_context_states)
            weights = torch.einsum("...cy,...c->...y", weights, state_mask)

        blank = weights[..., 0]
        lexical = weights[..., 1:]
        return blank, lexical
