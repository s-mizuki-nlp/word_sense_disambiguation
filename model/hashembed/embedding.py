from __future__ import unicode_literals, division

from typing import List, Callable, Optional

import numpy as np

import torch
import torch.nn as nn


class HashFunctionConstructor():

    def __init__(self, a:int, b:int, bins:int, moduler: int, mask_zero: bool):
        self.a = a
        self.b = b
        self.bins = bins
        self.moduler = moduler
        self.mask_zero = mask_zero

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_zero:
            # The return doesn't set 0 to 0 because that's taken into account in the hash embedding
            # if want to use for an integer then should uncomment second line !!!!!!!!!!!!!!!!!!!!!
            return ((self.a * x + self.b) % self.moduler) % (self.bins - 1) + 1
            # return lambda x: 0 if x == 0 else ((a*x + b) % self.moduler) % (self.bins-1) + 1
        else:
            return ((self.a * x + self.b) % self.moduler) % self.bins


class SequenceHashFunctionConstructor():

    def __init__(self, a:int, b:int, max_seq_len:int, bins:int, moduler: int, mask_zero: bool):
        self.a = a
        self.b = b
        self.max_seq_len = max_seq_len
        self.bins = bins
        self.moduler = moduler
        self.mask_zero = mask_zero
        self.hash_functions = [HashFunctionConstructor(a=a*(n+1)+1, b=b*(n+1)+1, bins=bins, moduler=moduler, mask_zero=True) for n in range(max_seq_len)]

    def __call__(self, x_2d: torch.Tensor):
        """
        it emits hash value for input sequence.
        specifically, let r as returned variable and r[b,n] = hash(x_2d[b,:n]).

        @param x_2d: two-dimensional LongTensor. (n_batch, n_seq_len)
        @output: two-dimensional LongTensor. (n_batch, n_seq_len)
        """

        n_seq_len = x_2d.shape[-1]
        assert n_seq_len <= len(self.hash_functions), f"too long sequence: {n_seq_len} > {self.max_seq_len}"

        t_seq = torch.stack([hash_function(x_n) for hash_function, x_n in zip(self.hash_functions[:n_seq_len], x_2d.swapaxes(0,1))], dim=-1)
        if self.mask_zero:
            t_hash_values = t_seq.cumprod(dim=-1) % (self.bins -1) + 1
        else:
            t_hash_values = t_seq.cumprod(dim=-1) % self.bins

        return t_hash_values


class HashFamily():
    r"""Universal hash family as proposed by Carter and Wegman.

    .. math::

            \begin{array}{ll}
            h_{{a,b}}(x)=((ax+b)~{\bmod  ~}p)~{\bmod  ~}m \ \mid p > m\\
            \end{array}

    Args:
        bins (int): Number of bins to hash to. Better if a prime number.
        mask_zero (bool, optional): Whether the 0 input is a special "padding" value to mask out.
        moduler (int,optional): Temporary hashing. Has to be a prime number.
    """

    def __init__(self, bins, mask_zero=False, moduler=None, is_sequence_input: bool = False, max_seq_len: Optional[int] = None, random_seed: int = 42):
        if moduler and moduler <= bins:
            raise ValueError("p (moduler) should be >> m (buckets)")

        if is_sequence_input:
            assert isinstance(max_seq_len, int), f"you must specify `max_seq_len` argument."

        self.random_seed = random_seed

        self.bins = bins
        np.random.seed(random_seed)
        self.moduler = moduler if moduler else self._next_prime(np.random.randint(self.bins + 1, int(1E15)))
        self.mask_zero = mask_zero
        self.is_sequence_input = is_sequence_input
        self.max_seq_len = max_seq_len

        # do not allow same a and b, as it could mean shifted hashes
        self.sampled_a = set()
        self.sampled_b = set()

    def _is_prime(self, x):
        """Naive is prime test."""
        for i in range(2, int(np.sqrt(x))):
            if x % i == 0:
                return False
        return True

    def _next_prime(self, n):
        """Naively gets the next prime larger than n."""
        while not self._is_prime(n):
            n += 1

        return n

    def draw_hash(self, a=None, b=None):
        """Draws a single hash function from the family."""
        if a is None:
            np.random.seed(self.random_seed)
            while a is None or a in self.sampled_a:
                a = np.random.randint(1, self.moduler - 1)
                assert len(self.sampled_a) < self.moduler - 2, "please give a bigger moduler"

            self.sampled_a.add(a)
        if b is None:
            np.random.seed(self.random_seed)
            while b is None or b in self.sampled_b:
                b = np.random.randint(0, self.moduler - 1)
                assert len(self.sampled_b) < self.moduler - 1, "please give a bigger moduler"

            self.sampled_b.add(b)

        if self.is_sequence_input:
            return SequenceHashFunctionConstructor(a=a, b=b, bins=self.bins, max_seq_len=self.max_seq_len,
                                                   moduler=self.moduler, mask_zero=self.mask_zero)
        else:
            return HashFunctionConstructor(a=a, b=b, bins=self.bins, moduler=self.moduler, mask_zero=self.mask_zero)

    def draw_hashes(self, n, **kwargs) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        """Draws n hash function from the family."""
        return [self.draw_hash() for i in range(n)]


class HashEmbedding(nn.Module):
    r"""Type of embedding which uses multiple hashes to approximate an Embedding layer using less parameters.

    This module is a new Embedding module that compresses the number of parameters. They are a
    generalization of vanilla Embeddings and the `hashing trick`. For more details, check Svenstrup,
    Dan Tito, Jonas Hansen, and Ole Winther. "Hash embeddings for efficient word
    representations." Advances in Neural Information Processing Systems. 2017.

    For each elements (usually word indices) in the input (mini_batch, sequence_length) the default
    computations are:

    .. math::

            \begin{array}{ll}
            H_i = E_{D_2^i(D_1(w)))} \ \forall i=1...k\\
            c_w = (H_1(w), ..., H_k(w))^T\\
            p_w = P_{D_1(w)}\\
            \hat{e}_w = p_w \cdot c_w\\
            e_w = \mathrm{concatenate}(\hat{e}_w,p_w)\\
            \end{array}

    where :math:`w:[0,T]` is the element of the input (word index), :math:`D_1:[0,T)\to [0,K)`
    is the token to ID hash/dictionnary, :math:`D_2:[0,K)\to[0,B)` is the ID to Bucket hash,
    :math:`E:\mathbb R^{B*d}` is the shared pool of embeddings, :math:`c_w:\mathbb R^{k*d}` contains all
    the vector embeddings to which :math:`w` maps, :math:`e_w:\mathbb R^{d+k}` is the outputed word
    embedding for :math:`w`.

    Args:
        num_embeddings (int): the number of different embeddings. K in the paper.
            Higher increases possible vocabulary size.
        embedding_dim (int): the size of each embedding vector in the shared pool. d in the paper.
            Higher improves downstream task for fixed vocabulary size.
        num_buckets (int,optional): the size of the shared pool of embeddings. B in the paper.
            Higher improves approximation quality. Typically num_buckets * 10 < num_embeddings.
        num_hashes (int,optional): the number of different hash functions. k in the paper.
            Higher improves approximation quality. Typically in [1,3].
        train_sharedEmbed (bool,optional): whether to train the shared pool of embeddings E.
        train_weight (bool,optional): whether to train the importance parameters / weight P.
        append_weight (bool,optional): whether to append the importance parameters / weight pw.
        aggregation_mode ({"sum","median","concatenate"},optional): how to aggregate the (weighted) component
            vectors of the different hashes. Sum should be the same as mean (because learnable parameters,
            can learn to divide by n)
        mask_zero (bool, optional): whether the 0 input is a special "padding" value to mask out.
        seed (int, optional): sets the seed for generating random numbers.
        oldAlgorithm (bool, optional): whether to use the algorithm in the paper rather than the improved version.
            I do not recommend to set to true besides for comparaison.

    Attributes:
        shared_embeddings (nn.Embedding): the shared pool of embeddings of shape (num_buckets, embedding_dim).
            E in the paper.
        importance_weights (nn.Embedding): the importance parameters / weight of shape
            (num_embeddings, num_hashes). P in the paper.
        output_dim (int): effective outputed number of embeddings.

    Shape:
        - Input: LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        - Output: `(N, W, output_dim)`, output_dim is the effective embedding dim.

    Examples::
        >>> # an HashEmbedding module containing approximating nn.Embedding(10, 5) with less param
        >>> embedding = HashEmbedding(10,5,append_weight=False)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
        1.00000e-04 *
           0.3988  0.5234 -0.6148  0.3000 -1.5525
           0.1259  0.4142 -0.8613  0.3018 -1.3547
           0.1367  0.2638 -0.2993  0.9541 -1.7194
          -0.4672 -0.7971 -0.2009  0.7829 -0.9448

        (1 ,.,.) =
        1.00000e-04 *
           0.1367  0.2638 -0.2993  0.9541 -1.7194
          -0.0878 -0.1680  0.3896  0.5288 -0.2060
           0.1259  0.4142 -0.8613  0.3018 -1.3547
          -0.3098  0.0357 -0.7532 -0.1216 -0.0366
        [torch.FloatTensor of size 2x4x5]

        >>> # example with mask_zero which corresponds to padding_idx=0
        >>> embedding = HashEmbedding(10,5,append_weight=False, mask_zero=True)
        >>> input = Variable(torch.LongTensor([[0,2,0,5]]))
        >>> embedding(input)

        Variable containing:
        (0 ,.,.) =
        1.00000e-04 *
           0.0000  0.0000  0.0000  0.0000  0.0000
          -1.4941 -1.3775 -0.5797  1.3187 -0.0555
           0.0000  0.0000  0.0000  0.0000  0.0000
          -0.7717 -0.5569 -0.1397  1.1101 -0.0939
        [torch.FloatTensor of size 1x4x5]
    """

    def __init__(self, num_embeddings, embedding_dim, num_buckets=None, num_hashes=2, train_sharedEmbed=True,
                 train_weight=True, append_weight=True, aggregation_mode='sum', padding_idx=None, seed=None, oldAlgorithm=False):
        super(HashEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_hashes = num_hashes
        defaultNBuckets = (num_embeddings * self.num_hashes) // (self.embedding_dim)
        self.num_buckets = num_buckets - 1 if num_buckets is not None else defaultNBuckets
        self.train_sharedEmbed = train_sharedEmbed
        self.train_weight = train_weight
        self.append_weight = append_weight
        self.padding_idx = padding_idx
        self.seed = seed
        # THERE IS NO ADVANTAGE OF SETTING THE FOLLOWING TO TRUE, I JUST HAVE TO COMPARE WITH THE ALGORITHM IN THE PAPER
        self.oldAlgorithm = oldAlgorithm

        self.importance_weights = nn.Embedding(self.num_embeddings,
                                               self.num_hashes)
        self.shared_embeddings = nn.Embedding(self.num_buckets + 1,
                                              self.embedding_dim,
                                              padding_idx=self.padding_idx)

        mask_zero = True if (padding_idx is not None) and (padding_idx == 0) else False
        hashFamily = HashFamily(self.num_buckets, mask_zero=mask_zero)
        self.hashes = hashFamily.draw_hashes(self.num_hashes)

        if aggregation_mode == 'sum':
            self.aggregate = self._aggregate_sum
        elif aggregation_mode == 'concatenate':
            # little bit quicker than permute/contiguous/view
            self.aggregate = self._aggregate_concat
        elif aggregation_mode == 'median':
            self.aggregate = self._aggregate_median
        else:
            raise ValueError('unknown aggregation function {}'.format(aggregation_mode))
        self.aggregation_mode = aggregation_mode

        self.output_dim = self.embedding_dim
        if aggregation_mode == "concatenate":
            self.output_dim *= self.num_hashes
        if self.append_weight:
            self.output_dim += self.num_hashes

        self.reset_parameters()

    def _aggregate_sum(self, tensor):
        return torch.sum(tensor, dim=-1)

    def _aggregate_concat(self, tensor):
        return torch.cat([tensor[:, :, :, i] for i in range(self.num_hashes)], dim=-1)

    def _aggregate_median(self, tensor):
        return torch.median(tensor, dim=-1)[0]

    def reset_parameters(self, std: float = 0.1):
        """Resets the trainable parameters."""
        def set_constant_row(parameters, iRow=0, value=0):
            """Return `parameters` with row `iRow` as s constant `value`."""
            data = parameters.data
            data[iRow, :] = value
            return torch.nn.Parameter(data, requires_grad=parameters.requires_grad)

        np.random.seed(self.seed)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.shared_embeddings.weight = torch.nn.init.normal_(self.shared_embeddings.weight, mean=0.0, std=std)
        self.importance_weights.weight = torch.nn.init.normal_(self.importance_weights.weight, mean=0.0, std=0.0005)

        if self.padding_idx is not None:
            # Unfortunately has to set weight to 0 even when paddingIdx = 0
            self.shared_embeddings.weight = set_constant_row(self.shared_embeddings.weight)
            self.importance_weights.weight = set_constant_row(self.importance_weights.weight)

        self.shared_embeddings.weight.requires_grad = self.train_sharedEmbed
        self.importance_weights.weight.requires_grad = self.train_weight

    def forward(self, input):
        idx_importance_weights = input % self.num_embeddings
        # THERE IS NO ADVANTAGE OF USING THE FOLLWOING LINE, I JUST HAVE TO COMPARE WITH THE ALGORITHM IN THE PAPER
        input = idx_importance_weights if self.oldAlgorithm else input
        idx_shared_embeddings = torch.stack([h(input).masked_fill_(input == 0, 0) for h in self.hashes], dim=-1)

        shared_embedding = torch.stack([self.shared_embeddings(idx_shared_embeddings[:, :, iHash])
                                        for iHash in range(self.num_hashes)], dim=-1)
        importance_weight = self.importance_weights(idx_importance_weights)
        importance_weight = importance_weight.unsqueeze(-2)
        word_embedding = self.aggregate(importance_weight * shared_embedding)
        if self.append_weight:
            # concateates the vector with the weights
            word_embedding = torch.cat([word_embedding, importance_weight.squeeze(-2)], dim=-1)
        return word_embedding

    def summary(self):
        ret = {}
        for attr_name in ("num_embeddings", "num_hashes", "num_buckets", "append_weight", "padding_idx"):
            ret[attr_name] = getattr(self, attr_name)
        return ret