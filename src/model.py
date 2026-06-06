"""
model.py

Small GPT-style decoder-only Transformer for GameTune.

The model learns next-token prediction over tokenized NES-style music.
It receives a sequence of token IDs and predicts the next token at each
position.

Architecture:
    token IDs
        -> token embeddings
        -> positional embeddings
        -> causal Transformer blocks
        -> final layer norm
        -> vocabulary logits

Design decisions:
    - Decoder-only architecture, like GPT.
    - Causal attention prevents tokens from seeing future tokens.
    - Fixed context size of 2048 tokens, based on token length analysis.
    - Small default model size to fit comfortably on a single RTX 3090.
    - Vocabulary size is 898, matching the current GameTune tokenizer.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GameTuneModelConfig:
    """
    Model hyperparameters.

    These values define the size and capacity of the Transformer.
    """

    vocab_size: int = 898
    context_size: int = 2048
    embedding_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Self-attention lets every token look at previous tokens in the sequence.
    Causal masking prevents the model from looking at future tokens.
    """

    def __init__(self, config: GameTuneModelConfig) -> None:
        super().__init__()

        if config.embedding_size % config.num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads")

        # PyTorch implementation of multi-head attention.
        # batch_first=True means tensors use shape:
        #     batch, sequence, embedding
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal self-attention.

        Input shape:
            batch, sequence, embedding

        Output shape:
            batch, sequence, embedding
        """
        sequence_length = x.size(1)

        # Upper-triangular mask blocks attention to future positions.
        #
        # Example for sequence length 4:
        #
        # token 0 can see: token 0
        # token 1 can see: token 0, token 1
        # token 2 can see: token 0, token 1, token 2
        # token 3 can see: token 0, token 1, token 2, token 3
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask,
            need_weights=False,
        )

        return output


class FeedForward(nn.Module):
    """
    Feed-forward network used inside each Transformer block.

    After attention mixes information across time, this network transforms
    each token position independently.
    """

    def __init__(self, config: GameTuneModelConfig) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # Expand the embedding dimension.
            nn.Linear(config.embedding_size, 4 * config.embedding_size),

            # Nonlinear activation.
            nn.GELU(),

            # Project back to the original embedding dimension.
            nn.Linear(4 * config.embedding_size, config.embedding_size),

            # Dropout helps reduce overfitting.
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward network.
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Transformer decoder block.

    Each block has:
        1. LayerNorm
        2. Causal self-attention
        3. Residual connection
        4. LayerNorm
        5. Feed-forward network
        6. Residual connection
    """

    def __init__(self, config: GameTuneModelConfig) -> None:
        super().__init__()

        # Normalization before attention improves training stability.
        self.attention_norm = nn.LayerNorm(config.embedding_size)
        self.attention = CausalSelfAttention(config)

        # Normalization before the feed-forward network.
        self.ffn_norm = nn.LayerNorm(config.embedding_size)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one Transformer block.
        """
        # Residual connection:
        # The block learns a correction added to the existing representation.
        x = x + self.attention(self.attention_norm(x))

        # Second residual connection after the feed-forward network.
        x = x + self.feed_forward(self.ffn_norm(x))

        return x


class GameTuneModel(nn.Module):
    """
    Decoder-only Transformer for next-token prediction.

    Given input tokens:
        [BOS, CHANNEL_P1, NOTE_64, ...]

    The model predicts:
        [CHANNEL_P1, NOTE_64, VELOCITY_2, ...]
    """

    def __init__(self, config: GameTuneModelConfig) -> None:
        super().__init__()

        self.config = config

        # Converts token IDs into dense vectors.
        #
        # Example:
        #     token id 42 -> vector of size embedding_size
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
        )

        # Adds position information.
        #
        # Without this, the model would know which tokens appear,
        # but not where they appear in the sequence.
        self.position_embedding = nn.Embedding(
            config.context_size,
            config.embedding_size,
        )

        # Stack of Transformer blocks.
        #
        # More layers increase model capacity.
        self.blocks = nn.ModuleList(
            TransformerBlock(config)
            for _ in range(config.num_layers)
        )

        # Final normalization before projecting to vocabulary logits.
        self.final_norm = nn.LayerNorm(config.embedding_size)

        # Converts final hidden vectors into scores for every vocabulary token.
        #
        # Output shape:
        #     batch, sequence, vocab_size
        self.output_head = nn.Linear(
            config.embedding_size,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Run the model.

        input_ids:
            Tensor of token IDs with shape:
                batch, sequence

        targets:
            Optional tensor of expected next tokens with shape:
                batch, sequence

        returns:
            logits:
                Prediction scores with shape:
                    batch, sequence, vocab_size

            loss:
                Cross-entropy loss if targets are provided, otherwise None.
        """
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.context_size:
            raise ValueError("sequence length exceeds model context size")

        # Position IDs:
        #     [0, 1, 2, ..., sequence_length - 1]
        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        # Combine token identity and token position.
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        # Pass the sequence through all Transformer blocks.
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Produce one vocabulary-sized prediction per token position.
        logits = self.output_head(x)

        loss = None

        if targets is not None:
            # Cross-entropy expects:
            #     predictions: batch*sequence, vocab_size
            #     targets:     batch*sequence
            loss = F.cross_entropy(
                logits.view(batch_size * sequence_length, -1),
                targets.view(batch_size * sequence_length),
            )

        return logits, loss

    def parameter_count(self) -> int:
        """
        Return the number of trainable parameters.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


