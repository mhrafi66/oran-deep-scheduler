from dataclasses import dataclass

from typing import Iterator

import torch


@dataclass(frozen=True)
class PPOCandidatePermutation:
    """
    Candidate-index permutation used for 1LDS
    training-data augmentation.

    new_to_old:
        Shape [candidate].

        new_to_old[j] gives the OLD candidate whose
        feature segment is placed in NEW candidate
        slot j.

    old_to_new:
        Shape [candidate].

        Inverse mapping.

        old_to_new[i] gives the NEW candidate index
        of OLD candidate i.

    Example:

        old candidate order:
            [A, B, C]

        new_to_old:
            [2, 0, 1]

        new candidate order:
            [C, A, B]

        therefore:

            old A index 0 -> new index 1
            old B index 1 -> new index 2
            old C index 2 -> new index 0

        and:

            old_to_new = [1, 2, 0]
    """

    new_to_old: torch.Tensor

    old_to_new: torch.Tensor


@dataclass(frozen=True)
class PPOCandidateAugmentationConfig:
    """
    Candidate-permutation data augmentation.

    PAPER-SPECIFIED:
        Each stored training experience undergoes
        N_Pi candidate permutations.

    PAPER-UNSPECIFIED:
        - numerical value of N_Pi
        - whether the unpermuted sample is retained
          in addition to the N_Pi shuffled samples

    Therefore both choices are explicit.

    num_permutations:
        Number of RANDOMLY PERMUTED copies produced
        from one original sample.

        This corresponds to N_Pi.

    include_original:
        If True:
            retain the original sample in addition
            to the permuted copies.

        If False:
            store only the N_Pi permuted copies.

    Example:

        num_permutations = 3
        include_original = True

        one physical experience produces:

            original
            permutation 1
            permutation 2
            permutation 3

        total = 4 stored samples.
    """

    num_permutations: int

    include_original: bool

    def __post_init__(self) -> None:
        if self.num_permutations < 0:
            raise ValueError(
                "num_permutations cannot be "
                "negative."
            )

        if (
            self.num_permutations == 0
            and not self.include_original
        ):
            raise ValueError(
                "Augmentation would produce zero "
                "training samples."
            )

def build_candidate_permutation(
    new_to_old: torch.Tensor,
) -> PPOCandidatePermutation:
    """
    Validate a candidate permutation and construct
    its inverse.
    """

    if new_to_old.ndim != 1:
        raise ValueError(
            "Candidate permutation must have shape "
            "[candidate]."
        )

    if new_to_old.numel() < 1:
        raise ValueError(
            "Candidate permutation cannot be empty."
        )

    if (
        torch.is_floating_point(
            new_to_old
        )
        or new_to_old.dtype
        == torch.bool
    ):
        raise ValueError(
            "Candidate permutation must use an "
            "integer dtype."
        )

    num_candidates = int(
        new_to_old.numel()
    )

    if torch.any(
        new_to_old < 0
    ):
        raise ValueError(
            "Candidate permutation cannot contain "
            "negative indices."
        )

    if torch.any(
        new_to_old >= num_candidates
    ):
        raise ValueError(
            "Candidate permutation contains an "
            "out-of-range index."
        )

    sorted_values = torch.sort(
        new_to_old
    ).values

    expected = torch.arange(
        num_candidates,
        dtype=new_to_old.dtype,
        device=new_to_old.device,
    )

    if not torch.equal(
        sorted_values,
        expected,
    ):
        raise ValueError(
            "Candidate permutation must contain "
            "every candidate index exactly once."
        )

    old_to_new = torch.empty_like(
        new_to_old
    )

    new_indices = torch.arange(
        num_candidates,
        dtype=new_to_old.dtype,
        device=new_to_old.device,
    )

    old_to_new[
        new_to_old
    ] = new_indices

    return PPOCandidatePermutation(
        new_to_old=(
            new_to_old
            .detach()
            .clone()
        ),
        old_to_new=(
            old_to_new
            .detach()
            .clone()
        ),
    )

def sample_candidate_permutation(
    *,
    num_candidates: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> PPOCandidatePermutation:
    """
    Sample one uniformly random candidate
    permutation.

    The permutation tensor is returned on `device`.

    If an explicit torch.Generator is supplied,
    random sampling is performed on that generator's
    device and then transferred if necessary.
    """

    if num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive."
        )

    target_device = torch.device(
        device
    )

    if generator is None:
        permutation = torch.randperm(
            num_candidates,
            dtype=torch.long,
            device=target_device,
        )

    else:
        generator_device = torch.device(
            generator.device
        )

        permutation = torch.randperm(
            num_candidates,
            dtype=torch.long,
            device=generator_device,
            generator=generator,
        ).to(
            target_device
        )

    return build_candidate_permutation(
        permutation
    )


def permute_1lds_state(
    *,
    state: torch.Tensor,
    permutation: PPOCandidatePermutation,
) -> torch.Tensor:
    """
    Shuffle complete candidate-UE feature segments
    in a flattened 1LDS state.

    Input:
        state:
            [state_size]

    For the paper configuration:

        K = 10 candidates
        RBG = 18

        features per candidate:
            5 + 2 * 18 = 41

        state size:
            10 * 41 = 410

    The entire 41-element UE feature segment is
    moved together.

    Features INSIDE one UE segment are never
    rearranged.
    """

    if state.ndim != 1:
        raise ValueError(
            "1LDS state must have shape "
            "[state_size]."
        )

    num_candidates = int(
        permutation
        .new_to_old
        .numel()
    )

    state_size = int(
        state.shape[0]
    )

    if (
        state_size
        % num_candidates
        != 0
    ):
        raise ValueError(
            "State size is not divisible by the "
            "number of candidates."
        )

    if (
        state.device
        != permutation.new_to_old.device
    ):
        raise ValueError(
            "State and candidate permutation must "
            "be on the same device."
        )

    feature_segment_size = (
        state_size
        // num_candidates
    )

    candidate_segments = state.reshape(
        num_candidates,
        feature_segment_size,
    )

    permuted_segments = (
        candidate_segments
        .index_select(
            dim=0,
            index=(
                permutation
                .new_to_old
            ),
        )
    )

    return (
        permuted_segments
        .reshape(
            state_size
        )
        .detach()
        .clone()
    )

def remap_candidate_actions(
    *,
    actions: torch.Tensor,
    permutation: PPOCandidatePermutation,
) -> torch.Tensor:
    """
    Convert candidate-index actions from the old
    candidate ordering to the permuted ordering.

    Action convention:

        0 ... K-1
            candidate action

        K
            NO ALLOCATION

    NO ALLOCATION never changes.

    Example:

        old candidate order:
            [A, B, C]

        new order:
            [C, A, B]

        old action 0 = A

        A is now candidate index 1

        therefore:
            0 -> 1
    """

    if (
        torch.is_floating_point(
            actions
        )
        or actions.dtype
        == torch.bool
    ):
        raise ValueError(
            "Actions must use an integer dtype."
        )

    if (
        actions.device
        != permutation.old_to_new.device
    ):
        raise ValueError(
            "Actions and candidate permutation must "
            "be on the same device."
        )

    num_candidates = int(
        permutation
        .old_to_new
        .numel()
    )

    if torch.any(
        actions < 0
    ):
        raise ValueError(
            "Actions cannot be negative."
        )

    if torch.any(
        actions > num_candidates
    ):
        raise ValueError(
            "Action exceeds candidate/no-allocation "
            "action space."
        )

    remapped = (
        actions
        .detach()
        .clone()
    )

    candidate_action_mask = (
        actions < num_candidates
    )

    old_candidate_actions = (
        actions[
            candidate_action_mask
        ]
    )

    remapped[
        candidate_action_mask
    ] = (
        permutation
        .old_to_new[
            old_candidate_actions
        ]
    )

    return remapped


def permute_candidate_action_mask(
    *,
    action_mask: torch.Tensor,
    permutation: PPOCandidatePermutation,
) -> torch.Tensor:
    """
    Permute candidate-action columns while keeping
    the final NO-ALLOCATION column fixed.

    Input may have arbitrary leading dimensions:

        [..., K + 1]

    Typical 1LDS decision mask:

        [RBG, K + 1]

    Paper-shaped:

        [18, 11]
    """

    if action_mask.ndim < 1:
        raise ValueError(
            "action_mask must have at least one "
            "dimension."
        )

    if action_mask.dtype != torch.bool:
        raise ValueError(
            "action_mask must use torch.bool."
        )

    if (
        action_mask.device
        != permutation.new_to_old.device
    ):
        raise ValueError(
            "Action mask and candidate permutation "
            "must be on the same device."
        )

    num_candidates = int(
        permutation
        .new_to_old
        .numel()
    )

    expected_num_actions = (
        num_candidates
        + 1
    )

    if int(
        action_mask.shape[-1]
    ) != expected_num_actions:
        raise ValueError(
            "Action mask final dimension must be "
            "num_candidates + 1."
        )

    candidate_columns = (
        action_mask[
            ...,
            :num_candidates,
        ]
    )

    no_allocation_column = (
        action_mask[
            ...,
            num_candidates:
            num_candidates + 1,
        ]
    )

    permuted_candidates = (
        candidate_columns
        .index_select(
            dim=-1,
            index=(
                permutation
                .new_to_old
            ),
        )
    )

    return torch.cat(
        (
            permuted_candidates,
            no_allocation_column,
        ),
        dim=-1,
    ).detach().clone()


@dataclass(frozen=True)
class PPOPermutedDecisionData:
    """
    One consistently permuted PPO training decision.

    next_state is optional because expert-buffer
    demonstrations do not contain a next state.
    """

    state: torch.Tensor

    actions: torch.Tensor

    action_mask: torch.Tensor

    next_state: torch.Tensor | None

    expert_actions: torch.Tensor | None


@dataclass(frozen=True)
class PPOPermutedExpertData:
    """
    One consistently permuted PF-expert
    demonstration.

    state:
        [state_size]

    expert_actions:
        [RBG]

    action_mask:
        [RBG, K + 1]
    """

    state: torch.Tensor

    expert_actions: torch.Tensor

    action_mask: torch.Tensor

def permute_ppo_expert_data(
    *,
    state: torch.Tensor,
    expert_actions: torch.Tensor,
    action_mask: torch.Tensor,
    permutation: PPOCandidatePermutation,
) -> PPOPermutedExpertData:
    """
    Apply one candidate permutation consistently to
    a PF-expert demonstration.

    The physical UE represented by the expert label
    does not change.

    Only its candidate index changes.
    """

    return PPOPermutedExpertData(
        state=permute_1lds_state(
            state=state,
            permutation=permutation,
        ),
        expert_actions=(
            remap_candidate_actions(
                actions=expert_actions,
                permutation=permutation,
            )
        ),
        action_mask=(
            permute_candidate_action_mask(
                action_mask=action_mask,
                permutation=permutation,
            )
        ),
    )

def iter_candidate_permutations(
    *,
    config: PPOCandidateAugmentationConfig,
    num_candidates: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> Iterator[
    PPOCandidatePermutation | None
]:
    """
    Yield all candidate-order representations that
    should be stored for one physical experience.

    `None` represents the original, unpermuted
    representation.

    Every non-None value represents one randomly
    sampled permutation.

    The paper does not publish N_Pi, so the caller
    must supply it explicitly through `config`.
    """

    if num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive."
        )

    if config.include_original:
        yield None

    for _ in range(
        config.num_permutations
    ):
        yield sample_candidate_permutation(
            num_candidates=num_candidates,
            device=device,
            generator=generator,
        )





def permute_ppo_decision_data(
    *,
    state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    permutation: PPOCandidatePermutation,
    next_state: torch.Tensor | None = None,
    expert_actions: torch.Tensor | None = None,
) -> PPOPermutedDecisionData:
    """
    Apply one candidate permutation consistently to
    all candidate-indexed components.

    IMPORTANT:
        Reward is intentionally absent.

        The paper states that augmentation preserves
        the reward, so callers simply retain the
        original reward unchanged.
    """

    permuted_state = permute_1lds_state(
        state=state,
        permutation=permutation,
    )

    permuted_actions = (
        remap_candidate_actions(
            actions=actions,
            permutation=permutation,
        )
    )

    permuted_action_mask = (
        permute_candidate_action_mask(
            action_mask=action_mask,
            permutation=permutation,
        )
    )

    permuted_next_state = None

    if next_state is not None:
        permuted_next_state = (
            permute_1lds_state(
                state=next_state,
                permutation=permutation,
            )
        )

    permuted_expert_actions = None

    if expert_actions is not None:
        permuted_expert_actions = (
            remap_candidate_actions(
                actions=expert_actions,
                permutation=permutation,
            )
        )

    return PPOPermutedDecisionData(
        state=permuted_state,
        actions=permuted_actions,
        action_mask=(
            permuted_action_mask
        ),
        next_state=(
            permuted_next_state
        ),
        expert_actions=(
            permuted_expert_actions
        ),
    )


