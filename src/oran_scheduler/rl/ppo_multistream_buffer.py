from dataclasses import dataclass

from oran_scheduler.rl.ppo_rollout import (
    PPORolloutTransition,
    validate_ppo_temporal_link,
)


@dataclass(frozen=True)
class PPOMultiStreamBufferConfig:
    """
    Centralized PPO collection across independent
    environment streams.

    A stream will later correspond to one gNB/cell.

    PAPER-SPECIFIED:
        PPO update/minibatch size M = 128.

    PAPER-SPECIFIED ARCHITECTURE:
        One shared model is trained centrally using
        experience gathered from multiple gNBs.

    IMPORTANT OPEN ISSUE:
        Under our current joint-layer interpretation,

            21 cells
            x 4 user slots
            =
            84 transitions / TTI.

        Therefore the centralized sample count does
        not naturally land on exactly 128.

        This class deliberately exposes that fact
        rather than silently discarding, carrying,
        or mixing samples.
    """

    num_streams: int

    update_size: int = 128

    def __post_init__(self) -> None:
        if self.num_streams <= 0:
            raise ValueError(
                "num_streams must be positive."
            )

        if self.update_size <= 0:
            raise ValueError(
                "update_size must be positive."
            )


@dataclass(frozen=True)
class PPOMultiStreamBufferStatus:
    """
    Snapshot of centralized collection state.
    """

    num_transitions: int

    update_size: int

    has_reached_update_size: bool

    num_transitions_beyond_update_size: int

    num_transitions_by_stream: tuple[
        int,
        ...
    ]


class PPOMultiStreamTransitionBuffer:
    """
    Store PPO transitions from several independent
    temporal streams.

    Temporal continuity is validated WITHIN each
    stream.

    No temporal relationship is required BETWEEN
    streams.

    Example:

        stream 0 = Cell 0
        stream 1 = Cell 1

        Cell 0:
            A0 -> A1 -> A2

        Cell 1:
            B0 -> B1 -> B2

    Valid:

        A0 -> A1
        B0 -> B1

    Not required:

        A1 == B0

    because those states belong to different cells.
    """

    def __init__(
        self,
        *,
        config: PPOMultiStreamBufferConfig,
    ) -> None:
        self.config = config

        self._transitions_by_stream: dict[
            int,
            list[
                PPORolloutTransition
            ],
        ] = {
            stream_id: []
            for stream_id
            in range(
                config.num_streams
            )
        }

        self._num_transitions = 0


    def __len__(
        self,
    ) -> int:
        return self._num_transitions


    @property
    def has_reached_update_size(
        self,
    ) -> bool:
        return (
            self._num_transitions
            >= self.config.update_size
        )


    @property
    def is_exactly_update_size(
        self,
    ) -> bool:
        return (
            self._num_transitions
            == self.config.update_size
        )


    @property
    def num_transitions_beyond_update_size(
        self,
    ) -> int:
        return max(
            0,
            (
                self._num_transitions
                - self.config.update_size
            ),
        )


    def num_transitions_for_stream(
        self,
        stream_id: int,
    ) -> int:
        self._validate_stream_id(
            stream_id
        )

        return len(
            self._transitions_by_stream[
                stream_id
            ]
        )


    def _validate_stream_id(
        self,
        stream_id: int,
    ) -> None:
        if not (
            0
            <= stream_id
            < self.config.num_streams
        ):
            raise ValueError(
                "stream_id is outside the "
                "configured stream range."
            )


    def add_transition(
        self,
        *,
        stream_id: int,
        transition: PPORolloutTransition,
    ) -> None:
        """
        Add one transition to one independent
        trajectory stream.

        Continuity is checked only against the most
        recent transition from THIS stream.
        """

        self._validate_stream_id(
            stream_id
        )

        stream = (
            self._transitions_by_stream[
                stream_id
            ]
        )

        if len(
            stream
        ) > 0:
            validate_ppo_temporal_link(
                stream[-1],
                transition,
            )

        stream.append(
            transition
        )

        self._num_transitions += 1


    def add_stream_transitions(
        self,
        *,
        stream_id: int,
        transitions: (
            list[
                PPORolloutTransition
            ]
            | tuple[
                PPORolloutTransition,
                ...
            ]
        ),
    ) -> None:
        """
        Atomically append several ordered
        transitions to one stream.

        If temporal validation fails, nothing is
        added.
        """

        self._validate_stream_id(
            stream_id
        )

        if len(
            transitions
        ) == 0:
            return

        candidate = list(
            transitions
        )

        existing = (
            self._transitions_by_stream[
                stream_id
            ]
        )

        if len(
            existing
        ) > 0:
            validate_ppo_temporal_link(
                existing[-1],
                candidate[0],
            )

        for index in range(
            len(candidate) - 1
        ):
            validate_ppo_temporal_link(
                candidate[index],
                candidate[index + 1],
            )

        existing.extend(
            candidate
        )

        self._num_transitions += len(
            candidate
        )


    def transitions_for_stream(
        self,
        stream_id: int,
    ) -> tuple[
        PPORolloutTransition,
        ...
    ]:
        """
        Return a snapshot of one trajectory stream.
        """

        self._validate_stream_id(
            stream_id
        )

        return tuple(
            self._transitions_by_stream[
                stream_id
            ]
        )


    def status(
        self,
    ) -> PPOMultiStreamBufferStatus:
        return PPOMultiStreamBufferStatus(
            num_transitions=(
                self._num_transitions
            ),
            update_size=(
                self.config.update_size
            ),
            has_reached_update_size=(
                self.has_reached_update_size
            ),
            num_transitions_beyond_update_size=(
                self
                .num_transitions_beyond_update_size
            ),
            num_transitions_by_stream=tuple(
                len(
                    self
                    ._transitions_by_stream[
                        stream_id
                    ]
                )
                for stream_id
                in range(
                    self.config.num_streams
                )
            ),
        )


    def clear(
        self,
    ) -> None:
        """
        Clear every trajectory stream.

        Later the centralized optimizer will own the
        exact conditions under which this is allowed.
        """

        for stream in (
            self
            ._transitions_by_stream
            .values()
        ):
            stream.clear()

        self._num_transitions = 0


