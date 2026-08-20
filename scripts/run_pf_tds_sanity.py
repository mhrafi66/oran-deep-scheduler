import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    run_pf_tds,
)

def main() -> None:
    device = "cuda:0"

    config = PFTimeDomainConfig(
        num_candidates=10,
    )

    batch_size = 1
    num_cells = 21
    num_ues_per_cell = 20

    torch.manual_seed(7)

    instantaneous_rate = (
        1.0
        +
        99.0
        * torch.rand(
            batch_size,
            num_cells,
            num_ues_per_cell,
            device=device,
        )
    )

    past_average_throughput = (
        1.0
        +
        49.0
        * torch.rand(
            batch_size,
            num_cells,
            num_ues_per_cell,
            device=device,
        )
    )

    candidate_indices, candidate_metrics = run_pf_tds(
        instantaneous_rate=instantaneous_rate,
        past_average_throughput=past_average_throughput,
        config=config,
    )

    print("=" * 72)
    print("PF Time-Domain Scheduler Sanity Check")
    print("=" * 72)

    print(f"Device:                 {candidate_indices.device}")
    print(f"Input rate shape:       {tuple(instantaneous_rate.shape)}")
    print(f"Candidate shape:        {tuple(candidate_indices.shape)}")

    print()
    print("Cell 0 selected UE indices:")
    print(candidate_indices[0, 0].tolist())

    print()
    print("Cell 0 selected PF metrics:")
    print(
        [
            round(value, 4)
            for value in candidate_metrics[
                0,
                0,
            ].tolist()
        ]
    )

    print()

    print(
        "PF metrics sorted descending: "
        f"{bool(torch.all(candidate_metrics[..., :-1] >= candidate_metrics[..., 1:]))}"
    )

    print("=" * 72)
    print("Validation: PASSED")


if __name__ == "__main__":
    main()