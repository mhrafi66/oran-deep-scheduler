"""Task 5: reuse final MU-MIMO post-precoder SINR for whole-grant Sionna LA.

Attach `capture.capture_rbg` to the optional callback on
`evaluate_cell_allocation(...)` ONCE, for the *final* allocation.
Then use `capture.whole_links(allocation=..., config=...)` to obtain TB-wide
MCS/TBLER per candidate; it groups already computed per-RBG post-SINR into
one frequency-selective TB SINR tensor per UE (Sionna PHYAbstraction + ILLA).

This is Sionna SYSTEM-LEVEL link abstraction, not CRC-level decoding,
measured OTA BLER, or HARQ. It assumes the same post-SINR across configured
slot symbols when only one OFDM symbol is generated. The current legacy
schedule_evaluator evaluates equal-width RBGs; partial-RBG channel support
must be added upstream before claiming a full 273-PRB carrier in that path.

Avoids re-generating the channel, precoders, interference or SINR; no GPU
scalar `.item()` per RBG. At most one bulk metadata-to-CPU copy per RBG,
only during the optional final allocation pass. Groups same-shaped grants
into one Sionna inference batch and reuses cached adaptation objects.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from oran_scheduler.phy.link_adaptation import LinkAdaptationConfig
from oran_scheduler.schedulers.allocation import CellAllocation
from oran_scheduler.simulator.service_aware_nr_grants import WholeGrantLink


class FinalAllocationSINRCapture:
    def __init__(self) -> None:
        # candidate -> rbg -> [symbols, subcarriers, rank] GPU tensor
        self._samples: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
        self._rank: dict[int, int] = {}

    def clear(self) -> None:
        self._samples.clear()
        self._rank.clear()

    def capture_rbg(
        self,
        *,
        rbg_index: int,
        selected_candidate_indices: torch.Tensor,
        selected_ranks: torch.Tensor,
        layer_sinr_linear: torch.Tensor,
        layer_ue_indices: torch.Tensor,
        layer_index_within_ue: torch.Tensor,
    ) -> None:
        """Passive evaluator callback, called only for scheduled RBGs."""
        if rbg_index < 0 or rbg_index in {r for rbgs in self._samples.values() for r in rbgs}:
            raise ValueError("RBG captured twice or invalid RBG index")
        if layer_sinr_linear.ndim != 3:
            raise ValueError("Expected [symbol, subcarrier, physical_layer] SINR")
        if any(t.device != layer_sinr_linear.device for t in
               (selected_candidate_indices, selected_ranks, layer_ue_indices, layer_index_within_ue)):
            raise ValueError("All tensors must reside on the same PHY device")
        # One device-to-host synchronization per RBG for all small metadata
        # vectors, not separate .item() or .tolist() calls on CUDA tensors.
        vectors = (selected_candidate_indices, selected_ranks,
                   layer_ue_indices, layer_index_within_ue)
        lengths = [int(x.numel()) for x in vectors]
        metadata = torch.cat([x.detach().reshape(-1).to(dtype=torch.long)
                              for x in vectors]).to(device='cpu').tolist()
        offset = 0
        unpacked = []
        for size in lengths:
            unpacked.append(metadata[offset:offset+size])
            offset += size
        candidates, ranks, user_indices, within = unpacked
        if len(candidates) != len(ranks) or sum(ranks) != len(user_indices) or len(user_indices) != len(within):
            raise ValueError("Inconsistent MIMO UE / layer metadata")
        for pos, (candidate, rank) in enumerate(zip(candidates, ranks)):
            if rank not in (1,2) or candidate < 0:
                raise ValueError("Invalid candidate/rank")
            if candidate in self._rank and self._rank[candidate] != rank:
                raise ValueError("Same candidate has inconsistent rank across RBGs")
            slots = [None] * rank
            for layer, (ue_pos, layer_number) in enumerate(zip(user_indices, within)):
                if ue_pos == pos:
                    if layer_number < 0 or layer_number >= rank or slots[layer_number] is not None:
                        raise ValueError("Duplicate or illegal physical stream")
                    slots[layer_number] = layer
            if any(slot is None for slot in slots):
                raise ValueError("Missing scheduled UE physical layer")
            if rbg_index in self._samples[candidate]:
                raise ValueError("Candidate/RBG captured twice")
            self._rank[candidate] = rank
            indices = torch.tensor(slots, dtype=torch.long, device=layer_sinr_linear.device)
            self._samples[candidate][rbg_index] = layer_sinr_linear.index_select(-1, indices)

    def whole_links(
        self,
        *,
        allocation: CellAllocation,
        config: LinkAdaptationConfig,
    ) -> dict[int, WholeGrantLink]:
        """Reuse Sionna ILLA/PHYAbstraction for aggregate subcarrier SINR.

        The input must be captured from THIS final allocation; stale samples,
        missing candidates or RBGs, padding errors and rank changes fail.
        """
        if not isinstance(allocation, CellAllocation):
            raise TypeError("Expected final CellAllocation")
        from oran_scheduler.phy.link_adaptation import (
            LinkAdaptationConfig, create_link_adaptation_blocks,
        )
        if not isinstance(config, LinkAdaptationConfig):
            raise TypeError("Expected LinkAdaptationConfig")
        matrix = allocation.candidate_by_user_slot
        if matrix.ndim != 2:
            raise ValueError("Allocation must have [user_slot, RBG] shape")
        actions = matrix.detach().to(device='cpu', dtype=torch.long).tolist()
        expected: dict[int, set[int]] = defaultdict(set)
        for rbg in range(matrix.shape[1]):
            for slot in range(matrix.shape[0]):
                candidate = actions[slot][rbg]
                if candidate >= 0:
                    if rbg in expected[candidate]:
                        raise ValueError("Candidate scheduled twice on RBG")
                    expected[candidate].add(rbg)
                elif candidate != -1:
                    raise ValueError("Invalid candidate in allocation")
        if set(expected) != set(self._samples):
            raise ValueError("Missing/stale candidate SINR captures")
        for cand, rbgs in expected.items():
            if set(self._samples[cand]) != rbgs:
                raise ValueError("Captured RBGs do not match final allocation")
        if not expected:
            self.clear()
            return {}

        # Compatible grant shapes share one Sionna call instead of one/UE.
        grouped: dict[tuple[int,int,int], list[tuple[int,torch.Tensor]]] = defaultdict(list)
        for cand, rbgs in sorted(expected.items()):
            sinr = torch.cat([self._samples[cand][r] for r in sorted(rbgs)],dim=1)
            if sinr.shape[0] not in (1,config.num_slot_ofdm_symbols):
                raise ValueError("Unexpected number of OFDM symbols in post-SINR")
            if sinr.shape[0]==1:
                sinr=sinr.expand(config.num_slot_ofdm_symbols,-1,-1)
            grouped[(sinr.shape[0],sinr.shape[1],sinr.shape[2])].append((cand,sinr))

        phy, illa = create_link_adaptation_blocks(config)
        result: dict[int, WholeGrantLink] = {}
        for (_, _, layers), items in grouped.items():
            # [batch=1, symbols, subcarriers, UE-group, rank]
            sinrs = torch.stack([x[1] for x in items],dim=2).unsqueeze(0)
            if not bool(torch.isfinite(sinrs).all()) or bool((sinrs < 0).any()):
                raise ValueError("Invalid post-MU-MIMO SINR")
            mcs, _ = illa(
                sinr=sinrs,
                mcs_table_index=config.mcs_table_index,
                mcs_category=config.mcs_category,
                return_lowest_available_mcs=True,
            )
            _, _, _, tbler, _ = phy(
                mcs_index=mcs,
                sinr=sinrs,
                mcs_table_index=config.mcs_table_index,
                mcs_category=config.mcs_category,
            )
            # One bulk transfer for both final small UE vectors.
            mcs_values = mcs.squeeze(0).detach().to(device='cpu',dtype=torch.long).tolist()
            tbler_values = tbler.squeeze(0).detach().to(device='cpu').tolist()
            if not isinstance(mcs_values,list): mcs_values=[mcs_values]
            if not isinstance(tbler_values,list): tbler_values=[tbler_values]
            for (candidate,_), mcs_idx, p in zip(items,mcs_values,tbler_values):
                result[candidate] = WholeGrantLink(
                    candidate_index=candidate,
                    assigned_rbgs=tuple(sorted(expected[candidate])),
                    mcs_index=int(mcs_idx),
                    tbler=float(p),
                    num_layers=layers,
                )
        self.clear()  # free GPU snapshot tensors before the next TTI
        return result
