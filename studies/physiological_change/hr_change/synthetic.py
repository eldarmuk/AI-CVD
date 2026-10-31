"""Explicitly invented contract examples. These are not biomedical study evidence."""

from .contracts import Observation, PairedEpisode, ReferenceSummary, Window


def synthetic_episodes() -> list[PairedEpisode]:
    # last two fields are reference means, independent of the band estimates.
    specifications = [
        ("stable", [75, 75, 75], [76, 74, 75], 75, 75, 0),
        ("rise_stale_prefix", [75, 75, 75], [75, 75, 105], 75, 105, 0),
        ("fall_stale_prefix", [105, 105, 105], [105, 105, 75], 105, 75, 0),
        ("artifact_same_input_as_rise", [75, 75, 75], [75, 75, 105], 75, 75, 0),
        ("rise_missing_input", [75, 75, 75], [], 75, 105, 0),
        ("uncertain_reference", [75, 75, 75], [85, 85, 85], 75, 85, 2),
        ("unavailable_reference", [75, 75, 75], [90, 90, 90], None, None, None),
        ("wrong_direction", [105, 105, 105], [95, 95, 95], 75, 105, 0),
        ("stable_with_fixed_bias", [95, 95, 95], [95, 95, 95], 75, 75, 0),
        ("rise_supported", [75, 75, 75], [105, 105, 105], 75, 105, 0),
    ]
    output = []
    for index, (name, before, after, ref_before, ref_after, bound) in enumerate(specifications):
        offset = index * 180

        def observations(values, start):
            return tuple(
                Observation(start + n * 10, start + n * 10 + 1, value)
                for n, value in enumerate(values)
            )

        def reference(value):
            return ReferenceSummary(
                value, bound if value is not None else None, 10 if value is not None else 0
            )

        episode = PairedEpisode(
            episode_id=f"synthetic_{name}",
            participant_key=f"synthetic_person_{index // 2 + 1:02d}",
            dataset_id="invented_contract_examples_v1",
            decision_time_s=offset + 120,
            baseline_window=Window(offset, offset + 60),
            recent_window=Window(offset + 60, offset + 120),
            baseline_observations=observations(before, offset),
            recent_observations=observations(after, offset + 60),
            baseline_reference=reference(ref_before),
            recent_reference=reference(ref_after),
        )
        episode.validate()
        output.append(episode)
    return output
