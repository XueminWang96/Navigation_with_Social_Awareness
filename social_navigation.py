from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Sequence


EPSILON = 1e-6
TAU = math.tau


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle(angle: float) -> float:
    while angle <= -math.pi:
        angle += TAU
    while angle > math.pi:
        angle -= TAU
    return angle


def lerp_angle(a0: float, a1: float, alpha: float) -> float:
    delta = wrap_angle(a1 - a0)
    return wrap_angle(a0 + alpha * delta)


def add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def scale(vector: tuple[float, float], gain: float) -> tuple[float, float]:
    return vector[0] * gain, vector[1] * gain


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return norm(sub(a, b))


def normalize(vector: tuple[float, float]) -> tuple[float, float]:
    magnitude = norm(vector)
    if magnitude < EPSILON:
        return 0.0, 0.0
    return vector[0] / magnitude, vector[1] / magnitude


def angle_of(vector: tuple[float, float]) -> float:
    return math.atan2(vector[1], vector[0])


def from_angle(angle: float) -> tuple[float, float]:
    return math.cos(angle), math.sin(angle)


def positive_angle_diff(start: float, end: float) -> float:
    delta = end - start
    while delta <= 0.0:
        delta += TAU
    return delta


def midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def perpendicular_left(vector: tuple[float, float]) -> tuple[float, float]:
    return -vector[1], vector[0]


def cross_z(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def limit_translation(
    origin: tuple[float, float],
    target: tuple[float, float],
    max_distance: float,
) -> tuple[tuple[float, float], bool]:
    offset = sub(target, origin)
    offset_norm = norm(offset)
    if offset_norm <= max_distance or offset_norm < EPSILON:
        return target, False
    limited = add(origin, scale(offset, max_distance / offset_norm))
    return limited, True


def affinity_value(person: PersonState) -> float:
    return clamp(person.affinity, 0.0, 1.0)


def affinity_bias(person: PersonState) -> float:
    return 2.0 * (affinity_value(person) - 0.5)


def spacing_distance_for_affinity(
    base_distance: float,
    affinity_score: float,
    params: SocialNavParams,
) -> float:
    scale = 1.0 - params.affinity_spacing_gain * (2.0 * affinity_score - 1.0)
    return max(EPSILON, base_distance * scale)


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class PersonState:
    person_id: str
    x: float
    y: float
    yaw: float | None = None
    engagement: float = 1.0
    speaking_score: float = 0.0
    inclusion_weight: float = 1.0
    affinity: float = 0.5

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class SocialNavParams:
    personal_distance: float = 1.05
    preferred_neighbor_distance: float = 1.35
    ring_padding: float = 0.55
    max_radial_extension: float = 1.85
    min_gap_radians: float = math.radians(28.0)
    gap_weight: float = 2.0
    travel_weight: float = 0.18
    shortest_distance_weight: float = 0.22
    comfort_weight: float = 0.95
    frontality_weight: float = 0.65
    radial_weight: float = 0.85
    center_weight: float = 0.7
    clearance_weight: float = 0.85
    settle_radius: float = 0.22
    approach_center_blend: float = 0.3
    gaze_dwell_s: float = 1.8
    gaze_speaker_bonus: float = 2.2
    gaze_group_scan_bonus: float = 0.35
    affinity_weight: float = 1.1
    affinity_spacing_gain: float = 0.18
    affinity_focus_weight: float = 0.75
    affinity_angular_bias_gain: float = 0.22
    affinity_slot_margin: float = 0.18
    inviter_neighbor_bonus: float = 0.85
    interaction_weibull_a: float = 5.102
    interaction_weibull_b: float = 0.748
    interaction_weibull_c: float = 0.087
    interaction_probability_threshold: float = 0.22
    path_grid_resolution: float = 0.18
    path_margin: float = 0.95
    path_person_weight: float = 1.15
    path_pair_weight: float = 6.0
    path_group_weight: float = 7.5
    path_person_sigma_front: float = 0.95
    path_person_sigma_side: float = 0.68
    path_person_sigma_back: float = 0.55
    path_bridge_sigma: float = 0.32
    path_group_sigma: float = 0.58
    path_block_cost: float = 14.0
    path_visual_stride: int = 2
    body_turn_rate: float = 1.8
    move_speed: float = 0.75


@dataclass
class SlotCandidate:
    slot_angle: float
    gap_radians: float
    radius: float
    target: tuple[float, float]
    score: float
    min_distance_to_people: float
    frontality: float
    travel_distance: float
    affinity_score: float
    neighbor_ids: tuple[str, str]


@dataclass
class SocialPlan:
    target_position: tuple[float, float]
    engage_body_yaw: float
    body_yaw: float
    gaze_yaw: float
    focus_point: tuple[float, float]
    focus_person_id: str | None
    stage: str
    group_center: tuple[float, float]
    group_radius: float
    slot_angle: float
    slot_gap_radians: float
    target_affinity: float
    path_points: list[tuple[float, float]] = field(default_factory=list)
    field_samples: list[tuple[float, float, float]] = field(default_factory=list)
    interaction_links: list[tuple[str, str, float]] = field(default_factory=list)
    candidates: list[SlotCandidate] = field(default_factory=list)


@dataclass
class SocialTransitPlan:
    target_position: tuple[float, float]
    body_yaw: float
    gaze_yaw: float
    stage: str
    group_center: tuple[float, float]
    group_radius: float
    path_points: list[tuple[float, float]] = field(default_factory=list)
    field_samples: list[tuple[float, float, float]] = field(default_factory=list)
    interaction_links: list[tuple[str, str, float]] = field(default_factory=list)
    direct_distance: float = 0.0
    path_length: float = 0.0
    extra_distance: float = 0.0
    distance_to_target: float = 0.0
    min_clearance: float = 0.0


@dataclass
class DyadTriadParams:
    dyad_personal_distance: float = 1.05
    triad_leg_distance: float = 1.45
    personal_distance: float = 1.0
    affinity_spacing_gain: float = 0.16
    admit_distance: float = 2.25
    pre_admit_threshold: float = 0.36
    admit_threshold: float = 0.58
    admit_distance_weight: float = 0.55
    admit_orientation_weight: float = 0.45
    continuity_weight: float = 0.85
    primary_continuity_weight: float = 0.8
    newcomer_inclusion_weight: float = 0.95
    same_side_bonus: float = 0.35
    max_tangential_shift: float = 0.45
    min_apex_offset: float = 0.45
    max_target_shift: float = 0.85
    settle_radius: float = 0.18


@dataclass
class DyadTriadPlan:
    target_position: tuple[float, float]
    body_yaw: float
    gaze_yaw: float
    focus_person_id: str | None
    stage: str
    joining_score: float
    admitted: bool
    shared_center: tuple[float, float]
    target_primary_distance: float
    target_newcomer_distance: float
    same_side_preserved: bool
    motion_limit_applied: bool


def weighted_centroid(people: Sequence[PersonState]) -> tuple[float, float]:
    if not people:
        return 0.0, 0.0
    weighted_positions = []
    weights = []
    for person in people:
        weight = max(person.engagement * person.inclusion_weight, 0.05)
        weighted_positions.append(scale(person.position, weight))
        weights.append(weight)
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return 0.0, 0.0
    sx = sum(point[0] for point in weighted_positions)
    sy = sum(point[1] for point in weighted_positions)
    return sx / total_weight, sy / total_weight


def estimate_group_center(people: Sequence[PersonState]) -> tuple[float, float]:
    centroid = weighted_centroid(people)
    # Keep the interaction center invariant to speaker changes.
    # When only body yaw changes, the o-space center should remain fixed unless
    # participant positions themselves move.
    return centroid


def group_radius(people: Sequence[PersonState], center: tuple[float, float], minimum: float = 0.55) -> float:
    if not people:
        return minimum
    radii = [distance(person.position, center) for person in people]
    return max(minimum, median(radii))


def ordered_people_by_angle(
    people: Sequence[PersonState], center: tuple[float, float]
) -> list[tuple[float, PersonState]]:
    return sorted(
        ((angle_of(sub(person.position, center)), person) for person in people),
        key=lambda item: item[0],
    )


def slot_radius_for_gap(
    human_ring_radius: float,
    gap_radians: float,
    preferred_distance: float,
    params: SocialNavParams,
) -> float:
    base_radius = human_ring_radius + params.ring_padding
    beta = gap_radians * 0.5
    min_possible = human_ring_radius * math.sin(beta)

    required_radius = base_radius
    if preferred_distance > min_possible + EPSILON:
        discriminant = preferred_distance * preferred_distance - human_ring_radius * human_ring_radius * math.sin(beta) ** 2
        required_radius = human_ring_radius * math.cos(beta) + math.sqrt(max(discriminant, 0.0))

    return clamp(
        max(base_radius, required_radius),
        base_radius,
        human_ring_radius + params.max_radial_extension,
    )


def slot_radius_for_person(
    person_radius: float,
    angle_delta: float,
    desired_distance: float,
    base_radius: float,
) -> float:
    min_possible = person_radius * abs(math.sin(angle_delta))
    required_radius = base_radius
    if desired_distance > min_possible + EPSILON:
        discriminant = desired_distance * desired_distance - person_radius * person_radius * math.sin(angle_delta) ** 2
        required_radius = person_radius * math.cos(angle_delta) + math.sqrt(max(discriminant, 0.0))
    return max(base_radius, required_radius)


def candidate_frontality(candidate: tuple[float, float], people: Sequence[PersonState]) -> float:
    scores = []
    for person in people:
        if person.yaw is None:
            continue
        facing = from_angle(person.yaw)
        to_candidate = normalize(sub(candidate, person.position))
        scores.append(0.5 * (dot(facing, to_candidate) + 1.0))
    if not scores:
        return 0.5
    return sum(scores) / len(scores)


def candidate_affinity_score(
    candidate: tuple[float, float],
    people: Sequence[PersonState],
    neighbor_people: Sequence[PersonState],
) -> float:
    if not people:
        return 0.5

    local_proximity_total = 0.0
    local_affinity_total = 0.0
    for person in neighbor_people:
        proximity = 1.0 / max(distance(candidate, person.position), 0.25)
        local_proximity_total += proximity
        local_affinity_total += proximity * affinity_value(person)
    local_affinity = (
        local_affinity_total / local_proximity_total
        if local_proximity_total > EPSILON
        else 0.5
    )

    proximity_total = 0.0
    weighted_affinity_total = 0.0
    for person in people:
        proximity = 1.0 / max(distance(candidate, person.position), 0.25)
        proximity_total += proximity
        weighted_affinity_total += proximity * affinity_value(person)

    if proximity_total < EPSILON:
        return local_affinity

    proximity_affinity = weighted_affinity_total / proximity_total
    return 0.8 * local_affinity + 0.2 * proximity_affinity


def candidate_affinity_comfort(
    candidate: tuple[float, float],
    people: Sequence[PersonState],
    params: SocialNavParams,
) -> float:
    if not people:
        return 0.0

    weighted_total = 0.0
    weight_sum = 0.0
    for person in people:
        candidate_distance = distance(candidate, person.position)
        desired_distance = spacing_distance_for_affinity(
            params.preferred_neighbor_distance,
            affinity_value(person),
            params,
        )
        comfort = clamp(
            (candidate_distance - desired_distance) / max(desired_distance, 0.1),
            -1.0,
            1.0,
        )
        weight = 1.0 / max(candidate_distance, 0.35)
        weighted_total += weight * comfort
        weight_sum += weight
    if weight_sum < EPSILON:
        return 0.0
    return weighted_total / weight_sum


def build_slot_candidates(
    robot_pose: Pose2D,
    people: Sequence[PersonState],
    center: tuple[float, float],
    params: SocialNavParams,
) -> list[SlotCandidate]:
    sorted_people = ordered_people_by_angle(people, center)
    if len(sorted_people) < 2:
        return []

    human_ring = group_radius(people, center)
    candidates: list[SlotCandidate] = []

    for index, (angle_start, person_start) in enumerate(sorted_people):
        angle_end, person_end = sorted_people[(index + 1) % len(sorted_people)]
        gap = positive_angle_diff(angle_start, angle_end)
        slot_fraction = clamp(
            0.5 + params.affinity_angular_bias_gain * (affinity_value(person_end) - affinity_value(person_start)),
            params.affinity_slot_margin,
            1.0 - params.affinity_slot_margin,
        )
        slot_angle = wrap_angle(angle_start + gap * slot_fraction)
        base_radius = human_ring + params.ring_padding
        desired_start_distance = spacing_distance_for_affinity(
            params.preferred_neighbor_distance,
            affinity_value(person_start),
            params,
        )
        desired_end_distance = spacing_distance_for_affinity(
            params.preferred_neighbor_distance,
            affinity_value(person_end),
            params,
        )
        start_angle_delta = positive_angle_diff(angle_start, slot_angle)
        end_angle_delta = positive_angle_diff(slot_angle, angle_end)
        radius = max(
            base_radius,
            slot_radius_for_person(
                distance(person_start.position, center),
                start_angle_delta,
                desired_start_distance,
                base_radius,
            ),
            slot_radius_for_person(
                distance(person_end.position, center),
                end_angle_delta,
                desired_end_distance,
                base_radius,
            ),
        )
        radius = clamp(radius, base_radius, human_ring + params.max_radial_extension)
        target = add(center, scale(from_angle(slot_angle), radius))
        min_distance = min(distance(target, person.position) for person in people)
        affinity_score = candidate_affinity_score(target, people, (person_start, person_end))
        frontality = candidate_frontality(target, people)
        travel = distance(robot_pose.position, target)
        comfort_score = candidate_affinity_comfort(target, people, params)
        clearance_score = clamp(
            (min_distance - params.personal_distance) / max(params.personal_distance, 0.1),
            -1.0,
            1.0,
        )
        gap_score = gap / math.pi
        radial_penalty = abs(radius - (human_ring + params.ring_padding))
        score = (
            params.gap_weight * gap_score
            + params.comfort_weight * comfort_score
            + params.clearance_weight * clearance_score
            + params.frontality_weight * frontality
            + params.affinity_weight * (2.0 * affinity_score - 1.0)
            - params.travel_weight * travel
            - params.shortest_distance_weight * travel
            - params.radial_weight * radial_penalty
        )
        if gap < params.min_gap_radians:
            score -= 2.5

        candidates.append(
            SlotCandidate(
                slot_angle=slot_angle,
                gap_radians=gap,
                radius=radius,
                target=target,
                score=score,
                min_distance_to_people=min_distance,
                frontality=frontality,
                travel_distance=travel,
                affinity_score=affinity_score,
                neighbor_ids=(person_start.person_id, person_end.person_id),
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.travel_distance,
            -candidate.min_distance_to_people,
            -candidate.frontality,
        ),
    )


def select_slot_candidate(
    candidates: Sequence[SlotCandidate],
    inviter_person_id: str | None,
    params: SocialNavParams,
) -> SlotCandidate | None:
    if not candidates:
        return None

    if inviter_person_id:
        inviter_adjacent = [
            candidate
            for candidate in candidates
            if inviter_person_id in candidate.neighbor_ids
        ]
        if inviter_adjacent:
            return max(
                inviter_adjacent,
                key=lambda candidate: (
                    candidate.score + params.inviter_neighbor_bonus,
                    -candidate.travel_distance,
                    candidate.min_distance_to_people,
                    candidate.frontality,
                ),
            )

    return max(
        candidates,
        key=lambda candidate: (
            candidate.score,
            -candidate.travel_distance,
            candidate.min_distance_to_people,
            candidate.frontality,
        ),
    )


def single_person_target(
    robot_pose: Pose2D,
    person: PersonState,
    params: SocialNavParams,
) -> tuple[tuple[float, float], float]:
    if person.yaw is not None:
        direction = from_angle(person.yaw)
    else:
        direction = normalize(sub(robot_pose.position, person.position))
        if norm(direction) < EPSILON:
            direction = 1.0, 0.0
    target_distance = spacing_distance_for_affinity(params.personal_distance, affinity_value(person), params)
    target = add(person.position, scale(direction, target_distance))
    return target, angle_of(sub(person.position, target))


def weighted_focus_point(
    people: Sequence[PersonState],
    params: SocialNavParams | None = None,
) -> tuple[float, float]:
    params = params or SocialNavParams()
    if not people:
        return 0.0, 0.0
    weights = []
    sx = 0.0
    sy = 0.0
    for person in people:
        weight = max(
            0.1,
            person.engagement
            + person.speaking_score * 2.0
            + params.affinity_focus_weight * affinity_bias(person),
        )
        sx += person.x * weight
        sy += person.y * weight
        weights.append(weight)
    total = sum(weights)
    if total < EPSILON:
        return 0.0, 0.0
    return sx / total, sy / total


def default_gaze_target(
    people: Sequence[PersonState],
    center: tuple[float, float],
    params: SocialNavParams | None = None,
) -> tuple[tuple[float, float], str | None]:
    params = params or SocialNavParams()
    if not people:
        return center, None

    speakers = [person for person in people if person.speaking_score > 0.15]
    if speakers:
        focus = weighted_focus_point(speakers, params)
        focus_person = max(
            speakers,
            key=lambda person: (
                person.speaking_score,
                affinity_value(person),
                person.engagement,
            ),
        ).person_id
        return focus, focus_person
    return center, None


def interaction_gate(cos_theta: float) -> float:
    return max(cos_theta, 0.0)


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, tuple[float, float], float]:
    segment = sub(end, start)
    length_sq = dot(segment, segment)
    if length_sq < EPSILON:
        return distance(point, start), start, 0.0
    t = clamp(dot(sub(point, start), segment) / length_sq, 0.0, 1.0)
    projection = add(start, scale(segment, t))
    return distance(point, projection), projection, t


def interaction_probability_between(
    person_a: PersonState,
    person_b: PersonState,
    params: SocialNavParams,
) -> float:
    direction_ab = normalize(sub(person_b.position, person_a.position))
    direction_ba = scale(direction_ab, -1.0)
    if norm(direction_ab) < EPSILON:
        return 1.0

    if person_a.yaw is None:
        gate_a = 0.5
    else:
        gate_a = interaction_gate(dot(from_angle(person_a.yaw), direction_ab))

    if person_b.yaw is None:
        gate_b = 0.5
    else:
        gate_b = interaction_gate(dot(from_angle(person_b.yaw), direction_ba))

    distance_sq = max(distance(person_a.position, person_b.position) ** 2, EPSILON)
    scale_term = (
        params.interaction_weibull_a
        * (gate_a + params.interaction_weibull_c)
        * (gate_b + params.interaction_weibull_c)
        / distance_sq
    )
    return clamp(1.0 - math.exp(-(scale_term ** params.interaction_weibull_b)), 0.0, 1.0)


def build_interaction_graph(
    people: Sequence[PersonState],
    params: SocialNavParams,
) -> list[tuple[int, int, float]]:
    interactions: list[tuple[int, int, float]] = []
    for index_a in range(len(people)):
        for index_b in range(index_a + 1, len(people)):
            probability = interaction_probability_between(people[index_a], people[index_b], params)
            if probability >= params.interaction_probability_threshold:
                interactions.append((index_a, index_b, probability))
    return sorted(interactions, key=lambda item: item[2], reverse=True)


def person_field_value(
    point: tuple[float, float],
    person: PersonState,
    params: SocialNavParams,
) -> float:
    relative = sub(point, person.position)
    if person.yaw is None:
        sigma_x = params.path_person_sigma_side
        sigma_y = params.path_person_sigma_side
        local_x = relative[0]
        local_y = relative[1]
    else:
        facing = from_angle(person.yaw)
        lateral = perpendicular_left(facing)
        local_x = dot(relative, facing)
        local_y = dot(relative, lateral)
        sigma_x = params.path_person_sigma_front if local_x >= 0.0 else params.path_person_sigma_back
        sigma_y = params.path_person_sigma_side
    return math.exp(
        -0.5 * ((local_x / max(sigma_x, EPSILON)) ** 2 + (local_y / max(sigma_y, EPSILON)) ** 2)
    )


def pair_bridge_field_value(
    point: tuple[float, float],
    person_a: PersonState,
    person_b: PersonState,
    params: SocialNavParams,
) -> float:
    segment_distance, _, segment_t = point_to_segment_distance(point, person_a.position, person_b.position)
    along_segment = math.exp(-0.5 * (((segment_t - 0.5) / 0.38) ** 2))
    across_segment = math.exp(-0.5 * ((segment_distance / max(params.path_bridge_sigma, EPSILON)) ** 2))
    return along_segment * across_segment


def group_center_field_value(
    point: tuple[float, float],
    center: tuple[float, float],
    radius: float,
    params: SocialNavParams,
) -> float:
    sigma = max(params.path_group_sigma, 0.3 + 0.25 * radius)
    return math.exp(-0.5 * ((distance(point, center) / sigma) ** 2))


def social_path_cost(
    point: tuple[float, float],
    people: Sequence[PersonState],
    center: tuple[float, float],
    group_radius_value: float,
    interaction_graph: Sequence[tuple[int, int, float]],
    params: SocialNavParams,
) -> float:
    person_term = sum(person_field_value(point, person, params) for person in people)
    pair_term = 0.0
    for index_a, index_b, probability in interaction_graph:
        pair_term += probability * pair_bridge_field_value(point, people[index_a], people[index_b], params)

    if interaction_graph:
        group_strength = sum(probability for _, _, probability in interaction_graph) / len(interaction_graph)
    else:
        group_strength = 0.0
    group_term = group_strength * group_center_field_value(point, center, group_radius_value, params)

    return (
        1.0
        + params.path_person_weight * person_term
        + params.path_pair_weight * pair_term
        + params.path_group_weight * group_term
    )


def compress_path(path_points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(path_points) <= 2:
        return list(path_points)

    compressed = [path_points[0]]
    for index in range(1, len(path_points) - 1):
        previous = compressed[-1]
        current = path_points[index]
        following = path_points[index + 1]
        previous_dir = normalize(sub(current, previous))
        next_dir = normalize(sub(following, current))
        if norm(previous_dir) < EPSILON or norm(next_dir) < EPSILON:
            continue
        if dot(previous_dir, next_dir) < 0.995 or distance(previous, current) > 0.42:
            compressed.append(current)
    compressed.append(path_points[-1])
    return compressed


def polyline_length(path_points: Sequence[tuple[float, float]]) -> float:
    if len(path_points) < 2:
        return 0.0
    return sum(
        distance(path_points[index - 1], path_points[index])
        for index in range(1, len(path_points))
    )


def path_minimum_clearance(
    path_points: Sequence[tuple[float, float]],
    people: Sequence[PersonState],
) -> float:
    if not path_points or not people:
        return 0.0

    if len(path_points) == 1:
        return min(distance(path_points[0], person.position) for person in people)

    minimum_clearance = float("inf")
    for start, end in zip(path_points, path_points[1:]):
        for person in people:
            segment_distance, _, _ = point_to_segment_distance(person.position, start, end)
            minimum_clearance = min(minimum_clearance, segment_distance)
    return 0.0 if minimum_clearance == float("inf") else minimum_clearance


def plan_social_path(
    start: tuple[float, float],
    goal: tuple[float, float],
    people: Sequence[PersonState],
    center: tuple[float, float],
    group_radius_value: float,
    params: SocialNavParams,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]], list[tuple[str, str, float]]]:
    if not people:
        return [start, goal], [], []

    interaction_graph = build_interaction_graph(people, params)
    interaction_links = [
        (people[index_a].person_id, people[index_b].person_id, probability)
        for index_a, index_b, probability in interaction_graph
    ]

    points_of_interest = [start, goal, center, *[person.position for person in people]]
    min_x = min(point[0] for point in points_of_interest) - params.path_margin
    max_x = max(point[0] for point in points_of_interest) + params.path_margin
    min_y = min(point[1] for point in points_of_interest) - params.path_margin
    max_y = max(point[1] for point in points_of_interest) + params.path_margin

    resolution = params.path_grid_resolution
    width = max(2, int(math.ceil((max_x - min_x) / resolution)) + 1)
    height = max(2, int(math.ceil((max_y - min_y) / resolution)) + 1)

    def grid_point(ix: int, iy: int) -> tuple[float, float]:
        return min_x + ix * resolution, min_y + iy * resolution

    def grid_index(point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round((point[0] - min_x) / resolution)),
            int(round((point[1] - min_y) / resolution)),
        )

    def in_bounds(ix: int, iy: int) -> bool:
        return 0 <= ix < width and 0 <= iy < height

    cost_grid = [[0.0 for _ in range(width)] for _ in range(height)]
    field_samples: list[tuple[float, float, float]] = []
    stride = max(1, params.path_visual_stride)
    for iy in range(height):
        for ix in range(width):
            point = grid_point(ix, iy)
            cost = social_path_cost(point, people, center, group_radius_value, interaction_graph, params)
            cost_grid[iy][ix] = cost
            if ix % stride == 0 and iy % stride == 0:
                field_samples.append((point[0], point[1], cost))

    start_ix, start_iy = grid_index(start)
    goal_ix, goal_iy = grid_index(goal)
    start_ix = clamp(start_ix, 0, width - 1)
    start_iy = clamp(start_iy, 0, height - 1)
    goal_ix = clamp(goal_ix, 0, width - 1)
    goal_iy = clamp(goal_iy, 0, height - 1)
    start_node = int(start_ix), int(start_iy)
    goal_node = int(goal_ix), int(goal_iy)

    frontier: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, start_node))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    g_score = {start_node: 0.0}
    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),            (1, 0),
        (-1, 1),  (0, 1),   (1, 1),
    ]

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_node:
            break

        for dx, dy in neighbors:
            next_ix = current[0] + dx
            next_iy = current[1] + dy
            if not in_bounds(next_ix, next_iy):
                continue
            next_node = (next_ix, next_iy)
            next_cost = cost_grid[next_iy][next_ix]
            if next_cost >= params.path_block_cost and next_node != goal_node:
                continue

            step_distance = resolution * (math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0)
            edge_cost = 0.5 * (cost_grid[current[1]][current[0]] + next_cost) * step_distance
            tentative_g = g_score[current] + edge_cost
            if tentative_g >= g_score.get(next_node, float("inf")):
                continue

            g_score[next_node] = tentative_g
            heuristic = distance(grid_point(next_ix, next_iy), goal)
            heapq.heappush(frontier, (tentative_g + heuristic, next_node))
            came_from[next_node] = current

    if goal_node not in came_from:
        return [start, goal], field_samples, interaction_links

    reconstructed_nodes = [goal_node]
    while reconstructed_nodes[-1] != start_node:
        parent = came_from[reconstructed_nodes[-1]]
        if parent is None:
            break
        reconstructed_nodes.append(parent)
    reconstructed_nodes.reverse()

    path_points = [start]
    for node in reconstructed_nodes[1:-1]:
        path_points.append(grid_point(node[0], node[1]))
    path_points.append(goal)
    return compress_path(path_points), field_samples, interaction_links


def compute_social_transit_plan(
    robot_pose: Pose2D,
    target_position: tuple[float, float],
    people: Sequence[PersonState],
    params: SocialNavParams | None = None,
) -> SocialTransitPlan:
    params = params or SocialNavParams()
    distance_to_target = distance(robot_pose.position, target_position)
    direct_heading = (
        angle_of(sub(target_position, robot_pose.position))
        if distance_to_target > EPSILON
        else robot_pose.yaw
    )

    if not people:
        return SocialTransitPlan(
            target_position=target_position,
            body_yaw=direct_heading,
            gaze_yaw=direct_heading,
            stage="arrived" if distance_to_target <= params.settle_radius else "transit",
            group_center=target_position,
            group_radius=0.0,
            path_points=[robot_pose.position, target_position],
            field_samples=[],
            interaction_links=[],
            direct_distance=distance_to_target,
            path_length=distance_to_target,
            extra_distance=0.0,
            distance_to_target=distance_to_target,
            min_clearance=0.0,
        )

    center = estimate_group_center(people)
    human_ring = group_radius(people, center)
    path_points, field_samples, interaction_links = plan_social_path(
        robot_pose.position,
        target_position,
        people,
        center,
        human_ring,
        params,
    )
    follow_target = next_path_waypoint(
        robot_pose.position,
        target_position,
        path_points,
        max(params.path_grid_resolution * 0.8, params.settle_radius * 0.6),
    )
    follow_distance = distance(robot_pose.position, follow_target)
    body_yaw = (
        angle_of(sub(follow_target, robot_pose.position))
        if follow_distance > EPSILON
        else direct_heading
    )
    path_length = polyline_length(path_points)
    extra_distance = max(0.0, path_length - distance_to_target)

    if distance_to_target <= params.settle_radius:
        stage = "arrived"
    elif extra_distance > max(params.path_grid_resolution * 1.5, 0.12):
        stage = "detour"
    else:
        stage = "transit"

    return SocialTransitPlan(
        target_position=target_position,
        body_yaw=body_yaw,
        gaze_yaw=direct_heading,
        stage=stage,
        group_center=center,
        group_radius=human_ring,
        path_points=path_points,
        field_samples=field_samples,
        interaction_links=interaction_links,
        direct_distance=distance_to_target,
        path_length=path_length,
        extra_distance=extra_distance,
        distance_to_target=distance_to_target,
        min_clearance=path_minimum_clearance(path_points, people),
    )


def next_path_waypoint(
    robot_position: tuple[float, float],
    target_position: tuple[float, float],
    path_points: Sequence[tuple[float, float]],
    tolerance: float,
) -> tuple[float, float]:
    for waypoint in path_points:
        if distance(robot_position, waypoint) > tolerance:
            return waypoint
    return target_position


def next_locked_path_waypoint(
    robot_position: tuple[float, float],
    target_position: tuple[float, float],
    path_points: Sequence[tuple[float, float]],
    tolerance: float,
) -> tuple[float, float]:
    if not path_points:
        return target_position

    nearest_index = min(
        range(len(path_points)),
        key=lambda index: distance(robot_position, path_points[index]),
    )
    for waypoint in path_points[nearest_index:]:
        if distance(robot_position, waypoint) > tolerance:
            return waypoint
    return target_position


def compute_dyad_to_triad_plan(
    robot_pose: Pose2D,
    primary_person: PersonState,
    newcomer_person: PersonState,
    params: DyadTriadParams | None = None,
) -> DyadTriadPlan:
    params = params or DyadTriadParams()

    # Baseline dyad target: keep a stable one-on-one interaction with the current user.
    if primary_person.yaw is not None:
        primary_facing = from_angle(primary_person.yaw)
    else:
        primary_facing = normalize(sub(robot_pose.position, primary_person.position))
        if norm(primary_facing) < EPSILON:
            primary_facing = 1.0, 0.0
    dyad_distance = spacing_distance_for_affinity(
        params.dyad_personal_distance,
        affinity_value(primary_person),
        params,
    )
    dyad_target = add(primary_person.position, scale(primary_facing, dyad_distance))

    dyad_center = midpoint(primary_person.position, robot_pose.position)
    newcomer_to_dyad = sub(dyad_center, newcomer_person.position)
    distance_score = clamp(
        1.0 - distance(newcomer_person.position, dyad_center) / max(params.admit_distance, 0.1),
        0.0,
        1.0,
    )
    if newcomer_person.yaw is None:
        orientation_score = 0.5
    else:
        newcomer_facing = from_angle(newcomer_person.yaw)
        orientation_score = 0.5 * (dot(newcomer_facing, normalize(newcomer_to_dyad)) + 1.0)
    joining_score = (
        params.admit_distance_weight * distance_score
        + params.admit_orientation_weight * orientation_score
    )

    if joining_score < params.pre_admit_threshold:
        body_yaw = angle_of(sub(primary_person.position, robot_pose.position))
        gaze_yaw = body_yaw
        return DyadTriadPlan(
            target_position=dyad_target,
            body_yaw=body_yaw,
            gaze_yaw=gaze_yaw,
            focus_person_id=primary_person.person_id,
            stage="hold_dyad",
            joining_score=joining_score,
            admitted=False,
            shared_center=dyad_center,
            target_primary_distance=distance(dyad_target, primary_person.position),
            target_newcomer_distance=distance(dyad_target, newcomer_person.position),
            same_side_preserved=True,
            motion_limit_applied=False,
        )

    pair_midpoint = midpoint(primary_person.position, newcomer_person.position)
    pair_vector = sub(newcomer_person.position, primary_person.position)
    pair_length = max(norm(pair_vector), EPSILON)
    pair_tangent = normalize(pair_vector)
    if norm(pair_tangent) < EPSILON:
        pair_tangent = 1.0, 0.0
    pair_normal = perpendicular_left(pair_tangent)

    desired_primary_distance = spacing_distance_for_affinity(
        params.triad_leg_distance,
        affinity_value(primary_person),
        params,
    )
    desired_newcomer_distance = spacing_distance_for_affinity(
        params.triad_leg_distance,
        affinity_value(newcomer_person),
        params,
    )
    tangential_shift = clamp(
        (desired_primary_distance * desired_primary_distance - desired_newcomer_distance * desired_newcomer_distance)
        / max(2.0 * pair_length, EPSILON),
        -params.max_tangential_shift,
        params.max_tangential_shift,
    )
    average_squared_distance = 0.5 * (
        desired_primary_distance * desired_primary_distance
        + desired_newcomer_distance * desired_newcomer_distance
    )
    apex_offset = math.sqrt(
        max(
            average_squared_distance - 0.25 * pair_length * pair_length - tangential_shift * tangential_shift,
            params.min_apex_offset * params.min_apex_offset,
        )
    )

    robot_side = cross_z(pair_vector, sub(robot_pose.position, pair_midpoint))
    if abs(robot_side) < EPSILON:
        robot_side = cross_z(pair_vector, sub(primary_person.position, dyad_target))
    preferred_sign = 1.0 if robot_side >= 0.0 else -1.0

    best_target = dyad_target
    best_score = -float("inf")
    best_same_side = True
    best_primary_distance = distance(dyad_target, primary_person.position)
    best_newcomer_distance = distance(dyad_target, newcomer_person.position)

    for normal_sign in (1.0, -1.0):
        candidate = add(
            pair_midpoint,
            add(
                scale(pair_tangent, tangential_shift),
                scale(pair_normal, normal_sign * apex_offset),
            ),
        )
        primary_distance = distance(candidate, primary_person.position)
        newcomer_distance = distance(candidate, newcomer_person.position)
        clearance_score = clamp(
            (
                min(primary_distance, newcomer_distance) - params.personal_distance
            ) / max(params.triad_leg_distance, 0.1),
            -1.0,
            1.0,
        )
        displacement_cost = distance(robot_pose.position, candidate)
        primary_error = abs(primary_distance - desired_primary_distance)
        newcomer_error = abs(newcomer_distance - desired_newcomer_distance)
        same_side = normal_sign == preferred_sign
        score = (
            clearance_score
            - params.continuity_weight * displacement_cost
            - params.primary_continuity_weight * primary_error
            - params.newcomer_inclusion_weight * newcomer_error
            + (params.same_side_bonus if same_side else 0.0)
        )
        if score > best_score:
            best_score = score
            best_target = candidate
            best_same_side = same_side
            best_primary_distance = primary_distance
            best_newcomer_distance = newcomer_distance

    limited_target, motion_limit_applied = limit_translation(
        robot_pose.position,
        best_target,
        params.max_target_shift,
    )

    if joining_score < params.admit_threshold:
        focus_point = pair_midpoint
        body_yaw = angle_of(sub(focus_point, robot_pose.position))
        gaze_yaw = angle_of(sub(newcomer_person.position, robot_pose.position))
        return DyadTriadPlan(
            target_position=robot_pose.position,
            body_yaw=body_yaw,
            gaze_yaw=gaze_yaw,
            focus_person_id=newcomer_person.person_id,
            stage="open_to_newcomer",
            joining_score=joining_score,
            admitted=False,
            shared_center=pair_midpoint,
            target_primary_distance=distance(robot_pose.position, primary_person.position),
            target_newcomer_distance=distance(robot_pose.position, newcomer_person.position),
            same_side_preserved=best_same_side,
            motion_limit_applied=False,
        )

    newcomer_priority = newcomer_person.speaking_score + 0.45 * affinity_value(newcomer_person)
    primary_priority = primary_person.speaking_score + 0.45 * affinity_value(primary_person)
    if newcomer_priority > primary_priority + 0.05:
        focus_person = newcomer_person
    elif primary_priority > newcomer_priority + 0.05:
        focus_person = primary_person
    else:
        focus_person = None

    shared_center = midpoint(primary_person.position, newcomer_person.position)
    if focus_person is None:
        gaze_reference = shared_center
        focus_person_id = None
    else:
        gaze_reference = focus_person.position
        focus_person_id = focus_person.person_id

    body_reference = shared_center
    distance_to_target = distance(robot_pose.position, limited_target)
    stage = "stabilize_triad" if distance_to_target <= params.settle_radius else "reposition_to_triad"

    return DyadTriadPlan(
        target_position=limited_target,
        body_yaw=angle_of(sub(body_reference, robot_pose.position)),
        gaze_yaw=angle_of(sub(gaze_reference, robot_pose.position)),
        focus_person_id=focus_person_id,
        stage=stage,
        joining_score=joining_score,
        admitted=True,
        shared_center=shared_center,
        target_primary_distance=best_primary_distance,
        target_newcomer_distance=best_newcomer_distance,
        same_side_preserved=best_same_side,
        motion_limit_applied=motion_limit_applied,
    )


def compute_social_plan(
    robot_pose: Pose2D,
    people: Sequence[PersonState],
    params: SocialNavParams | None = None,
    inviter_person_id: str | None = None,
) -> SocialPlan:
    params = params or SocialNavParams()

    if not people:
        return SocialPlan(
            target_position=robot_pose.position,
            engage_body_yaw=robot_pose.yaw,
            body_yaw=robot_pose.yaw,
            gaze_yaw=robot_pose.yaw,
            focus_point=robot_pose.position,
            focus_person_id=None,
            stage="idle",
            group_center=robot_pose.position,
            group_radius=0.0,
            slot_angle=robot_pose.yaw,
            slot_gap_radians=0.0,
            target_affinity=0.5,
            path_points=[],
            field_samples=[],
            interaction_links=[],
        )

    center = estimate_group_center(people)
    human_ring = group_radius(people, center)

    if len(people) == 1:
        target_position, engage_body_yaw = single_person_target(robot_pose, people[0], params)
        slot_angle = angle_of(sub(target_position, center))
        slot_gap = TAU
        target_affinity = affinity_value(people[0])
        candidates: list[SlotCandidate] = []
    else:
        candidates = build_slot_candidates(robot_pose, people, center, params)
        best = select_slot_candidate(candidates, inviter_person_id, params)
        if best is None:
            fallback_angle = angle_of(sub(robot_pose.position, center))
            target_radius = human_ring + params.ring_padding
            target_position = add(center, scale(from_angle(fallback_angle), target_radius))
            slot_angle = fallback_angle
            slot_gap = 0.0
            target_affinity = 0.5
        else:
            if candidates and candidates[0] is not best:
                candidates = [best, *[candidate for candidate in candidates if candidate is not best]]
            target_position = best.target
            slot_angle = best.slot_angle
            slot_gap = best.gap_radians
            target_affinity = best.affinity_score
        engage_body_yaw = angle_of(sub(center, target_position))

    path_points, field_samples, interaction_links = plan_social_path(
        robot_pose.position,
        target_position,
        people,
        center,
        human_ring,
        params,
    )
    follow_target = next_path_waypoint(
        robot_pose.position,
        target_position,
        path_points,
        max(params.path_grid_resolution * 0.8, params.settle_radius * 0.6),
    )

    travel_heading = angle_of(sub(follow_target, robot_pose.position))
    current_center_heading = angle_of(sub(center, robot_pose.position))
    distance_to_target = distance(robot_pose.position, target_position)
    stage = "engage" if distance_to_target <= params.settle_radius else "approach"
    if stage == "engage":
        body_yaw = engage_body_yaw
    else:
        body_yaw = lerp_angle(travel_heading, current_center_heading, params.approach_center_blend)

    focus_point, focus_person_id = default_gaze_target(people, center, params)
    gaze_reference = focus_point if stage == "engage" else center
    gaze_yaw = angle_of(sub(gaze_reference, robot_pose.position))

    return SocialPlan(
        target_position=target_position,
        engage_body_yaw=engage_body_yaw,
        body_yaw=body_yaw,
        gaze_yaw=gaze_yaw,
        focus_point=focus_point,
        focus_person_id=focus_person_id,
        stage=stage,
        group_center=center,
        group_radius=human_ring,
        slot_angle=slot_angle,
        slot_gap_radians=slot_gap,
        target_affinity=target_affinity,
        path_points=path_points,
        field_samples=field_samples,
        interaction_links=interaction_links,
        candidates=candidates,
    )


def compute_social_plan_with_locked_target(
    robot_pose: Pose2D,
    people: Sequence[PersonState],
    locked_plan: SocialPlan,
    params: SocialNavParams | None = None,
) -> SocialPlan:
    params = params or SocialNavParams()

    if not people:
        return SocialPlan(
            target_position=robot_pose.position,
            engage_body_yaw=robot_pose.yaw,
            body_yaw=robot_pose.yaw,
            gaze_yaw=robot_pose.yaw,
            focus_point=robot_pose.position,
            focus_person_id=None,
            stage="idle",
            group_center=robot_pose.position,
            group_radius=0.0,
            slot_angle=robot_pose.yaw,
            slot_gap_radians=0.0,
            target_affinity=0.5,
            path_points=[],
            field_samples=[],
            interaction_links=[],
            candidates=[],
        )

    center = estimate_group_center(people)
    human_ring = group_radius(people, center)
    target_position = locked_plan.target_position
    engage_body_yaw = angle_of(sub(center, target_position))
    path_points, field_samples, interaction_links = plan_social_path(
        robot_pose.position,
        target_position,
        people,
        center,
        human_ring,
        params,
    )
    follow_target = next_path_waypoint(
        robot_pose.position,
        target_position,
        path_points,
        max(params.path_grid_resolution * 0.8, params.settle_radius * 0.6),
    )

    travel_heading = angle_of(sub(follow_target, robot_pose.position))
    current_center_heading = angle_of(sub(center, robot_pose.position))
    distance_to_target = distance(robot_pose.position, target_position)
    stage = "engage" if distance_to_target <= params.settle_radius else "approach"
    if stage == "engage":
        body_yaw = engage_body_yaw
    else:
        body_yaw = lerp_angle(travel_heading, current_center_heading, params.approach_center_blend)

    focus_point, focus_person_id = default_gaze_target(people, center, params)
    gaze_reference = focus_point if stage == "engage" else center
    gaze_yaw = angle_of(sub(gaze_reference, robot_pose.position))

    return SocialPlan(
        target_position=target_position,
        engage_body_yaw=engage_body_yaw,
        body_yaw=body_yaw,
        gaze_yaw=gaze_yaw,
        focus_point=focus_point,
        focus_person_id=focus_person_id,
        stage=stage,
        group_center=center,
        group_radius=human_ring,
        slot_angle=locked_plan.slot_angle,
        slot_gap_radians=locked_plan.slot_gap_radians,
        target_affinity=locked_plan.target_affinity,
        path_points=path_points,
        field_samples=field_samples,
        interaction_links=interaction_links,
        candidates=locked_plan.candidates,
    )


class SocialNavigator:
    def __init__(self, params: SocialNavParams | None = None) -> None:
        self.params = params or SocialNavParams()
        self._focus_person_id: str | None = None
        self._focus_switch_deadline = 0.0
        self._round_robin_index = 0

    def reset_attention(self) -> None:
        self._focus_person_id = None
        self._focus_switch_deadline = 0.0
        self._round_robin_index = 0

    def update(
        self,
        robot_pose: Pose2D,
        people: Sequence[PersonState],
        now_s: float | None = None,
        inviter_person_id: str | None = None,
        precomputed_plan: SocialPlan | None = None,
    ) -> SocialPlan:
        now_s = time.monotonic() if now_s is None else now_s
        plan = precomputed_plan or compute_social_plan(robot_pose, people, self.params, inviter_person_id=inviter_person_id)
        focus_point, focus_person_id = self._update_focus(robot_pose, people, plan, now_s)
        return SocialPlan(
            target_position=plan.target_position,
            engage_body_yaw=plan.engage_body_yaw,
            body_yaw=plan.body_yaw,
            gaze_yaw=angle_of(sub(focus_point, robot_pose.position)),
            focus_point=focus_point,
            focus_person_id=focus_person_id,
            stage=plan.stage,
            group_center=plan.group_center,
            group_radius=plan.group_radius,
            slot_angle=plan.slot_angle,
            slot_gap_radians=plan.slot_gap_radians,
            target_affinity=plan.target_affinity,
            path_points=plan.path_points,
            field_samples=plan.field_samples,
            interaction_links=plan.interaction_links,
            candidates=plan.candidates,
        )

    def _update_focus(
        self,
        robot_pose: Pose2D,
        people: Sequence[PersonState],
        plan: SocialPlan,
        now_s: float,
    ) -> tuple[tuple[float, float], str | None]:
        if not people:
            self.reset_attention()
            return robot_pose.position, None

        if plan.stage != "engage":
            self._focus_person_id = None
            return plan.group_center, None

        candidate_people = [person for person in people if person.speaking_score > 0.18]
        candidate_people = sorted(
            candidate_people or people,
            key=lambda person: (
                -(
                    person.engagement
                    + person.speaking_score * self.params.gaze_speaker_bonus
                    + self.params.affinity_focus_weight * affinity_bias(person)
                ),
                person.person_id,
            ),
        )

        candidate_ids = [person.person_id for person in candidate_people]
        if not candidate_ids:
            self.reset_attention()
            return plan.group_center, None

        if (
            self._focus_person_id in candidate_ids
            and now_s < self._focus_switch_deadline
        ):
            focused = next(person for person in candidate_people if person.person_id == self._focus_person_id)
            return focused.position, focused.person_id

        if self._focus_person_id in candidate_ids:
            current_index = candidate_ids.index(self._focus_person_id)
            self._round_robin_index = (current_index + 1) % len(candidate_people)
        else:
            self._round_robin_index %= len(candidate_people)

        focused = candidate_people[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(candidate_people)
        self._focus_person_id = focused.person_id
        dwell_bonus = (
            focused.speaking_score * 0.5
            + self.params.gaze_group_scan_bonus * max(len(candidate_people) - 1, 0)
            + 0.35 * max(0.0, affinity_bias(focused))
        )
        self._focus_switch_deadline = now_s + self.params.gaze_dwell_s + dwell_bonus
        return focused.position, focused.person_id


def step_robot_toward_plan(
    robot_pose: Pose2D,
    plan: SocialPlan,
    dt_s: float,
    params: SocialNavParams | None = None,
) -> Pose2D:
    params = params or SocialNavParams()
    follow_target = next_path_waypoint(
        robot_pose.position,
        plan.target_position,
        plan.path_points,
        max(params.path_grid_resolution * 0.8, params.settle_radius * 0.6),
    )
    to_target = sub(follow_target, robot_pose.position)
    distance_to_target = norm(to_target)
    if distance_to_target > EPSILON:
        move_distance = min(distance_to_target, params.move_speed * dt_s)
        motion = scale(normalize(to_target), move_distance)
        new_position = add(robot_pose.position, motion)
    else:
        new_position = robot_pose.position

    yaw_error = wrap_angle(plan.body_yaw - robot_pose.yaw)
    max_turn = params.body_turn_rate * dt_s
    new_yaw = wrap_angle(robot_pose.yaw + clamp(yaw_error, -max_turn, max_turn))
    return Pose2D(new_position[0], new_position[1], new_yaw)


def step_robot_toward_transit_plan(
    robot_pose: Pose2D,
    plan: SocialTransitPlan,
    dt_s: float,
    params: SocialNavParams | None = None,
) -> Pose2D:
    params = params or SocialNavParams()
    follow_target = next_path_waypoint(
        robot_pose.position,
        plan.target_position,
        plan.path_points,
        max(params.path_grid_resolution * 0.8, params.settle_radius * 0.6),
    )
    to_target = sub(follow_target, robot_pose.position)
    distance_to_target = norm(to_target)
    if distance_to_target > EPSILON:
        move_distance = min(distance_to_target, params.move_speed * dt_s)
        motion = scale(normalize(to_target), move_distance)
        new_position = add(robot_pose.position, motion)
    else:
        new_position = robot_pose.position

    yaw_error = wrap_angle(plan.body_yaw - robot_pose.yaw)
    max_turn = params.body_turn_rate * dt_s
    new_yaw = wrap_angle(robot_pose.yaw + clamp(yaw_error, -max_turn, max_turn))
    return Pose2D(new_position[0], new_position[1], new_yaw)


def format_plan_summary(plan: SocialPlan) -> str:
    focus_label = plan.focus_person_id or "group"
    return (
        f"stage={plan.stage}, "
        f"target=({plan.target_position[0]:.2f}, {plan.target_position[1]:.2f}), "
        f"body={math.degrees(plan.body_yaw):.1f}deg, "
        f"gaze={math.degrees(plan.gaze_yaw):.1f}deg, "
        f"target_affinity={plan.target_affinity:.2f}, "
        f"focus={focus_label}"
    )


def orient_people_toward_speaker(
    people: Sequence[PersonState],
    speaker_person_id: str | None,
) -> None:
    if not people:
        return

    speaker = next((person for person in people if person.person_id == speaker_person_id), None)
    if speaker is None:
        center = estimate_group_center(people)
        for person in people:
            person.yaw = angle_of(sub(center, person.position))
        return

    listeners = [person for person in people if person.person_id != speaker.person_id]
    if listeners:
        speaker_focus = weighted_centroid(listeners)
        if distance(speaker.position, speaker_focus) > EPSILON:
            speaker.yaw = angle_of(sub(speaker_focus, speaker.position))
    for listener in listeners:
        listener.yaw = angle_of(sub(speaker.position, listener.position))


def demo_people(
    count: int = 4,
    speaker_person_id: str | None = "P1",
) -> list[PersonState]:
    count = max(1, min(4, count))
    center = (0.0, 0.0)
    radius = 1.15
    people = []
    default_affinities = [0.5, 0.5, 0.5, 0.5]
    preset_angles = {
        1: [math.radians(270.0)],
        2: [math.radians(220.0), math.radians(320.0)],
        3: [math.radians(205.0), math.radians(330.0), math.radians(88.0)],
        4: [math.radians(205.0), math.radians(308.0), math.radians(38.0), math.radians(128.0)],
    }
    for index, angle in enumerate(preset_angles[count]):
        position = add(center, scale(from_angle(angle), radius))
        yaw = angle_of(sub(center, position))
        people.append(
            PersonState(
                person_id=f"P{index + 1}",
                x=position[0],
                y=position[1],
                yaw=yaw,
                engagement=1.0,
                speaking_score=1.0 if f"P{index + 1}" == speaker_person_id else 0.0,
                affinity=default_affinities[index],
            )
        )
    orient_people_toward_speaker(people, speaker_person_id)
    return people


__all__ = [
    "DyadTriadParams",
    "DyadTriadPlan",
    "PersonState",
    "Pose2D",
    "SlotCandidate",
    "SocialNavParams",
    "SocialNavigator",
    "SocialPlan",
    "SocialTransitPlan",
    "compute_dyad_to_triad_plan",
    "compute_social_plan",
    "compute_social_plan_with_locked_target",
    "compute_social_transit_plan",
    "demo_people",
    "estimate_group_center",
    "format_plan_summary",
    "next_locked_path_waypoint",
    "orient_people_toward_speaker",
    "step_robot_toward_plan",
    "step_robot_toward_transit_plan",
]


def main() -> None:
    print("social_navigation.py is a library module.")
    print("Run `python3 social_navigation_demo.py` for the Tk demo.")
    print("Run `python3 social_navigation_demo.py --web` for the browser demo.")
    print("Run `python3 social_navigation_demo.py --headless` for a terminal-only simulation.")


if __name__ == "__main__":
    main()
