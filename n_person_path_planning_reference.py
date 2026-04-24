from __future__ import annotations

from dataclasses import dataclass, field

from social_navigation import PersonState, Pose2D, SocialNavParams, compute_social_plan, demo_people


@dataclass
class PlannerInput:
    robot: Pose2D
    people: list[PersonState]
    inviter_person_id: str | None = None
    params: SocialNavParams = field(default_factory=SocialNavParams)


@dataclass
class PlannerOutput:
    stage: str
    target_position: tuple[float, float]
    body_yaw: float
    gaze_yaw: float
    group_center: tuple[float, float]
    group_radius: float
    slot_gap_radians: float
    focus_person_id: str | None
    path_points: list[tuple[float, float]]
    candidates: list[dict[str, object]]


def candidate_score_formula(
    gap_score: float,
    comfort_score: float,
    clearance_score: float,
    frontality: float,
    affinity_score: float,
    travel_distance: float,
    radial_penalty: float,
    params: SocialNavParams,
) -> float:
    """
    Current N-person slot score used by the demo.

    score =
        gap_weight * gap_score
      + comfort_weight * comfort_score
      + clearance_weight * clearance_score
      + frontality_weight * frontality
      + affinity_weight * (2 * affinity_score - 1)
      - travel_weight * travel_distance
      - shortest_distance_weight * travel_distance
      - radial_weight * radial_penalty
    """
    return (
        params.gap_weight * gap_score
        + params.comfort_weight * comfort_score
        + params.clearance_weight * clearance_score
        + params.frontality_weight * frontality
        + params.affinity_weight * (2.0 * affinity_score - 1.0)
        - params.travel_weight * travel_distance
        - params.shortest_distance_weight * travel_distance
        - params.radial_weight * radial_penalty
    )


def compute_reference_plan(planner_input: PlannerInput) -> PlannerOutput:
    """
    Parameterized reference wrapper for the current N-person planner.

    Inputs:
    - robot: robot world pose
    - people: people world states
    - inviter_person_id: who said "Come play!"
    - params: all planning weights and safety distances

    Output:
    - best target slot
    - body/gaze orientation
    - planned path
    - candidate slots with scores and neighboring people
    """
    plan = compute_social_plan(
        planner_input.robot,
        planner_input.people,
        planner_input.params,
        inviter_person_id=planner_input.inviter_person_id,
    )
    return PlannerOutput(
        stage=plan.stage,
        target_position=plan.target_position,
        body_yaw=plan.body_yaw,
        gaze_yaw=plan.gaze_yaw,
        group_center=plan.group_center,
        group_radius=plan.group_radius,
        slot_gap_radians=plan.slot_gap_radians,
        focus_person_id=plan.focus_person_id,
        path_points=plan.path_points,
        candidates=[
            {
                "target": candidate.target,
                "score": candidate.score,
                "travel_distance": candidate.travel_distance,
                "min_distance_to_people": candidate.min_distance_to_people,
                "frontality": candidate.frontality,
                "affinity_score": candidate.affinity_score,
                "neighbor_ids": candidate.neighbor_ids,
            }
            for candidate in plan.candidates
        ],
    )


def example_input(group_size: int = 4, inviter_person_id: str | None = "P1") -> PlannerInput:
    return PlannerInput(
        robot=Pose2D(-2.8, -2.2, 0.61),
        people=demo_people(count=group_size, speaker_person_id=inviter_person_id or "P1"),
        inviter_person_id=inviter_person_id,
    )


if __name__ == "__main__":
    planner_input = example_input(group_size=4, inviter_person_id="P2")
    planner_output = compute_reference_plan(planner_input)
    print("stage =", planner_output.stage)
    print("target_position =", planner_output.target_position)
    print("group_center =", planner_output.group_center)
    print("focus_person_id =", planner_output.focus_person_id)
    print("candidate_count =", len(planner_output.candidates))
