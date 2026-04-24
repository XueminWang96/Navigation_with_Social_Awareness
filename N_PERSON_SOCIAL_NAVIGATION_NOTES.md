# N-Person Social Navigation

## 1. Inputs

### Robot state
- `robot_pose: Pose2D`
- `x, y`: robot world position
- `yaw`: robot body heading in world frame

### Human states
- `people: list[PersonState]`
- For each person:
- `person_id`
- `x, y`
- `yaw`
- `engagement`
- `speaking_score`
- `inclusion_weight`
- `affinity`

### Invitation state
- `inviter_person_id: str | None`
- Meaning: who explicitly said `"Come play!"`
- Effect: slot selection prefers a gap adjacent to this inviter

### Planner parameters
- `params: SocialNavParams`
- Defined in [social_navigation.py](/Users/demo/Documents/Social_Navigation/social_navigation.py#L148)

## 2. Main Parameters

### Social distance
- `personal_distance`: minimum comfortable clearance to each person
- `preferred_neighbor_distance`: preferred robot-to-neighbor distance when joining a group
- `ring_padding`: base offset outside the human ring
- `max_radial_extension`: maximum outward expansion allowed for a slot
- `settle_radius`: distance threshold for switching from `approach` to `engage`

### Slot scoring
- `gap_weight`: reward for larger angular gaps
- `comfort_weight`: reward for distance comfort around people
- `clearance_weight`: reward for keeping a safe minimum distance
- `frontality_weight`: reward for being in front-facing human regions
- `affinity_weight`: reward for affinity-favored positions
- `travel_weight`: motion cost
- `shortest_distance_weight`: extra explicit preference for shorter travel
- `radial_weight`: penalty for drifting too far away from the expected ring
- `inviter_neighbor_bonus`: extra bonus if the chosen gap is adjacent to the inviter

### Affinity shaping
- `affinity_spacing_gain`
- `affinity_focus_weight`
- `affinity_angular_bias_gain`
- `affinity_slot_margin`

### Path planning
- `path_grid_resolution`
- `path_margin`
- `path_person_weight`
- `path_pair_weight`
- `path_group_weight`
- `path_person_sigma_front`
- `path_person_sigma_side`
- `path_person_sigma_back`
- `path_bridge_sigma`
- `path_group_sigma`
- `path_block_cost`
- `path_visual_stride`

### Motion execution
- `body_turn_rate`
- `move_speed`
- `approach_center_blend`

## 3. Core Computation Logic

### Step 1. Estimate the interaction center
- Function: `estimate_group_center(...)`
- Current behavior: use the weighted centroid of human positions
- Important design choice: the center is position-driven and should not drift when only speaker orientation changes

### Step 2. Estimate the group radius
- Function: `group_radius(...)`
- Uses the median distance from people to the estimated center

### Step 3. Build candidate joining slots
- Function: `build_slot_candidates(...)`
- Process:
- Sort people by angle around the interaction center
- For each adjacent pair, treat the pair gap as a possible joining gap
- Compute the candidate slot angle inside that gap
- Compute the candidate radius just outside the human ring
- Form a target point in world coordinates

### Step 4. Score each candidate slot
- Candidate score combines:
- gap size
- interpersonal comfort
- clearance
- frontality
- affinity preference
- travel distance
- radial penalty

Formula reference:
- See `candidate_score_formula(...)` in [n_person_path_planning_reference.py](/Users/demo/Documents/Social_Navigation/n_person_path_planning_reference.py#L30)

### Step 5. Prefer inviter-adjacent gaps
- Function: `select_slot_candidate(...)`
- Logic:
- If there is an inviter, first look at candidate slots whose neighboring pair contains that inviter
- Among those, choose the highest-score candidate with inviter bonus
- If no such candidate exists, fall back to the globally best candidate

### Step 6. Plan a socially aware path
- Function: `plan_social_path(...)`
- Process:
- Build an interaction graph between humans
- Construct a cost field from:
- person cost
- pair bridge cost
- group center cost
- Run grid-based A* from the robot to the selected target slot
- Compress the path into a short waypoint sequence

### Step 7. Choose body yaw and gaze yaw
- `body_yaw`
- In `approach`: blend motion direction with group-center direction
- In `engage`: face the interaction center
- `gaze_yaw`
- Prefer active speakers
- Otherwise look toward the shared interaction center

### Step 8. Execute motion
- Function: `step_robot_toward_plan(...)`
- Behavior:
- Move toward the next path waypoint
- Clamp translation by `move_speed * dt`
- Clamp body rotation by `body_turn_rate * dt`

## 4. Invitation-Locked Behavior In The Demo

In the current web demo:
- Before invitation: robot remains in `wait_for_invite`
- When someone says `"Come play!"`: compute and lock one target slot
- After locking: speaker rotation changes only affect orientations and gaze
- The locked target position itself does not change

Related controller logic:
- [social_navigation_demo.py](/Users/demo/Documents/Social_Navigation/social_navigation_demo.py#L1950)

## 5. Small Mental Model

You can think of the planner as:

1. Find the group center and ring
2. Enumerate possible insertion gaps around the group
3. Pick the socially best gap
4. Bias toward the inviter's neighboring gap
5. Among good options, prefer the shortest travel
6. Plan a path that avoids cutting through the interaction
7. During approach, orient toward the group
8. During engagement, orient and gaze toward the active interaction

## 6. Reference Files

- Planner implementation:
- [social_navigation.py](/Users/demo/Documents/Social_Navigation/social_navigation.py)

- Demo controller and UI behavior:
- [social_navigation_demo.py](/Users/demo/Documents/Social_Navigation/social_navigation_demo.py)

- Parameterized wrapper:
- [n_person_path_planning_reference.py](/Users/demo/Documents/Social_Navigation/n_person_path_planning_reference.py)
