from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import tkinter as tk
except Exception:  # pragma: no cover - fallback for headless-only environments
    tk = None

from social_navigation import (
    DyadTriadParams,
    PersonState,
    Pose2D,
    SocialPlan,
    SocialNavParams,
    SocialNavigator,
    SocialTransitPlan,
    compute_dyad_to_triad_plan,
    compute_social_plan,
    compute_social_plan_with_locked_target,
    compute_social_transit_plan,
    demo_people,
    format_plan_summary,
    orient_people_toward_speaker,
    step_robot_toward_plan,
    step_robot_toward_transit_plan,
)


WORLD_HALF_EXTENT = 4.2
WORKSPACE_ROOT = Path(__file__).resolve().parent


@dataclass
class DemoState:
    robot: Pose2D
    people: list[PersonState]


def active_speaker_id(people: list[PersonState]) -> str | None:
    if not people:
        return None
    speakers = [person for person in people if person.speaking_score > 0.18]
    if not speakers:
        return None
    return max(
        speakers,
        key=lambda person: (person.speaking_score, person.engagement, person.person_id),
    ).person_id


def set_group_speaker(people: list[PersonState], speaker_person_id: str | None) -> str | None:
    valid_ids = {person.person_id for person in people}
    resolved_speaker_id = speaker_person_id if speaker_person_id in valid_ids else None
    for person in people:
        person.speaking_score = 1.0 if person.person_id == resolved_speaker_id else 0.0
    orient_people_toward_speaker(people, resolved_speaker_id)
    return resolved_speaker_id


class SocialNavigationDemo:
    def __init__(self, width: int = 980, height: int = 760) -> None:
        if tk is None:
            raise RuntimeError("tkinter is unavailable in this Python environment.")
        self.width = width
        self.height = height
        self.params = SocialNavParams()
        self.navigator = SocialNavigator(self.params)
        self.state = DemoState(
            robot=Pose2D(-2.8, -2.2, math.radians(35.0)),
            people=demo_people(),
        )
        self.last_time = time.monotonic()
        self.last_speaker_rotation = self.last_time
        self.drag_person_id: str | None = None
        self.mouse_world = (0.0, 0.0)

        self.root = tk.Tk()
        self.root.title("N-Person Social Navigation Demo")

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="#fbfaf6", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            bg="#f1ebdd",
            fg="#1f2529",
            padx=12,
            pady=10,
            font=("Menlo", 11),
        )
        self.status_label.pack(fill=tk.X)

        self._bind_events()
        self._update_loop()

    def run(self) -> None:
        self.root.mainloop()

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.root.bind("<space>", self._cycle_primary_speaker)
        self.root.bind("<r>", self._randomize_people)
        self.root.bind("<R>", self._randomize_people)
        self.root.bind("<n>", self._add_person)
        self.root.bind("<N>", self._add_person)
        self.root.bind("<Delete>", self._remove_nearest_person)
        self.root.bind("<BackSpace>", self._remove_nearest_person)

    def _world_to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        scale = min(self.width, self.height) / (2.0 * WORLD_HALF_EXTENT)
        x = self.width * 0.5 + point[0] * scale
        y = self.height * 0.5 - point[1] * scale
        return x, y

    def _canvas_to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        scale = min(self.width, self.height) / (2.0 * WORLD_HALF_EXTENT)
        x = (point[0] - self.width * 0.5) / scale
        y = -(point[1] - self.height * 0.5) / scale
        return x, y

    def _nearest_person(self, world_point: tuple[float, float], radius: float = 0.55) -> PersonState | None:
        if not self.state.people:
            return None
        nearest = min(self.state.people, key=lambda person: math.hypot(person.x - world_point[0], person.y - world_point[1]))
        if math.hypot(nearest.x - world_point[0], nearest.y - world_point[1]) <= radius:
            return nearest
        return None

    def _on_press(self, event: tk.Event[tk.Misc]) -> None:
        world = self._canvas_to_world((event.x, event.y))
        self.mouse_world = world
        nearest = self._nearest_person(world)
        self.drag_person_id = nearest.person_id if nearest else None

    def _on_drag(self, event: tk.Event[tk.Misc]) -> None:
        world = self._canvas_to_world((event.x, event.y))
        self.mouse_world = world
        if not self.drag_person_id:
            return
        person = next((candidate for candidate in self.state.people if candidate.person_id == self.drag_person_id), None)
        if person is None:
            return
        person.x, person.y = world
        orient_people_toward_speaker(self.state.people, active_speaker_id(self.state.people))

    def _on_release(self, _event: tk.Event[tk.Misc]) -> None:
        self.drag_person_id = None

    def _on_motion(self, event: tk.Event[tk.Misc]) -> None:
        self.mouse_world = self._canvas_to_world((event.x, event.y))

    def _cycle_primary_speaker(self, _event: tk.Event[tk.Misc]) -> None:
        if not self.state.people:
            return
        person_ids = [person.person_id for person in self.state.people]
        current_speaker_id = active_speaker_id(self.state.people)
        current_index = person_ids.index(current_speaker_id) if current_speaker_id in person_ids else -1
        next_index = (current_index + 1) % len(self.state.people)
        set_group_speaker(self.state.people, person_ids[next_index])

    def _randomize_people(self, _event: tk.Event[tk.Misc]) -> None:
        person_count = max(2, min(4, len(self.state.people)))
        radius = random.uniform(0.95, 1.55)
        center = (
            random.uniform(-0.3, 0.3),
            random.uniform(-0.15, 0.25),
        )
        self.state.people = []
        for index in range(person_count):
            angle = math.tau * index / person_count + random.uniform(-0.22, 0.22)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            yaw = math.atan2(center[1] - y, center[0] - x)
            speaking = 1.0 if index == 0 else (0.4 if index == 2 % person_count else 0.0)
            self.state.people.append(
                PersonState(
                    person_id=f"P{index + 1}",
                    x=x,
                    y=y,
                    yaw=yaw,
                    engagement=1.0,
                    speaking_score=speaking,
                    affinity=0.5,
                )
            )
        set_group_speaker(self.state.people, "P1")
        self.navigator.reset_attention()

    def _add_person(self, _event: tk.Event[tk.Misc]) -> None:
        new_index = len(self.state.people) + 1
        x, y = self.mouse_world
        center = compute_social_plan(self.state.robot, self.state.people, self.params).group_center
        yaw = math.atan2(center[1] - y, center[0] - x)
        self.state.people.append(
            PersonState(
                person_id=f"P{new_index}",
                x=x,
                y=y,
                yaw=yaw,
                engagement=1.0,
                speaking_score=0.0,
                affinity=0.5,
            )
        )
        orient_people_toward_speaker(self.state.people, active_speaker_id(self.state.people))
        self.navigator.reset_attention()

    def _remove_nearest_person(self, _event: tk.Event[tk.Misc]) -> None:
        if len(self.state.people) <= 1:
            return
        nearest = self._nearest_person(self.mouse_world, radius=1.0)
        if nearest is None:
            return
        self.state.people = [person for person in self.state.people if person.person_id != nearest.person_id]
        orient_people_toward_speaker(self.state.people, active_speaker_id(self.state.people))
        self.navigator.reset_attention()

    def _auto_rotate_speaker(self, now_s: float) -> None:
        if now_s - self.last_speaker_rotation < 3.6 or len(self.state.people) < 2:
            return
        self.last_speaker_rotation = now_s
        self._cycle_primary_speaker(None)

    def _draw_arrow(
        self,
        origin: tuple[float, float],
        angle: float,
        length: float,
        color: str,
        width: int = 3,
    ) -> None:
        ox, oy = self._world_to_canvas(origin)
        tip = (origin[0] + length * math.cos(angle), origin[1] + length * math.sin(angle))
        tx, ty = self._world_to_canvas(tip)
        self.canvas.create_line(ox, oy, tx, ty, fill=color, width=width, arrow=tk.LAST, smooth=True)

    def _draw_people(self, focus_person_id: str | None) -> None:
        for person in self.state.people:
            x, y = self._world_to_canvas(person.position)
            radius_px = 16
            fill = "#f0b74e" if person.person_id == focus_person_id else "#4fa1c7"
            outline = "#8f4900" if person.speaking_score > 0.7 else "#163444"
            self.canvas.create_oval(x - radius_px, y - radius_px, x + radius_px, y + radius_px, fill=fill, outline=outline, width=3)
            self.canvas.create_text(x, y - 24, text=person.person_id, fill="#1f2529", font=("Menlo", 11, "bold"))
            if person.yaw is not None:
                self._draw_arrow(person.position, person.yaw, 0.42, "#163444", width=2)
            if person.speaking_score > 0.1:
                halo = 24 + 10 * person.speaking_score
                self.canvas.create_oval(x - halo, y - halo, x + halo, y + halo, outline="#e7892d", width=2)

    def _draw_robot(self, plan) -> None:
        x, y = self._world_to_canvas(self.state.robot.position)
        size = 18
        self.canvas.create_oval(x - size, y - size, x + size, y + size, fill="#d5573b", outline="#6d1d10", width=3)
        self._draw_arrow(self.state.robot.position, self.state.robot.yaw, 0.54, "#6d1d10", width=3)
        self._draw_arrow(self.state.robot.position, plan.gaze_yaw, 0.84, "#d3a300", width=2)
        target = self._world_to_canvas(plan.target_position)
        self.canvas.create_oval(target[0] - 8, target[1] - 8, target[0] + 8, target[1] + 8, outline="#d5573b", width=3)
        self.canvas.create_line(x, y, target[0], target[1], fill="#d5573b", dash=(6, 6), width=2)

    def _draw_center_and_slots(self, plan) -> None:
        center = self._world_to_canvas(plan.group_center)
        ring_scale = min(self.width, self.height) / (2.0 * WORLD_HALF_EXTENT)
        self.canvas.create_oval(center[0] - 6, center[1] - 6, center[0] + 6, center[1] + 6, fill="#1f2529", outline="")
        self.canvas.create_oval(
            center[0] - plan.group_radius * ring_scale,
            center[1] - plan.group_radius * ring_scale,
            center[0] + plan.group_radius * ring_scale,
            center[1] + plan.group_radius * ring_scale,
            outline="#bcb39f",
            width=2,
            dash=(4, 6),
        )
        for candidate in plan.candidates[:4]:
            slot = self._world_to_canvas(candidate.target)
            color = "#5d7e72" if candidate == plan.candidates[0] else "#c3b9a1"
            self.canvas.create_oval(slot[0] - 6, slot[1] - 6, slot[0] + 6, slot[1] + 6, outline=color, width=2)

    def _draw_ground(self) -> None:
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill="#fbfaf6", outline="")
        for world_line in range(-4, 5):
            start = self._world_to_canvas((-WORLD_HALF_EXTENT, float(world_line)))
            end = self._world_to_canvas((WORLD_HALF_EXTENT, float(world_line)))
            self.canvas.create_line(start[0], start[1], end[0], end[1], fill="#ece5d8")
            start = self._world_to_canvas((float(world_line), -WORLD_HALF_EXTENT))
            end = self._world_to_canvas((float(world_line), WORLD_HALF_EXTENT))
            self.canvas.create_line(start[0], start[1], end[0], end[1], fill="#ece5d8")

    def _update_status(self, plan) -> None:
        self.status_var.set(
            "\n".join(
                [
                    "Drag people to reshape the group. Press N to add, Delete to remove, R to randomize, Space to rotate speakers.",
                    format_plan_summary(plan),
                    f"slot_gap={math.degrees(plan.slot_gap_radians):.1f}deg, center=({plan.group_center[0]:.2f}, {plan.group_center[1]:.2f}), focused={plan.focus_person_id or 'group'}",
                ]
            )
        )

    def _update_loop(self) -> None:
        now_s = time.monotonic()
        dt_s = min(0.05, now_s - self.last_time)
        self.last_time = now_s

        self._auto_rotate_speaker(now_s)
        plan = self.navigator.update(self.state.robot, self.state.people, now_s)
        self.state.robot = step_robot_toward_plan(self.state.robot, plan, dt_s, self.params)

        self.canvas.delete("all")
        self._draw_ground()
        self._draw_center_and_slots(plan)
        self._draw_people(plan.focus_person_id)
        self._draw_robot(plan)
        self._update_status(plan)

        self.root.after(33, self._update_loop)


def run_headless(steps: int = 30, dt_s: float = 0.2) -> None:
    params = SocialNavParams()
    navigator = SocialNavigator(params)
    robot = Pose2D(-2.8, -2.2, math.radians(35.0))
    people = demo_people()

    for step in range(steps):
        now_s = step * dt_s
        plan = navigator.update(robot, people, now_s)
        print(f"step={step:02d} {format_plan_summary(plan)}")
        robot = step_robot_toward_plan(robot, plan, dt_s, params)


def format_dyad_triad_summary(plan) -> str:
    focus_label = plan.focus_person_id or "shared"
    return (
        f"stage={plan.stage}, "
        f"joining={plan.joining_score:.2f}, "
        f"target=({plan.target_position[0]:.2f}, {plan.target_position[1]:.2f}), "
        f"focus={focus_label}"
    )


def step_pose_toward_target(
    pose: Pose2D,
    target_position: tuple[float, float],
    target_yaw: float,
    dt_s: float,
    move_speed: float = 0.75,
    turn_rate: float = 1.8,
) -> Pose2D:
    dx = target_position[0] - pose.x
    dy = target_position[1] - pose.y
    distance_to_target = math.hypot(dx, dy)
    if distance_to_target > 1e-6:
        move_distance = min(distance_to_target, move_speed * dt_s)
        x = pose.x + dx * move_distance / distance_to_target
        y = pose.y + dy * move_distance / distance_to_target
    else:
        x = pose.x
        y = pose.y

    yaw_delta = target_yaw - pose.yaw
    while yaw_delta <= -math.pi:
        yaw_delta += math.tau
    while yaw_delta > math.pi:
        yaw_delta -= math.tau
    max_turn = turn_rate * dt_s
    yaw_delta = max(-max_turn, min(max_turn, yaw_delta))
    yaw = pose.yaw + yaw_delta
    while yaw <= -math.pi:
        yaw += math.tau
    while yaw > math.pi:
        yaw -= math.tau
    return Pose2D(x, y, yaw)


def format_dyad_event_summary(event_phase: str, plan) -> str:
    if event_phase == "p2_approaching":
        return (
            f"phase=p2_approaching, "
            f"stage={plan.stage}, "
            f"joining={plan.joining_score:.2f}"
        )
    if event_phase == "reconfiguring":
        return (
            f"phase=reconfiguring, "
            f"stage={plan.stage}, "
            f"target=({plan.target_position[0]:.2f}, {plan.target_position[1]:.2f})"
        )
    if event_phase == "triad_formed":
        return (
            f"phase=triad_formed, "
            f"stage={plan.stage}, "
            f"focus={plan.focus_person_id or 'shared'}"
        )
    return (
        f"phase=dyad, "
        f"stage={plan.stage}, "
        f"joining={plan.joining_score:.2f}"
    )


def format_detour_summary(scenario_mode: str, plan: SocialTransitPlan) -> str:
    scenario_label = "single_person" if scenario_mode == "single_person" else "interactive_group"
    return (
        f"scenario={scenario_label}, "
        f"stage={plan.stage}, "
        f"remaining={plan.distance_to_target:.2f}m, "
        f"extra={plan.extra_distance:.2f}m, "
        f"clearance={plan.min_clearance:.2f}m"
    )


WEB_APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Social Navigation Web Demo</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f7f4ec;
      --panel: rgba(255, 252, 246, 0.92);
      --ink: #1f2529;
      --muted: #5f6b70;
      --grid: #e8dfcf;
      --accent: #d5573b;
      --accent-2: #4fa1c7;
      --accent-3: #e3a12d;
      --ring: #bcb39f;
      --slot: #5d7e72;
      --slot-alt: #cabda5;
      --shadow: 0 24px 60px rgba(44, 32, 18, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top, rgba(255, 255, 255, 0.75), transparent 36%),
        linear-gradient(180deg, #f8f2e5 0%, #efe7d8 100%);
    }

    .shell {
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      gap: 18px;
    }

    .hero,
    .panel {
      background: var(--panel);
      border: 1px solid rgba(110, 91, 62, 0.12);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .hero {
      padding: 20px 22px;
      display: grid;
      gap: 12px;
    }

    .hero h1 {
      margin: 0;
      font-size: clamp(1.4rem, 2.2vw, 2rem);
      font-weight: 700;
      letter-spacing: 0.01em;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      font-weight: 600;
      color: white;
      background: linear-gradient(135deg, #d5573b, #c24128);
      cursor: pointer;
      box-shadow: 0 10px 22px rgba(213, 87, 59, 0.22);
    }

    button.secondary {
      background: linear-gradient(135deg, #5c7d71, #456156);
      box-shadow: 0 10px 22px rgba(70, 97, 86, 0.18);
    }

    button.active {
      box-shadow:
        inset 0 0 0 2px rgba(255, 255, 255, 0.72),
        0 10px 22px rgba(213, 87, 59, 0.22);
    }

    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      box-shadow: none;
    }

    button.ghost {
      background: white;
      color: var(--ink);
      box-shadow: none;
      border: 1px solid rgba(31, 37, 41, 0.12);
    }

    .nav-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 10px 16px;
      font-weight: 600;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid rgba(31, 37, 41, 0.12);
      text-decoration: none;
    }

    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(31, 37, 41, 0.08);
      color: var(--ink);
      font-size: 0.95rem;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr);
      gap: 18px;
    }

    .panel {
      padding: 16px;
    }

    .canvas-wrap {
      position: relative;
      overflow: hidden;
      min-height: 560px;
    }

    canvas {
      width: 100%;
      height: min(70vh, 760px);
      display: block;
      border-radius: 18px;
      background: var(--paper);
      touch-action: none;
    }

    .legend,
    .metrics {
      display: grid;
      gap: 12px;
    }

    .affinity-controls {
      display: grid;
      gap: 10px;
    }

    .affinity-row {
      display: grid;
      grid-template-columns: 48px minmax(0, 1fr) 52px;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(31, 37, 41, 0.08);
    }

    .affinity-id {
      font-weight: 700;
    }

    .affinity-row input[type="range"] {
      width: 100%;
      accent-color: #d5573b;
    }

    .affinity-value {
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }

    .legend h2,
    .metrics h2 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(31, 37, 41, 0.08);
    }

    .metric .label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 0.85rem;
    }

    .metric .value {
      display: block;
      font-size: 1rem;
      font-weight: 700;
    }

    .hint {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }

    @media (max-width: 900px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>N-Person Social Navigation</h1>
      <p>Drag people to reshape the group, double-click to add a person, and Alt-click a person to remove them. You can switch between 2-, 3- and 4-person groups, let any visible person say "Come play!", and watch the robot approach the inviter's side using the shortest socially valid slot.</p>
      <div class="toolbar">
        <button id="toggle-run" type="button">Pause</button>
        <button id="rotate-speaker" class="secondary" type="button">Rotate Speaker</button>
        <button id="randomize" class="secondary" type="button">Randomize Group</button>
        <button id="reset" class="ghost" type="button">Reset Scene</button>
        <a class="nav-link" href="/detour">Detour To Dock</a>
        <a class="nav-link" href="/dyad-triad">Dyad -> Triad Page</a>
      </div>
      <div class="toolbar">
        <button id="group-2" class="secondary" data-group-size="2" type="button">2-Person</button>
        <button id="group-3" class="secondary" data-group-size="3" type="button">3-Person</button>
        <button id="group-4" class="secondary" data-group-size="4" type="button">4-Person</button>
        <button id="invite-p1" class="secondary" data-invite-person="P1" type="button">P1 Says "Come play!"</button>
        <button id="invite-p2" class="secondary" data-invite-person="P2" type="button">P2 Says "Come play!"</button>
        <button id="invite-p3" class="secondary" data-invite-person="P3" type="button">P3 Says "Come play!"</button>
        <button id="invite-p4" class="secondary" data-invite-person="P4" type="button">P4 Says "Come play!"</button>
      </div>
      <div class="status-row">
        <div class="pill" id="connection-pill">Connecting...</div>
        <div class="pill" id="invitation-pill">Waiting for invitation...</div>
        <div class="pill" id="summary-pill">Waiting for state...</div>
      </div>
    </section>

    <div class="layout">
      <section class="panel canvas-wrap">
        <canvas id="viz" width="980" height="760"></canvas>
      </section>

      <aside class="panel">
        <div class="metrics">
          <h2>Live Metrics</h2>
          <div class="metric-grid">
            <div class="metric"><span class="label">Stage</span><span class="value" id="stage-value">-</span></div>
            <div class="metric"><span class="label">Focus</span><span class="value" id="focus-value">-</span></div>
            <div class="metric"><span class="label">Target</span><span class="value" id="target-value">-</span></div>
            <div class="metric"><span class="label">Body Yaw</span><span class="value" id="body-value">-</span></div>
            <div class="metric"><span class="label">Gaze Yaw</span><span class="value" id="gaze-value">-</span></div>
            <div class="metric"><span class="label">Slot Gap</span><span class="value" id="gap-value">-</span></div>
            <div class="metric"><span class="label">Target Affinity</span><span class="value" id="affinity-value">-</span></div>
            <div class="metric"><span class="label">Group Size</span><span class="value" id="group-value">-</span></div>
          </div>
          <h2>Interaction</h2>
          <p class="hint">The browser view polls the demo service for the current robot pose, selected approach slot, group center, attention target, and affinity-aware score in real time.</p>
          <p class="hint">Use the canvas to inspect the dynamics: blue circles are people, orange halo marks active speakers, and the red robot waits outside until one of the visible people says "Come play!". After that, it prefers the shortest socially valid slot next to that inviter rather than cutting through the middle of the group.</p>
          <h2>Affinity</h2>
          <p class="hint">`0.50` is neutral. Higher values mean the robot is more comfortable orienting toward and standing nearer that person; lower values make it keep a little more distance.</p>
          <div class="affinity-controls" id="affinity-controls"></div>
        </div>
      </aside>
    </div>
  </div>

  <script>
    const WORLD_HALF_EXTENT = 4.2;
    const canvas = document.getElementById("viz");
    const ctx = canvas.getContext("2d");
    const connectionPill = document.getElementById("connection-pill");
    const invitationPill = document.getElementById("invitation-pill");
    const summaryPill = document.getElementById("summary-pill");
    const stageValue = document.getElementById("stage-value");
    const focusValue = document.getElementById("focus-value");
    const targetValue = document.getElementById("target-value");
    const bodyValue = document.getElementById("body-value");
    const gazeValue = document.getElementById("gaze-value");
    const gapValue = document.getElementById("gap-value");
    const affinityValue = document.getElementById("affinity-value");
    const groupValue = document.getElementById("group-value");
    const affinityControls = document.getElementById("affinity-controls");
    const toggleRunButton = document.getElementById("toggle-run");
    const inviteButtons = Array.from(document.querySelectorAll("[data-invite-person]"));
    const groupButtons = Array.from(document.querySelectorAll("[data-group-size]"));

    let state = null;
    let dragPersonId = null;
    let lastDragSentAt = 0;
    let lastPointerWorld = {x: 0, y: 0};
    let affinityControlSignature = "";
    const affinityUpdateTimers = new Map();

    function resizeCanvas() {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width * ratio));
      const height = Math.max(260, Math.round(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function worldToCanvas(point) {
      const scale = Math.min(canvas.width, canvas.height) / (2 * WORLD_HALF_EXTENT);
      return {
        x: canvas.width * 0.5 + point.x * scale,
        y: canvas.height * 0.5 - point.y * scale,
      };
    }

    function canvasToWorld(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const x = (clientX - rect.left) * ratio;
      const y = (clientY - rect.top) * ratio;
      const scale = Math.min(canvas.width, canvas.height) / (2 * WORLD_HALF_EXTENT);
      return {
        x: (x - canvas.width * 0.5) / scale,
        y: -(y - canvas.height * 0.5) / scale,
      };
    }

    function angleTip(origin, angle, length) {
      return {
        x: origin.x + Math.cos(angle) * length,
        y: origin.y + Math.sin(angle) * length,
      };
    }

    function roundValue(value) {
      return Number(value).toFixed(2);
    }

    function deg(value) {
      return `${(value * 180 / Math.PI).toFixed(1)} deg`;
    }

    function drawArrow(origin, angle, length, color, width) {
      const start = worldToCanvas(origin);
      const tip = worldToCanvas(angleTip(origin, angle, length));
      const headLength = 12 * (window.devicePixelRatio || 1);

      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(tip.x, tip.y);
      ctx.stroke();

      const theta = Math.atan2(tip.y - start.y, tip.x - start.x);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(tip.x - headLength * Math.cos(theta - Math.PI / 7), tip.y - headLength * Math.sin(theta - Math.PI / 7));
      ctx.lineTo(tip.x - headLength * Math.cos(theta + Math.PI / 7), tip.y - headLength * Math.sin(theta + Math.PI / 7));
      ctx.closePath();
      ctx.fill();
    }

    function drawGrid() {
      ctx.fillStyle = "#fbfaf6";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#ece5d8";
      ctx.lineWidth = 1;
      for (let world = -4; world <= 4; world += 1) {
        let a = worldToCanvas({x: -WORLD_HALF_EXTENT, y: world});
        let b = worldToCanvas({x: WORLD_HALF_EXTENT, y: world});
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();

        a = worldToCanvas({x: world, y: -WORLD_HALF_EXTENT});
        b = worldToCanvas({x: world, y: WORLD_HALF_EXTENT});
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    function drawFieldHeatmap(scale) {
      if (!state.plan.field_samples || state.plan.field_samples.length === 0) {
        return;
      }
      const blockCost = Math.max(2.0, state.params?.path_block_cost || 10.0);
      const cellSize = Math.max(8, scale * (state.params?.path_grid_resolution || 0.18) * 2.0);
      for (const sample of state.plan.field_samples) {
        const normalized = Math.max(0, Math.min(1, (sample.cost - 1.0) / (blockCost - 1.0)));
        if (normalized < 0.03) {
          continue;
        }
        const point = worldToCanvas(sample);
        ctx.fillStyle = `rgba(160, 95, 34, ${0.03 + 0.22 * normalized})`;
        ctx.fillRect(point.x - cellSize * 0.5, point.y - cellSize * 0.5, cellSize, cellSize);
      }
    }

    function drawScene() {
      resizeCanvas();
      drawGrid();
      if (!state) {
        return;
      }

      const scale = Math.min(canvas.width, canvas.height) / (2 * WORLD_HALF_EXTENT);
      const center = worldToCanvas(state.plan.group_center);
      drawFieldHeatmap(scale);

      ctx.setLineDash([10, 12]);
      ctx.strokeStyle = "#bcb39f";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(center.x, center.y, state.plan.group_radius * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#1f2529";
      ctx.beginPath();
      ctx.arc(center.x, center.y, 6, 0, Math.PI * 2);
      ctx.fill();

      for (const candidate of state.plan.candidates) {
        const slot = worldToCanvas(candidate.target);
        ctx.strokeStyle = candidate.is_best ? "#5d7e72" : "#c3b9a1";
        ctx.lineWidth = candidate.is_best ? 3 : 2;
        ctx.beginPath();
        ctx.arc(slot.x, slot.y, candidate.is_best ? 7 : 5, 0, Math.PI * 2);
        ctx.stroke();
      }

      const robot = state.robot;
      const target = worldToCanvas(state.plan.target_position);
      const robotCanvas = worldToCanvas(robot.position);
      const peopleById = Object.fromEntries(state.people.map((person) => [person.person_id, person]));

      for (const link of state.plan.interaction_links || []) {
        const personA = peopleById[link.a];
        const personB = peopleById[link.b];
        if (!personA || !personB) {
          continue;
        }
        const a = worldToCanvas(personA.position);
        const b = worldToCanvas(personB.position);
        ctx.strokeStyle = `rgba(121, 104, 74, ${0.12 + 0.35 * link.probability})`;
        ctx.lineWidth = 1.5 + 4.0 * link.probability;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }

      if (state.plan.path_points && state.plan.path_points.length > 1) {
        ctx.strokeStyle = "rgba(213, 87, 59, 0.92)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        const first = worldToCanvas(state.plan.path_points[0]);
        ctx.moveTo(first.x, first.y);
        for (const waypoint of state.plan.path_points.slice(1)) {
          const point = worldToCanvas(waypoint);
          ctx.lineTo(point.x, point.y);
        }
        ctx.stroke();
      } else {
        ctx.strokeStyle = "#d5573b";
        ctx.lineWidth = 2;
        ctx.setLineDash([12, 10]);
        ctx.beginPath();
        ctx.moveTo(robotCanvas.x, robotCanvas.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.strokeStyle = "#d5573b";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(target.x, target.y, 8, 0, Math.PI * 2);
      ctx.stroke();

      for (const person of state.people) {
        const point = worldToCanvas(person.position);
        if (person.speaking_score > 0.1) {
          ctx.strokeStyle = "#e7892d";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(point.x, point.y, 24 + 10 * person.speaking_score, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.fillStyle = person.person_id === state.plan.focus_person_id ? "#f0b74e" : "#4fa1c7";
        ctx.strokeStyle = person.speaking_score > 0.7 ? "#8f4900" : "#163444";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 16, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        if (person.yaw !== null) {
          drawArrow(person.position, person.yaw, 0.42, "#163444", 2);
        }

        ctx.fillStyle = "#1f2529";
        ctx.font = `${12 * (window.devicePixelRatio || 1)}px Menlo, monospace`;
        ctx.textAlign = "center";
        ctx.fillText(person.person_id, point.x, point.y - 22);
        ctx.fillStyle = "#5f6b70";
        ctx.font = `${10 * (window.devicePixelRatio || 1)}px Menlo, monospace`;
        ctx.fillText(`a=${roundValue(person.affinity)}`, point.x, point.y + 31);
      }

      ctx.fillStyle = "#d5573b";
      ctx.strokeStyle = "#6d1d10";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(robotCanvas.x, robotCanvas.y, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      drawArrow(robot.position, robot.yaw, 0.54, "#6d1d10", 3);
      drawArrow(robot.position, state.plan.gaze_yaw, 0.84, "#d3a300", 2);
    }

    function updateMetrics() {
      if (!state) {
        return;
      }
      stageValue.textContent = state.plan.stage;
      focusValue.textContent = state.plan.focus_person_id || "group";
      targetValue.textContent = `(${roundValue(state.plan.target_position.x)}, ${roundValue(state.plan.target_position.y)})`;
      bodyValue.textContent = deg(state.plan.body_yaw);
      gazeValue.textContent = deg(state.plan.gaze_yaw);
      gapValue.textContent = `${(state.plan.slot_gap_radians * 180 / Math.PI).toFixed(1)} deg`;
      affinityValue.textContent = roundValue(state.plan.target_affinity);
      groupValue.textContent = `${state.group_size}`;
      toggleRunButton.textContent = state.running ? "Pause" : "Resume";
      summaryPill.textContent = state.summary;
      if (state.invitation_active) {
        invitationPill.textContent = `${state.inviter_person_id || "P1"} said "Come play!"`;
      } else {
        invitationPill.textContent = `Waiting for a participant to say "Come play!"`;
      }
      const visiblePeople = new Set(state.people.map((person) => person.person_id));
      inviteButtons.forEach((button) => {
        const personId = button.dataset.invitePerson;
        button.disabled = !visiblePeople.has(personId);
        button.classList.toggle("active", state.inviter_person_id === personId);
      });
      groupButtons.forEach((button) => {
        button.classList.toggle("active", Number(button.dataset.groupSize) === Number(state.group_size));
      });
      connectionPill.textContent = connectionLabel;
      syncAffinityControls();
    }

    const isLocalDemo = ["127.0.0.1", "localhost"].includes(window.location.hostname);
    const connectionLabel = isLocalDemo
      ? `Connected to local demo on port ${window.location.port || "80"}`
      : `Connected to online demo at ${window.location.host}`;
    const pollIntervalMs = isLocalDemo ? 120 : 350;
    let pollTimer = null;
    let stateFetchPromise = null;
    let consecutiveFetchFailures = 0;

    function scheduleNextPoll(delayMs = pollIntervalMs) {
      if (pollTimer !== null) {
        window.clearTimeout(pollTimer);
      }
      pollTimer = window.setTimeout(() => {
        void fetchState();
      }, delayMs);
    }

    function queueAffinityUpdate(personId, affinity) {
      const previousTimer = affinityUpdateTimers.get(personId);
      if (previousTimer) {
        clearTimeout(previousTimer);
      }
      const timer = setTimeout(() => {
        affinityUpdateTimers.delete(personId);
        postCommand({
          action: "set_affinity",
          person_id: personId,
          affinity: affinity,
        });
      }, 70);
      affinityUpdateTimers.set(personId, timer);
    }

    function syncAffinityControls() {
      if (!state) {
        return;
      }

      const signature = state.people.map((person) => person.person_id).join("|");
      if (signature !== affinityControlSignature) {
        affinityControls.innerHTML = "";
        affinityControlSignature = signature;
        for (const person of state.people) {
          const row = document.createElement("label");
          row.className = "affinity-row";

          const id = document.createElement("span");
          id.className = "affinity-id";
          id.textContent = person.person_id;

          const slider = document.createElement("input");
          slider.type = "range";
          slider.min = "0";
          slider.max = "1";
          slider.step = "0.01";
          slider.dataset.personId = person.person_id;
          slider.id = `affinity-${person.person_id}`;

          const value = document.createElement("span");
          value.className = "affinity-value";
          value.id = `affinity-value-${person.person_id}`;

          slider.addEventListener("input", (event) => {
            const input = event.target;
            const nextAffinity = Number(input.value);
            value.textContent = nextAffinity.toFixed(2);
            queueAffinityUpdate(person.person_id, nextAffinity);
          });

          row.appendChild(id);
          row.appendChild(slider);
          row.appendChild(value);
          affinityControls.appendChild(row);
        }
      }

      for (const person of state.people) {
        const slider = document.getElementById(`affinity-${person.person_id}`);
        const value = document.getElementById(`affinity-value-${person.person_id}`);
        if (!slider || !value) {
          continue;
        }
        if (document.activeElement !== slider) {
          slider.value = String(person.affinity);
        }
        value.textContent = roundValue(person.affinity);
      }
    }

    async function fetchState({awaitInFlight = false} = {}) {
      if (stateFetchPromise) {
        return awaitInFlight ? stateFetchPromise : undefined;
      }
      stateFetchPromise = (async () => {
        try {
          const response = await fetch("/state", {
            cache: "no-store",
            headers: {"Accept": "application/json"},
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          state = await response.json();
          consecutiveFetchFailures = 0;
          updateMetrics();
        } catch (error) {
          consecutiveFetchFailures += 1;
          if (consecutiveFetchFailures >= 2) {
            const message = error instanceof Error ? error.message : String(error);
            connectionPill.textContent = `Reconnecting to demo... (${message})`;
          }
        } finally {
          stateFetchPromise = null;
          scheduleNextPoll();
        }
      })();
      return stateFetchPromise;
    }

    async function postCommand(payload) {
      try {
        const response = await fetch("/command", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        await fetchState({awaitInFlight: true});
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        connectionPill.textContent = `Command failed: ${message}`;
      }
    }

    function nearestPerson(worldPoint, radius = 0.55) {
      if (!state) {
        return null;
      }
      let nearest = null;
      let nearestDistance = Infinity;
      for (const person of state.people) {
        const dx = person.position.x - worldPoint.x;
        const dy = person.position.y - worldPoint.y;
        const distance = Math.hypot(dx, dy);
        if (distance < nearestDistance) {
          nearest = person;
          nearestDistance = distance;
        }
      }
      return nearestDistance <= radius ? nearest : null;
    }

    canvas.addEventListener("pointerdown", async (event) => {
      lastPointerWorld = canvasToWorld(event.clientX, event.clientY);
      const nearest = nearestPerson(lastPointerWorld);
      if (event.altKey && nearest) {
        await postCommand({action: "remove_person", person_id: nearest.person_id});
        return;
      }
      if (nearest) {
        dragPersonId = nearest.person_id;
        canvas.setPointerCapture(event.pointerId);
      }
    });

    canvas.addEventListener("pointermove", async (event) => {
      if (!dragPersonId) {
        return;
      }
      lastPointerWorld = canvasToWorld(event.clientX, event.clientY);
      const now = performance.now();
      if (now - lastDragSentAt < 33) {
        return;
      }
      lastDragSentAt = now;
      await postCommand({
        action: "move_person",
        person_id: dragPersonId,
        x: lastPointerWorld.x,
        y: lastPointerWorld.y,
      });
    });

    canvas.addEventListener("pointerup", async (event) => {
      if (dragPersonId) {
        lastPointerWorld = canvasToWorld(event.clientX, event.clientY);
        await postCommand({
          action: "move_person",
          person_id: dragPersonId,
          x: lastPointerWorld.x,
          y: lastPointerWorld.y,
        });
      }
      dragPersonId = null;
      try {
        canvas.releasePointerCapture(event.pointerId);
      } catch (_error) {
      }
    });

    canvas.addEventListener("dblclick", async (event) => {
      const world = canvasToWorld(event.clientX, event.clientY);
      await postCommand({action: "add_person", x: world.x, y: world.y});
    });

    document.getElementById("toggle-run").addEventListener("click", () => postCommand({action: "toggle_running"}));
    document.getElementById("rotate-speaker").addEventListener("click", () => postCommand({action: "cycle_speaker"}));
    document.getElementById("randomize").addEventListener("click", () => postCommand({action: "randomize"}));
    document.getElementById("reset").addEventListener("click", () => postCommand({action: "reset"}));
    inviteButtons.forEach((button) => {
      button.addEventListener("click", () => postCommand({action: "invite_robot", person_id: button.dataset.invitePerson}));
    });
    groupButtons.forEach((button) => {
      button.addEventListener("click", () => postCommand({action: "set_group_size", count: Number(button.dataset.groupSize)}));
    });

    function renderLoop() {
      drawScene();
      requestAnimationFrame(renderLoop);
    }

    window.addEventListener("resize", drawScene);
    scheduleNextPoll(0);
    renderLoop();
  </script>
</body>
</html>
"""


DYAD_TRIAD_WEB_APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dyad to Triad Reconfiguration Demo</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f5f1e8;
      --panel: rgba(255, 251, 245, 0.94);
      --ink: #20252a;
      --muted: #5f6a71;
      --grid: #e4dccd;
      --robot: #d5573b;
      --primary: #4f8aa3;
      --newcomer: #7b9d62;
      --focus: #efb34d;
      --shadow: 0 24px 56px rgba(51, 39, 24, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 255, 255, 0.72), transparent 34%),
        linear-gradient(180deg, #f7efe0 0%, #efe5d4 100%);
    }

    .shell {
      width: min(1240px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      gap: 18px;
    }

    .hero,
    .panel {
      background: var(--panel);
      border: 1px solid rgba(85, 65, 40, 0.12);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .hero {
      padding: 20px 22px;
      display: grid;
      gap: 12px;
    }

    .hero h1 {
      margin: 0;
      font-size: clamp(1.45rem, 2.2vw, 2rem);
    }

    .hero p,
    .hint {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    button,
    .nav-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
    }

    button {
      color: white;
      background: linear-gradient(135deg, #d5573b, #c24128);
      box-shadow: 0 10px 22px rgba(213, 87, 59, 0.22);
    }

    button.secondary {
      background: linear-gradient(135deg, #5f7d67, #4b6350);
      box-shadow: 0 10px 22px rgba(75, 99, 80, 0.18);
    }

    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      box-shadow: none;
    }

    button.ghost,
    .nav-link {
      color: var(--ink);
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid rgba(31, 37, 41, 0.12);
      box-shadow: none;
    }

    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(31, 37, 41, 0.08);
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.85fr);
      gap: 18px;
    }

    .panel {
      padding: 16px;
    }

    canvas {
      width: 100%;
      height: min(72vh, 760px);
      display: block;
      border-radius: 18px;
      background: var(--paper);
      touch-action: none;
    }

    .metrics,
    .controls {
      display: grid;
      gap: 12px;
    }

    h2 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric,
    .control-card {
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(31, 37, 41, 0.08);
    }

    .metric .label,
    .control-card .label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 0.85rem;
    }

    .metric .value {
      display: block;
      font-size: 1rem;
      font-weight: 700;
    }

    .control-card {
      display: grid;
      gap: 10px;
    }

    .row-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-weight: 700;
    }

    .chip {
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      color: white;
    }

    .chip.primary {
      background: var(--primary);
    }

    .chip.newcomer {
      background: var(--newcomer);
    }

    .slider-row {
      display: grid;
      grid-template-columns: 64px minmax(0, 1fr) 56px;
      gap: 10px;
      align-items: center;
    }

    .slider-row input[type="range"] {
      width: 100%;
      accent-color: #d5573b;
    }

    .slider-value {
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }

    @media (max-width: 940px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Dyad -> Triad Reconfiguration</h1>
      <p>This page simulates an event-triggered dyad-to-triad transition: the robot is already interacting with P1, then P2 approaches, stops beside the dyad, and only after that the robot begins a small reconfiguration toward the nearest suitable triadic formation.</p>
      <div class="toolbar">
        <button id="toggle-run" type="button">Pause</button>
        <button id="p2-approach" class="secondary" type="button">P2 Approach</button>
        <button id="primary-speaks" class="secondary" type="button">P1 Speaks</button>
        <button id="newcomer-speaks" class="secondary" type="button">P2 Speaks</button>
        <button id="clear-speaker" class="ghost" type="button">Clear Speaker</button>
        <button id="reset" class="ghost" type="button">Reset Scene</button>
        <a class="nav-link" href="/detour">Detour To Dock</a>
        <a class="nav-link" href="/">Back To Group Demo</a>
      </div>
      <div class="status-row">
        <div class="pill" id="connection-pill">Connecting...</div>
        <div class="pill" id="event-pill">Event: dyad</div>
        <div class="pill" id="summary-pill">Waiting for state...</div>
      </div>
    </section>

    <div class="layout">
      <section class="panel">
        <canvas id="viz" width="980" height="760"></canvas>
      </section>

      <aside class="panel">
        <div class="metrics">
          <h2>Plan State</h2>
          <div class="metric-grid">
            <div class="metric"><span class="label">Event</span><span class="value" id="event-value">-</span></div>
            <div class="metric"><span class="label">Stage</span><span class="value" id="stage-value">-</span></div>
            <div class="metric"><span class="label">Focus</span><span class="value" id="focus-value">-</span></div>
            <div class="metric"><span class="label">Joining Score</span><span class="value" id="joining-value">-</span></div>
            <div class="metric"><span class="label">Admitted</span><span class="value" id="admitted-value">-</span></div>
            <div class="metric"><span class="label">Target</span><span class="value" id="target-value">-</span></div>
            <div class="metric"><span class="label">Shared Center</span><span class="value" id="shared-value">-</span></div>
            <div class="metric"><span class="label">Dist To Primary</span><span class="value" id="primary-dist-value">-</span></div>
            <div class="metric"><span class="label">Dist To Newcomer</span><span class="value" id="newcomer-dist-value">-</span></div>
            <div class="metric"><span class="label">Same Side</span><span class="value" id="same-side-value">-</span></div>
            <div class="metric"><span class="label">Motion Limited</span><span class="value" id="motion-limit-value">-</span></div>
          </div>

          <h2>Interpretation</h2>
          <p class="hint">Use `P2 Approach` to trigger the scenario. P2 walks toward the robot-P1 dyad and stops at a staging point beside the interaction. Until P2 arrives, the robot holds the dyad.</p>
          <p class="hint">Once P2 is close enough to count as joining, the page switches into `Reconfiguration`, and the robot moves to the nearest socially suitable triadic position instead of making a large relocation.</p>

          <h2>Actors</h2>
          <div class="controls">
            <div class="control-card">
              <div class="row-title"><span>Primary</span><span class="chip primary">P1</span></div>
              <div class="slider-row">
                <span class="label">Affinity</span>
                <input id="primary-affinity" type="range" min="0" max="1" step="0.01">
                <span class="slider-value" id="primary-affinity-value">0.50</span>
              </div>
              <div class="slider-row">
                <span class="label">Yaw</span>
                <input id="primary-yaw" type="range" min="-180" max="180" step="1">
                <span class="slider-value" id="primary-yaw-value">0 deg</span>
              </div>
            </div>

            <div class="control-card">
              <div class="row-title"><span>Newcomer</span><span class="chip newcomer">P2</span></div>
              <div class="slider-row">
                <span class="label">Affinity</span>
                <input id="newcomer-affinity" type="range" min="0" max="1" step="0.01">
                <span class="slider-value" id="newcomer-affinity-value">0.50</span>
              </div>
              <div class="slider-row">
                <span class="label">Yaw</span>
                <input id="newcomer-yaw" type="range" min="-180" max="180" step="1">
                <span class="slider-value" id="newcomer-yaw-value">0 deg</span>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </div>
  </div>

  <script>
    const WORLD_HALF_EXTENT = 4.2;
    const canvas = document.getElementById("viz");
    const ctx = canvas.getContext("2d");
    const connectionPill = document.getElementById("connection-pill");
    const eventPill = document.getElementById("event-pill");
    const summaryPill = document.getElementById("summary-pill");
    const eventValue = document.getElementById("event-value");
    const stageValue = document.getElementById("stage-value");
    const focusValue = document.getElementById("focus-value");
    const joiningValue = document.getElementById("joining-value");
    const admittedValue = document.getElementById("admitted-value");
    const targetValue = document.getElementById("target-value");
    const sharedValue = document.getElementById("shared-value");
    const primaryDistValue = document.getElementById("primary-dist-value");
    const newcomerDistValue = document.getElementById("newcomer-dist-value");
    const sameSideValue = document.getElementById("same-side-value");
    const motionLimitValue = document.getElementById("motion-limit-value");
    const toggleRunButton = document.getElementById("toggle-run");
    const approachButton = document.getElementById("p2-approach");
    const primaryAffinity = document.getElementById("primary-affinity");
    const newcomerAffinity = document.getElementById("newcomer-affinity");
    const primaryAffinityValue = document.getElementById("primary-affinity-value");
    const newcomerAffinityValue = document.getElementById("newcomer-affinity-value");
    const primaryYaw = document.getElementById("primary-yaw");
    const newcomerYaw = document.getElementById("newcomer-yaw");
    const primaryYawValue = document.getElementById("primary-yaw-value");
    const newcomerYawValue = document.getElementById("newcomer-yaw-value");

    let state = null;
    let dragActorId = null;
    let lastDragSentAt = 0;
    const pendingTimers = new Map();

    function resizeCanvas() {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width * ratio));
      const height = Math.max(260, Math.round(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function worldToCanvas(point) {
      const scale = Math.min(canvas.width, canvas.height) / (2 * WORLD_HALF_EXTENT);
      return {
        x: canvas.width * 0.5 + point.x * scale,
        y: canvas.height * 0.5 - point.y * scale,
      };
    }

    function canvasToWorld(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const x = (clientX - rect.left) * ratio;
      const y = (clientY - rect.top) * ratio;
      const scale = Math.min(canvas.width, canvas.height) / (2 * WORLD_HALF_EXTENT);
      return {
        x: (x - canvas.width * 0.5) / scale,
        y: -(y - canvas.height * 0.5) / scale,
      };
    }

    function roundValue(value) {
      return Number(value).toFixed(2);
    }

    function deg(value) {
      return `${(value * 180 / Math.PI).toFixed(1)} deg`;
    }

    function toSliderDegrees(value) {
      return `${Math.round(value)} deg`;
    }

    function angleTip(origin, angle, length) {
      return {
        x: origin.x + Math.cos(angle) * length,
        y: origin.y + Math.sin(angle) * length,
      };
    }

    function drawArrow(origin, angle, length, color, width) {
      const start = worldToCanvas(origin);
      const tip = worldToCanvas(angleTip(origin, angle, length));
      const headLength = 12 * (window.devicePixelRatio || 1);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(tip.x, tip.y);
      ctx.stroke();
      const theta = Math.atan2(tip.y - start.y, tip.x - start.x);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(tip.x - headLength * Math.cos(theta - Math.PI / 7), tip.y - headLength * Math.sin(theta - Math.PI / 7));
      ctx.lineTo(tip.x - headLength * Math.cos(theta + Math.PI / 7), tip.y - headLength * Math.sin(theta + Math.PI / 7));
      ctx.closePath();
      ctx.fill();
    }

    function drawGrid() {
      ctx.fillStyle = "#fbfaf6";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#ece5d8";
      ctx.lineWidth = 1;
      for (let world = -4; world <= 4; world += 1) {
        let a = worldToCanvas({x: -WORLD_HALF_EXTENT, y: world});
        let b = worldToCanvas({x: WORLD_HALF_EXTENT, y: world});
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        a = worldToCanvas({x: world, y: -WORLD_HALF_EXTENT});
        b = worldToCanvas({x: world, y: WORLD_HALF_EXTENT});
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    function actorEntries() {
      if (!state) {
        return [];
      }
      return [
        {id: "robot", actor: state.robot, label: "R", fill: "#d5573b", outline: "#6d1d10"},
        {id: "primary", actor: state.primary, label: "P1", fill: "#4f8aa3", outline: "#1e4254"},
        {id: "newcomer", actor: state.newcomer, label: "P2", fill: "#7b9d62", outline: "#3e5b2d"},
      ];
    }

    function nearestActor(worldPoint, radius = 0.55) {
      let nearest = null;
      let bestDistance = Infinity;
      for (const entry of actorEntries()) {
        const dx = entry.actor.position.x - worldPoint.x;
        const dy = entry.actor.position.y - worldPoint.y;
        const distance = Math.hypot(dx, dy);
        if (distance < bestDistance) {
          bestDistance = distance;
          nearest = entry;
        }
      }
      return bestDistance <= radius ? nearest : null;
    }

    function drawScene() {
      resizeCanvas();
      drawGrid();
      if (!state) {
        return;
      }

      const dyadCenter = worldToCanvas(state.plan.dyad_center);
      const sharedCenter = worldToCanvas(state.plan.shared_center);
      const target = worldToCanvas(state.plan.target_position);
      const newcomerApproachTarget = state.event.approach_target_position
        ? worldToCanvas(state.event.approach_target_position)
        : null;
      const robot = state.robot.position;
      const robotCanvas = worldToCanvas(robot);
      const scale = Math.min(canvas.width, canvas.height) / (2 * WORLD_HALF_EXTENT);

      ctx.setLineDash([12, 12]);
      ctx.strokeStyle = "rgba(95, 106, 113, 0.45)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(dyadCenter.x, dyadCenter.y, state.params.admit_distance * scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = "rgba(95, 106, 113, 0.35)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(worldToCanvas(state.primary.position).x, worldToCanvas(state.primary.position).y);
      ctx.lineTo(robotCanvas.x, robotCanvas.y);
      ctx.stroke();

      ctx.strokeStyle = "rgba(123, 157, 98, 0.4)";
      ctx.beginPath();
      ctx.moveTo(worldToCanvas(state.primary.position).x, worldToCanvas(state.primary.position).y);
      ctx.lineTo(worldToCanvas(state.newcomer.position).x, worldToCanvas(state.newcomer.position).y);
      ctx.stroke();

      if (state.event.phase === "p2_approaching" && newcomerApproachTarget) {
        const newcomerCanvas = worldToCanvas(state.newcomer.position);
        ctx.setLineDash([10, 10]);
        ctx.strokeStyle = "rgba(123, 157, 98, 0.8)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(newcomerCanvas.x, newcomerCanvas.y);
        ctx.lineTo(newcomerApproachTarget.x, newcomerApproachTarget.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = "#7b9d62";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(newcomerApproachTarget.x, newcomerApproachTarget.y, 7, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.fillStyle = "#1f2529";
      ctx.beginPath();
      ctx.arc(dyadCenter.x, dyadCenter.y, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#5f6a71";
      ctx.beginPath();
      ctx.arc(sharedCenter.x, sharedCenter.y, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.setLineDash([10, 10]);
      ctx.strokeStyle = "#d5573b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(robotCanvas.x, robotCanvas.y);
      ctx.lineTo(target.x, target.y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = "#d5573b";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(target.x, target.y, 8, 0, Math.PI * 2);
      ctx.stroke();

      for (const entry of actorEntries()) {
        const point = worldToCanvas(entry.actor.position);
        ctx.fillStyle = entry.fill;
        ctx.strokeStyle = entry.outline;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(point.x, point.y, entry.id === "robot" ? 18 : 16, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        drawArrow(entry.actor.position, entry.actor.yaw, entry.id === "robot" ? 0.56 : 0.42, entry.outline, 2);
        ctx.fillStyle = "#1f2529";
        ctx.font = `${12 * (window.devicePixelRatio || 1)}px Menlo, monospace`;
        ctx.textAlign = "center";
        ctx.fillText(entry.label, point.x, point.y - 24);
      }

      drawArrow(state.robot.position, state.plan.gaze_yaw, 0.9, "#e0a219", 2);
    }

    function syncControls() {
      if (!state) {
        return;
      }

      primaryAffinityValue.textContent = roundValue(state.primary.affinity);
      newcomerAffinityValue.textContent = roundValue(state.newcomer.affinity);
      if (document.activeElement !== primaryAffinity) {
        primaryAffinity.value = String(state.primary.affinity);
      }
      if (document.activeElement !== newcomerAffinity) {
        newcomerAffinity.value = String(state.newcomer.affinity);
      }

      const primaryYawDeg = state.primary.yaw * 180 / Math.PI;
      const newcomerYawDeg = state.newcomer.yaw * 180 / Math.PI;
      primaryYawValue.textContent = toSliderDegrees(primaryYawDeg);
      newcomerYawValue.textContent = toSliderDegrees(newcomerYawDeg);
      if (document.activeElement !== primaryYaw) {
        primaryYaw.value = String(primaryYawDeg);
      }
      if (document.activeElement !== newcomerYaw) {
        newcomerYaw.value = String(newcomerYawDeg);
      }
    }

    function updateMetrics() {
      if (!state) {
        return;
      }
      eventValue.textContent = state.event.phase;
      stageValue.textContent = state.plan.stage;
      focusValue.textContent = state.plan.focus_person_id || "shared";
      joiningValue.textContent = roundValue(state.plan.joining_score);
      admittedValue.textContent = state.plan.admitted ? "yes" : "no";
      targetValue.textContent = `(${roundValue(state.plan.target_position.x)}, ${roundValue(state.plan.target_position.y)})`;
      sharedValue.textContent = `(${roundValue(state.plan.shared_center.x)}, ${roundValue(state.plan.shared_center.y)})`;
      primaryDistValue.textContent = roundValue(state.plan.target_primary_distance);
      newcomerDistValue.textContent = roundValue(state.plan.target_newcomer_distance);
      sameSideValue.textContent = state.plan.same_side_preserved ? "yes" : "no";
      motionLimitValue.textContent = state.plan.motion_limit_applied ? "yes" : "no";
      toggleRunButton.textContent = state.running ? "Pause" : "Resume";
      approachButton.disabled = state.event.phase !== "dyad";
      eventPill.textContent = `Event: ${state.event.phase}`;
      summaryPill.textContent = state.summary;
      connectionPill.textContent = connectionLabel;
      syncControls();
    }

    const isLocalDemo = ["127.0.0.1", "localhost"].includes(window.location.hostname);
    const connectionLabel = isLocalDemo
      ? `Connected to dyad -> triad demo on port ${window.location.port || "80"}`
      : `Connected to dyad -> triad demo at ${window.location.host}`;
    const pollIntervalMs = isLocalDemo ? 120 : 350;
    let pollTimer = null;
    let stateFetchPromise = null;
    let consecutiveFetchFailures = 0;

    function scheduleNextPoll(delayMs = pollIntervalMs) {
      if (pollTimer !== null) {
        window.clearTimeout(pollTimer);
      }
      pollTimer = window.setTimeout(() => {
        void fetchState();
      }, delayMs);
    }

    async function fetchState({awaitInFlight = false} = {}) {
      if (stateFetchPromise) {
        return awaitInFlight ? stateFetchPromise : undefined;
      }
      stateFetchPromise = (async () => {
        try {
          const response = await fetch("/dyad-triad/state", {
            cache: "no-store",
            headers: {"Accept": "application/json"},
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          state = await response.json();
          consecutiveFetchFailures = 0;
          updateMetrics();
        } catch (error) {
          consecutiveFetchFailures += 1;
          if (consecutiveFetchFailures >= 2) {
            const message = error instanceof Error ? error.message : String(error);
            connectionPill.textContent = `Reconnecting to demo... (${message})`;
          }
        } finally {
          stateFetchPromise = null;
          scheduleNextPoll();
        }
      })();
      return stateFetchPromise;
    }

    async function postCommand(payload) {
      try {
        const response = await fetch("/dyad-triad/command", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        await fetchState({awaitInFlight: true});
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        connectionPill.textContent = `Command failed: ${message}`;
      }
    }

    function queueCommand(key, payload) {
      const existing = pendingTimers.get(key);
      if (existing) {
        clearTimeout(existing);
      }
      const timer = setTimeout(() => {
        pendingTimers.delete(key);
        postCommand(payload);
      }, 70);
      pendingTimers.set(key, timer);
    }

    canvas.addEventListener("pointerdown", (event) => {
      const world = canvasToWorld(event.clientX, event.clientY);
      const nearest = nearestActor(world);
      if (nearest) {
        dragActorId = nearest.id;
        canvas.setPointerCapture(event.pointerId);
      }
    });

    canvas.addEventListener("pointermove", async (event) => {
      if (!dragActorId) {
        return;
      }
      const world = canvasToWorld(event.clientX, event.clientY);
      const now = performance.now();
      if (now - lastDragSentAt < 33) {
        return;
      }
      lastDragSentAt = now;
      await postCommand({action: "move_actor", actor_id: dragActorId, x: world.x, y: world.y});
    });

    canvas.addEventListener("pointerup", async (event) => {
      if (dragActorId) {
        const world = canvasToWorld(event.clientX, event.clientY);
        await postCommand({action: "move_actor", actor_id: dragActorId, x: world.x, y: world.y});
      }
      dragActorId = null;
      try {
        canvas.releasePointerCapture(event.pointerId);
      } catch (_error) {
      }
    });

    primaryAffinity.addEventListener("input", (event) => {
      const value = Number(event.target.value);
      primaryAffinityValue.textContent = roundValue(value);
      queueCommand("primary-affinity", {action: "set_affinity", actor_id: "primary", affinity: value});
    });

    newcomerAffinity.addEventListener("input", (event) => {
      const value = Number(event.target.value);
      newcomerAffinityValue.textContent = roundValue(value);
      queueCommand("newcomer-affinity", {action: "set_affinity", actor_id: "newcomer", affinity: value});
    });

    primaryYaw.addEventListener("input", (event) => {
      const value = Number(event.target.value);
      primaryYawValue.textContent = toSliderDegrees(value);
      queueCommand("primary-yaw", {action: "set_yaw", actor_id: "primary", yaw: value * Math.PI / 180});
    });

    newcomerYaw.addEventListener("input", (event) => {
      const value = Number(event.target.value);
      newcomerYawValue.textContent = toSliderDegrees(value);
      queueCommand("newcomer-yaw", {action: "set_yaw", actor_id: "newcomer", yaw: value * Math.PI / 180});
    });

    document.getElementById("toggle-run").addEventListener("click", () => postCommand({action: "toggle_running"}));
    document.getElementById("p2-approach").addEventListener("click", () => postCommand({action: "trigger_approach"}));
    document.getElementById("primary-speaks").addEventListener("click", () => postCommand({action: "set_speaker", actor_id: "primary"}));
    document.getElementById("newcomer-speaks").addEventListener("click", () => postCommand({action: "set_speaker", actor_id: "newcomer"}));
    document.getElementById("clear-speaker").addEventListener("click", () => postCommand({action: "set_speaker", actor_id: "none"}));
    document.getElementById("reset").addEventListener("click", () => postCommand({action: "reset"}));

    function renderLoop() {
      drawScene();
      requestAnimationFrame(renderLoop);
    }

    window.addEventListener("resize", drawScene);
    scheduleNextPoll(0);
    renderLoop();
  </script>
</body>
</html>
"""


DETOUR_WEB_APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Social Detour To Charge Dock</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f6f1e6;
      --panel: rgba(255, 252, 246, 0.94);
      --ink: #20252a;
      --muted: #5f6a71;
      --grid: #e7decd;
      --robot: #d5573b;
      --dock: #4d8a73;
      --person: #4f9dc2;
      --speaker: #efb24a;
      --path: #d5573b;
      --direct: #7f8d99;
      --shadow: 0 24px 56px rgba(51, 39, 24, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(255, 255, 255, 0.74), transparent 34%),
        linear-gradient(180deg, #f7efe0 0%, #ece2cf 100%);
    }

    .shell {
      width: min(1240px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      gap: 18px;
    }

    .hero,
    .panel {
      background: var(--panel);
      border: 1px solid rgba(85, 65, 40, 0.12);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .hero {
      padding: 20px 22px;
      display: grid;
      gap: 12px;
    }

    .hero h1 {
      margin: 0;
      font-size: clamp(1.45rem, 2.2vw, 2rem);
    }

    .hero p,
    .hint {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    button,
    .nav-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
    }

    button {
      color: white;
      background: linear-gradient(135deg, #d5573b, #c24128);
      box-shadow: 0 10px 22px rgba(213, 87, 59, 0.22);
    }

    button.secondary {
      background: linear-gradient(135deg, #5c7d71, #456156);
      box-shadow: 0 10px 22px rgba(70, 97, 86, 0.18);
    }

    button.active {
      box-shadow:
        inset 0 0 0 2px rgba(255, 255, 255, 0.75),
        0 10px 22px rgba(213, 87, 59, 0.22);
    }

    button.ghost,
    .nav-link {
      color: var(--ink);
      background: rgba(255, 255, 255, 0.86);
      border: 1px solid rgba(31, 37, 41, 0.12);
      box-shadow: none;
    }

    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(31, 37, 41, 0.08);
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.85fr);
      gap: 18px;
    }

    .panel {
      padding: 16px;
    }

    canvas {
      width: 100%;
      height: min(72vh, 760px);
      display: block;
      border-radius: 18px;
      background: var(--paper);
      touch-action: none;
    }

    .metrics {
      display: grid;
      gap: 12px;
    }

    h2 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(31, 37, 41, 0.08);
    }

    .metric .label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 0.85rem;
    }

    .metric .value {
      display: block;
      font-size: 1rem;
      font-weight: 700;
    }

    .legend {
      display: grid;
      gap: 10px;
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(31, 37, 41, 0.08);
    }

    @media (max-width: 940px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Social Detour To The Charge Dock</h1>
      <p>This page adds a dedicated detour scenario inspired by Zhou et al. (2022): the robot must reach a distant charging dock while a person or an interacting social group blocks the geometric shortest path. The planner therefore minimizes path length together with cumulative social influence, so it detours around people instead of cutting through them.</p>
      <div class="toolbar">
        <button id="toggle-run" type="button">Pause</button>
        <button id="scenario-single" class="secondary active" type="button">Single Person</button>
        <button id="scenario-group" class="secondary" type="button">Interactive Group</button>
        <button id="rotate-scene" class="ghost" type="button">Rotate Blocker</button>
        <button id="reset" class="ghost" type="button">Reset Scene</button>
        <a class="nav-link" href="/">Group Demo</a>
        <a class="nav-link" href="/dyad-triad">Dyad -> Triad</a>
      </div>
      <div class="status-row">
        <div class="pill" id="connection-pill">Connecting...</div>
        <div class="pill" id="scenario-pill">Scenario: single person</div>
        <div class="pill" id="summary-pill">Waiting for state...</div>
      </div>
    </section>

    <div class="layout">
      <section class="panel">
        <canvas id="viz" width="980" height="760"></canvas>
      </section>

      <aside class="panel">
        <div class="metrics">
          <h2>Transit State</h2>
          <div class="metric-grid">
            <div class="metric"><span class="label">Stage</span><span class="value" id="stage-value">-</span></div>
            <div class="metric"><span class="label">People</span><span class="value" id="people-value">-</span></div>
            <div class="metric"><span class="label">Direct Line</span><span class="value" id="direct-value">-</span></div>
            <div class="metric"><span class="label">Social Path</span><span class="value" id="path-value">-</span></div>
            <div class="metric"><span class="label">Extra Distance</span><span class="value" id="extra-value">-</span></div>
            <div class="metric"><span class="label">Remaining</span><span class="value" id="remaining-value">-</span></div>
            <div class="metric"><span class="label">Min Clearance</span><span class="value" id="clearance-value">-</span></div>
            <div class="metric"><span class="label">Interaction Links</span><span class="value" id="links-value">-</span></div>
          </div>

          <h2>Interpretation</h2>
          <div class="legend">
            <p class="hint">Dashed grey line: pure geometric shortest path to the dock.</p>
            <p class="hint">Solid red line: socially aware path generated from the influence field around people.</p>
            <p class="hint">Blue circles: blockers. In the single-person case, the facing direction acts as the social cue that biases passing behavior toward the person's back side.</p>
            <p class="hint">Drag the robot, the dock, or any person on the canvas to reshape the scenario in real time.</p>
          </div>
        </div>
      </aside>
    </div>
  </div>

  <script>
    const WORLD_HALF_EXTENT = 4.2;
    const canvas = document.getElementById("viz");
    const ctx = canvas.getContext("2d");
    const connectionPill = document.getElementById("connection-pill");
    const scenarioPill = document.getElementById("scenario-pill");
    const summaryPill = document.getElementById("summary-pill");
    const stageValue = document.getElementById("stage-value");
    const peopleValue = document.getElementById("people-value");
    const directValue = document.getElementById("direct-value");
    const pathValue = document.getElementById("path-value");
    const extraValue = document.getElementById("extra-value");
    const remainingValue = document.getElementById("remaining-value");
    const clearanceValue = document.getElementById("clearance-value");
    const linksValue = document.getElementById("links-value");
    const toggleRunButton = document.getElementById("toggle-run");
    const singleButton = document.getElementById("scenario-single");
    const groupButton = document.getElementById("scenario-group");
    const rotateButton = document.getElementById("rotate-scene");

    let state = null;
    let dragActorId = null;
    let lastDragSentAt = 0;

    function resizeCanvas() {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width * ratio));
      const height = Math.max(260, Math.round(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function worldToCanvas(point) {
      const scale = Math.min(canvas.width, canvas.height) / (2 * WORLD_HALF_EXTENT);
      return {
        x: canvas.width * 0.5 + point.x * scale,
        y: canvas.height * 0.5 - point.y * scale,
      };
    }

    function canvasToWorld(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const x = (clientX - rect.left) * ratio;
      const y = (clientY - rect.top) * ratio;
      const scale = Math.min(canvas.width, canvas.height) / (2 * WORLD_HALF_EXTENT);
      return {
        x: (x - canvas.width * 0.5) / scale,
        y: -(y - canvas.height * 0.5) / scale,
      };
    }

    function roundValue(value) {
      return Number(value).toFixed(2);
    }

    function angleTip(origin, angle, length) {
      return {
        x: origin.x + Math.cos(angle) * length,
        y: origin.y + Math.sin(angle) * length,
      };
    }

    function drawArrow(origin, angle, length, color, width) {
      const start = worldToCanvas(origin);
      const tip = worldToCanvas(angleTip(origin, angle, length));
      const headLength = 12 * (window.devicePixelRatio || 1);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(tip.x, tip.y);
      ctx.stroke();
      const theta = Math.atan2(tip.y - start.y, tip.x - start.x);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(tip.x - headLength * Math.cos(theta - Math.PI / 7), tip.y - headLength * Math.sin(theta - Math.PI / 7));
      ctx.lineTo(tip.x - headLength * Math.cos(theta + Math.PI / 7), tip.y - headLength * Math.sin(theta + Math.PI / 7));
      ctx.closePath();
      ctx.fill();
    }

    function drawGrid() {
      ctx.fillStyle = "#fbfaf6";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#ece5d8";
      ctx.lineWidth = 1;
      for (let world = -4; world <= 4; world += 1) {
        let a = worldToCanvas({x: -WORLD_HALF_EXTENT, y: world});
        let b = worldToCanvas({x: WORLD_HALF_EXTENT, y: world});
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        a = worldToCanvas({x: world, y: -WORLD_HALF_EXTENT});
        b = worldToCanvas({x: world, y: WORLD_HALF_EXTENT});
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    function drawFieldHeatmap(scale) {
      if (!state || !state.plan.field_samples || state.plan.field_samples.length === 0) {
        return;
      }
      const blockCost = Math.max(2.0, state.params?.path_block_cost || 10.0);
      const cellSize = Math.max(8, scale * (state.params?.path_grid_resolution || 0.18) * 2.0);
      for (const sample of state.plan.field_samples) {
        const normalized = Math.max(0, Math.min(1, (sample.cost - 1.0) / (blockCost - 1.0)));
        if (normalized < 0.03) {
          continue;
        }
        const point = worldToCanvas(sample);
        ctx.fillStyle = `rgba(166, 102, 41, ${0.03 + 0.22 * normalized})`;
        ctx.fillRect(point.x - cellSize * 0.5, point.y - cellSize * 0.5, cellSize, cellSize);
      }
    }

    function actorEntries() {
      if (!state) {
        return [];
      }
      return [
        {id: "robot", type: "robot", point: state.robot.position},
        {id: "dock", type: "dock", point: state.charge_dock},
        ...state.people.map((person) => ({id: person.person_id, type: "person", point: person.position})),
      ];
    }

    function nearestActor(worldPoint, radius = 0.62) {
      let nearest = null;
      let bestDistance = Infinity;
      for (const actor of actorEntries()) {
        const dx = actor.point.x - worldPoint.x;
        const dy = actor.point.y - worldPoint.y;
        const distance = Math.hypot(dx, dy);
        if (distance < bestDistance) {
          bestDistance = distance;
          nearest = actor;
        }
      }
      return bestDistance <= radius ? nearest : null;
    }

    function drawDock(point) {
      const dock = worldToCanvas(point);
      ctx.fillStyle = "#4d8a73";
      ctx.strokeStyle = "#274a3d";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.rect(dock.x - 18, dock.y - 18, 36, 36);
      ctx.fill();
      ctx.stroke();

      ctx.strokeStyle = "rgba(255, 255, 255, 0.86)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(dock.x - 4, dock.y - 10);
      ctx.lineTo(dock.x + 4, dock.y - 2);
      ctx.lineTo(dock.x, dock.y - 2);
      ctx.lineTo(dock.x + 6, dock.y + 10);
      ctx.stroke();

      ctx.fillStyle = "#1f2529";
      ctx.font = `${12 * (window.devicePixelRatio || 1)}px Menlo, monospace`;
      ctx.textAlign = "center";
      ctx.fillText("Dock", dock.x, dock.y - 28);
    }

    function drawScene() {
      resizeCanvas();
      drawGrid();
      if (!state) {
        return;
      }

      const scale = Math.min(canvas.width, canvas.height) / (2 * WORLD_HALF_EXTENT);
      drawFieldHeatmap(scale);

      if (state.people.length > 1 && state.plan.group_radius > 0.01) {
        const center = worldToCanvas(state.plan.group_center);
        ctx.setLineDash([10, 10]);
        ctx.strokeStyle = "#bcb39f";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(center.x, center.y, state.plan.group_radius * scale, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#1f2529";
        ctx.beginPath();
        ctx.arc(center.x, center.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }

      const peopleById = Object.fromEntries(state.people.map((person) => [person.person_id, person]));
      for (const link of state.plan.interaction_links || []) {
        const aPerson = peopleById[link.a];
        const bPerson = peopleById[link.b];
        if (!aPerson || !bPerson) {
          continue;
        }
        const a = worldToCanvas(aPerson.position);
        const b = worldToCanvas(bPerson.position);
        ctx.strokeStyle = `rgba(110, 98, 78, ${0.12 + 0.36 * link.probability})`;
        ctx.lineWidth = 1.5 + 4.2 * link.probability;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }

      const robotCanvas = worldToCanvas(state.robot.position);
      const dockCanvas = worldToCanvas(state.charge_dock);
      ctx.setLineDash([12, 12]);
      ctx.strokeStyle = "#7f8d99";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(robotCanvas.x, robotCanvas.y);
      ctx.lineTo(dockCanvas.x, dockCanvas.y);
      ctx.stroke();
      ctx.setLineDash([]);

      if (state.plan.path_points && state.plan.path_points.length > 1) {
        ctx.strokeStyle = "rgba(213, 87, 59, 0.94)";
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        const first = worldToCanvas(state.plan.path_points[0]);
        ctx.moveTo(first.x, first.y);
        for (const waypoint of state.plan.path_points.slice(1)) {
          const point = worldToCanvas(waypoint);
          ctx.lineTo(point.x, point.y);
        }
        ctx.stroke();
      }

      drawDock(state.charge_dock);

      for (const person of state.people) {
        const point = worldToCanvas(person.position);
        if (person.speaking_score > 0.1) {
          ctx.strokeStyle = "#efb24a";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(point.x, point.y, 24 + 10 * person.speaking_score, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.fillStyle = "#4f9dc2";
        ctx.strokeStyle = person.speaking_score > 0.1 ? "#8f4900" : "#163444";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 16, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        drawArrow(person.position, person.yaw ?? 0, 0.46, "#163444", 2);

        ctx.fillStyle = "#1f2529";
        ctx.font = `${12 * (window.devicePixelRatio || 1)}px Menlo, monospace`;
        ctx.textAlign = "center";
        ctx.fillText(person.person_id, point.x, point.y - 24);
      }

      ctx.fillStyle = "#d5573b";
      ctx.strokeStyle = "#6d1d10";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(robotCanvas.x, robotCanvas.y, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      drawArrow(state.robot.position, state.robot.yaw, 0.56, "#6d1d10", 3);
      drawArrow(state.robot.position, state.plan.gaze_yaw, 0.9, "#e0a219", 2);
    }

    function updateMetrics() {
      if (!state) {
        return;
      }
      stageValue.textContent = state.plan.stage;
      peopleValue.textContent = `${state.people.length}`;
      directValue.textContent = `${roundValue(state.plan.direct_distance)} m`;
      pathValue.textContent = `${roundValue(state.plan.path_length)} m`;
      extraValue.textContent = `${roundValue(state.plan.extra_distance)} m`;
      remainingValue.textContent = `${roundValue(state.plan.distance_to_target)} m`;
      clearanceValue.textContent = `${roundValue(state.plan.min_clearance)} m`;
      linksValue.textContent = `${state.plan.interaction_links.length}`;
      toggleRunButton.textContent = state.running ? "Pause" : "Resume";
      summaryPill.textContent = state.summary;
      scenarioPill.textContent = `Scenario: ${state.scenario_label}`;
      connectionPill.textContent = connectionLabel;
      singleButton.classList.toggle("active", state.scenario_mode === "single_person");
      groupButton.classList.toggle("active", state.scenario_mode === "interactive_group");
      rotateButton.textContent = state.scenario_mode === "single_person" ? "Rotate Blocker" : "Rotate Speaker";
    }

    const isLocalDemo = ["127.0.0.1", "localhost"].includes(window.location.hostname);
    const connectionLabel = isLocalDemo
      ? `Connected to detour demo on port ${window.location.port || "80"}`
      : `Connected to detour demo at ${window.location.host}`;
    const pollIntervalMs = isLocalDemo ? 120 : 350;
    let pollTimer = null;
    let stateFetchPromise = null;
    let consecutiveFetchFailures = 0;

    function scheduleNextPoll(delayMs = pollIntervalMs) {
      if (pollTimer !== null) {
        window.clearTimeout(pollTimer);
      }
      pollTimer = window.setTimeout(() => {
        void fetchState();
      }, delayMs);
    }

    async function fetchState({awaitInFlight = false} = {}) {
      if (stateFetchPromise) {
        return awaitInFlight ? stateFetchPromise : undefined;
      }
      stateFetchPromise = (async () => {
        try {
          const response = await fetch("/detour/state", {
            cache: "no-store",
            headers: {"Accept": "application/json"},
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          state = await response.json();
          consecutiveFetchFailures = 0;
          updateMetrics();
        } catch (error) {
          consecutiveFetchFailures += 1;
          if (consecutiveFetchFailures >= 2) {
            const message = error instanceof Error ? error.message : String(error);
            connectionPill.textContent = `Reconnecting to demo... (${message})`;
          }
        } finally {
          stateFetchPromise = null;
          scheduleNextPoll();
        }
      })();
      return stateFetchPromise;
    }

    async function postCommand(payload) {
      try {
        const response = await fetch("/detour/command", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        await fetchState({awaitInFlight: true});
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        connectionPill.textContent = `Command failed: ${message}`;
      }
    }

    canvas.addEventListener("pointerdown", (event) => {
      const world = canvasToWorld(event.clientX, event.clientY);
      const nearest = nearestActor(world);
      if (nearest) {
        dragActorId = nearest.id;
        canvas.setPointerCapture(event.pointerId);
      }
    });

    canvas.addEventListener("pointermove", async (event) => {
      if (!dragActorId) {
        return;
      }
      const world = canvasToWorld(event.clientX, event.clientY);
      const now = performance.now();
      if (now - lastDragSentAt < 33) {
        return;
      }
      lastDragSentAt = now;
      await postCommand({action: "move_actor", actor_id: dragActorId, x: world.x, y: world.y});
    });

    canvas.addEventListener("pointerup", async (event) => {
      if (dragActorId) {
        const world = canvasToWorld(event.clientX, event.clientY);
        await postCommand({action: "move_actor", actor_id: dragActorId, x: world.x, y: world.y});
      }
      dragActorId = null;
      try {
        canvas.releasePointerCapture(event.pointerId);
      } catch (_error) {
      }
    });

    document.getElementById("toggle-run").addEventListener("click", () => postCommand({action: "toggle_running"}));
    document.getElementById("scenario-single").addEventListener("click", () => postCommand({action: "set_scenario", mode: "single_person"}));
    document.getElementById("scenario-group").addEventListener("click", () => postCommand({action: "set_scenario", mode: "interactive_group"}));
    document.getElementById("rotate-scene").addEventListener("click", () => postCommand({action: "rotate_scene"}));
    document.getElementById("reset").addEventListener("click", () => postCommand({action: "reset"}));

    function renderLoop() {
      drawScene();
      requestAnimationFrame(renderLoop);
    }

    window.addEventListener("resize", drawScene);
    scheduleNextPoll(0);
    renderLoop();
  </script>
</body>
</html>
"""


class WebDemoController:
    def __init__(self) -> None:
        self.params = SocialNavParams()
        self.navigator = SocialNavigator(self.params)
        self.lock = threading.Lock()
        self.running = True
        self.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._reset_locked()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _reset_locked(self) -> None:
        self.group_size = 4
        self.robot = Pose2D(-2.8, -2.2, math.radians(35.0))
        self.people = demo_people(count=self.group_size, speaker_person_id="P1")
        self.last_time = time.monotonic()
        self.last_speaker_rotation = self.last_time
        self.inviter_person_id = self._preferred_inviter_id_locked("P1")
        self.invitation_active = False
        self.locked_invitation_plan: SocialPlan | None = None
        self.navigator.reset_attention()
        self.latest_plan = self._compute_plan_locked(self.last_time)

    def reset(self) -> None:
        with self.lock:
            self._reset_locked()

    def _resolve_person_id_locked(self, requested_person_id: str | None = None) -> str | None:
        person_ids = [person.person_id for person in self.people]
        if not person_ids:
            return None
        if requested_person_id and requested_person_id in person_ids:
            return requested_person_id
        return None

    def _preferred_inviter_id_locked(self, requested_person_id: str | None = None) -> str | None:
        person_ids = [person.person_id for person in self.people]
        if not person_ids:
            return None
        if requested_person_id and requested_person_id in person_ids:
            return requested_person_id
        if getattr(self, "inviter_person_id", None) in person_ids:
            return self.inviter_person_id
        if "P1" in person_ids:
            return "P1"
        return person_ids[0]

    def _active_speaker_id_locked(self) -> str | None:
        return active_speaker_id(self.people)

    def _set_speaker_locked(self, speaker_person_id: str | None) -> str | None:
        resolved_speaker_id = self._resolve_person_id_locked(speaker_person_id)
        return set_group_speaker(self.people, resolved_speaker_id)

    def _clear_locked_invitation_plan_locked(self) -> None:
        self.locked_invitation_plan = None

    def _compute_plan_locked(self, now_s: float):
        inviter_person_id = self._preferred_inviter_id_locked(getattr(self, "inviter_person_id", None))
        self.inviter_person_id = inviter_person_id
        if not self.invitation_active:
            self.locked_invitation_plan = None
            return self.navigator.update(
                self.robot,
                self.people,
                now_s,
                inviter_person_id=None,
            )

        if self.locked_invitation_plan is None:
            self.locked_invitation_plan = compute_social_plan(
                self.robot,
                self.people,
                self.params,
                inviter_person_id=inviter_person_id,
            )

        plan = compute_social_plan_with_locked_target(
            self.robot,
            self.people,
            self.locked_invitation_plan,
            self.params,
        )
        return self.navigator.update(
            self.robot,
            self.people,
            now_s,
            inviter_person_id=inviter_person_id,
            precomputed_plan=plan,
        )

    def _cycle_primary_speaker_locked(self) -> None:
        if not self.people:
            return
        person_ids = [person.person_id for person in self.people]
        current_speaker_id = self._active_speaker_id_locked()
        current_index = person_ids.index(current_speaker_id) if current_speaker_id in person_ids else -1
        next_index = (current_index + 1) % len(self.people)
        self._set_speaker_locked(person_ids[next_index])
        self.last_speaker_rotation = time.monotonic()

    def cycle_primary_speaker(self) -> None:
        with self.lock:
            self._cycle_primary_speaker_locked()
            self.latest_plan = self._compute_plan_locked(time.monotonic())

    def invite_robot(self, person_id: str | None = None) -> None:
        with self.lock:
            inviter_person_id = self._preferred_inviter_id_locked(person_id)
            if inviter_person_id is None:
                return
            self.inviter_person_id = inviter_person_id
            self.invitation_active = True
            self._clear_locked_invitation_plan_locked()
            self._set_speaker_locked(inviter_person_id)
            now_s = time.monotonic()
            self.last_speaker_rotation = now_s
            self.latest_plan = self._compute_plan_locked(now_s)

    def set_group_size(self, person_count: int) -> None:
        with self.lock:
            self.group_size = max(2, min(4, person_count))
            self.people = demo_people(count=self.group_size, speaker_person_id="P1")
            self.inviter_person_id = self._preferred_inviter_id_locked("P1")
            self.invitation_active = False
            self._clear_locked_invitation_plan_locked()
            self.navigator.reset_attention()
            now_s = time.monotonic()
            self.last_time = now_s
            self.last_speaker_rotation = now_s
            self.latest_plan = self._compute_plan_locked(now_s)

    def randomize_people(self) -> None:
        with self.lock:
            person_count = max(2, min(4, getattr(self, "group_size", len(self.people))))
            radius = random.uniform(0.95, 1.55)
            center = (
                random.uniform(-0.3, 0.3),
                random.uniform(-0.15, 0.25),
            )
            self.people = []
            for index in range(person_count):
                angle = math.tau * index / person_count + random.uniform(-0.22, 0.22)
                x = center[0] + radius * math.cos(angle)
                y = center[1] + radius * math.sin(angle)
                yaw = math.atan2(center[1] - y, center[0] - x)
                speaking = 1.0 if index == 0 else (0.4 if index == 2 % person_count else 0.0)
                self.people.append(
                    PersonState(
                        person_id=f"P{index + 1}",
                        x=x,
                        y=y,
                        yaw=yaw,
                        engagement=1.0,
                        speaking_score=speaking,
                        affinity=0.5,
                    )
                )
            self.group_size = person_count
            self._set_speaker_locked("P1")
            self.navigator.reset_attention()
            self.last_speaker_rotation = time.monotonic()
            self.inviter_person_id = self._preferred_inviter_id_locked("P1")
            self.invitation_active = False
            self._clear_locked_invitation_plan_locked()
            self.latest_plan = self._compute_plan_locked(self.last_speaker_rotation)

    def toggle_running(self) -> bool:
        with self.lock:
            self.running = not self.running
            return self.running

    def add_person(self, x: float, y: float) -> None:
        with self.lock:
            if len(self.people) >= 4:
                return
            new_index = len(self.people) + 1
            center = compute_social_plan(self.robot, self.people, self.params).group_center
            yaw = math.atan2(center[1] - y, center[0] - x)
            self.people.append(
                PersonState(
                    person_id=f"P{new_index}",
                    x=x,
                    y=y,
                    yaw=yaw,
                    engagement=1.0,
                    speaking_score=0.0,
                    affinity=0.5,
                )
            )
            self.group_size = len(self.people)
            self._set_speaker_locked(self._active_speaker_id_locked() or "P1")
            self.navigator.reset_attention()
            self.inviter_person_id = self._preferred_inviter_id_locked(self.inviter_person_id)
            self._clear_locked_invitation_plan_locked()
            self.latest_plan = self._compute_plan_locked(time.monotonic())

    def set_affinity(self, person_id: str, affinity: float) -> None:
        with self.lock:
            person = next((candidate for candidate in self.people if candidate.person_id == person_id), None)
            if person is None:
                return
            person.affinity = max(0.0, min(1.0, affinity))
            self.navigator.reset_attention()
            self._clear_locked_invitation_plan_locked()
            self.latest_plan = self._compute_plan_locked(time.monotonic())

    def move_person(self, person_id: str, x: float, y: float) -> None:
        with self.lock:
            person = next((candidate for candidate in self.people if candidate.person_id == person_id), None)
            if person is None:
                return
            person.x = x
            person.y = y
            orient_people_toward_speaker(self.people, self._active_speaker_id_locked())
            self._clear_locked_invitation_plan_locked()
            self.latest_plan = self._compute_plan_locked(time.monotonic())

    def remove_person(self, person_id: str) -> None:
        with self.lock:
            if len(self.people) <= 2:
                return
            removed_inviter = person_id == self.inviter_person_id
            removed_speaker = person_id == self._active_speaker_id_locked()
            self.people = [person for person in self.people if person.person_id != person_id]
            self.group_size = len(self.people)
            if removed_speaker:
                self._set_speaker_locked(self._preferred_inviter_id_locked("P1"))
            else:
                orient_people_toward_speaker(self.people, self._active_speaker_id_locked())
            self.inviter_person_id = self._preferred_inviter_id_locked(self.inviter_person_id)
            if self.inviter_person_id is None or removed_inviter:
                self.invitation_active = False
            self._clear_locked_invitation_plan_locked()
            self.navigator.reset_attention()
            self.latest_plan = self._compute_plan_locked(time.monotonic())

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            plan = self.latest_plan
            waiting_for_invite = not self.invitation_active
            focus_person_id = self._active_speaker_id_locked() if waiting_for_invite else plan.focus_person_id
            if waiting_for_invite:
                summary = (
                    f"stage=wait_for_invite, "
                    f"target=({self.robot.x:.2f}, {self.robot.y:.2f}), "
                    f"focus={focus_person_id or 'group'}"
                )
                stage = "wait_for_invite"
                target_position = {"x": self.robot.x, "y": self.robot.y}
                path_points: list[dict[str, float]] = []
                field_samples: list[dict[str, float]] = []
                candidates: list[dict[str, object]] = []
            else:
                summary = format_plan_summary(plan)
                stage = plan.stage
                target_position = {"x": plan.target_position[0], "y": plan.target_position[1]}
                path_points = [
                    {"x": point[0], "y": point[1]}
                    for point in plan.path_points
                ]
                field_samples = [
                    {"x": sample[0], "y": sample[1], "cost": sample[2]}
                    for sample in plan.field_samples
                ]
                candidates = [
                    {
                        "target": {"x": candidate.target[0], "y": candidate.target[1]},
                        "score": candidate.score,
                        "affinity_score": candidate.affinity_score,
                        "is_best": index == 0,
                    }
                    for index, candidate in enumerate(plan.candidates[:4])
                ]
            return {
                "running": self.running,
                "summary": summary,
                "invitation_active": self.invitation_active,
                "inviter_person_id": self.inviter_person_id,
                "group_size": self.group_size,
                "speaker_person_id": self._active_speaker_id_locked(),
                "robot": {
                    "position": {"x": self.robot.x, "y": self.robot.y},
                    "yaw": self.robot.yaw,
                },
                "people": [
                    {
                        "person_id": person.person_id,
                        "position": {"x": person.x, "y": person.y},
                        "yaw": person.yaw,
                        "speaking_score": person.speaking_score,
                        "engagement": person.engagement,
                        "affinity": person.affinity,
                    }
                    for person in self.people
                ],
                "plan": {
                    "stage": stage,
                    "target_position": target_position,
                    "body_yaw": plan.body_yaw,
                    "gaze_yaw": plan.gaze_yaw,
                    "focus_person_id": focus_person_id,
                    "group_center": {"x": plan.group_center[0], "y": plan.group_center[1]},
                    "group_radius": plan.group_radius,
                    "slot_gap_radians": plan.slot_gap_radians,
                    "target_affinity": plan.target_affinity,
                    "path_points": path_points,
                    "field_samples": field_samples,
                    "interaction_links": [
                        {"a": link[0], "b": link[1], "probability": link[2]}
                        for link in plan.interaction_links
                    ],
                    "candidates": candidates,
                },
                "params": {
                    "path_grid_resolution": self.params.path_grid_resolution,
                    "path_block_cost": self.params.path_block_cost,
                },
            }

    def _loop(self) -> None:
        while not self.shutdown_event.is_set():
            now_s = time.monotonic()
            with self.lock:
                dt_s = min(0.05, max(0.0, now_s - self.last_time))
                self.last_time = now_s
                if self.running:
                    if self.invitation_active and now_s - self.last_speaker_rotation >= 3.6 and len(self.people) >= 2:
                        self._cycle_primary_speaker_locked()
                    plan = self._compute_plan_locked(now_s)
                    self.latest_plan = plan
                    if self.invitation_active:
                        self.robot = step_robot_toward_plan(self.robot, plan, dt_s, self.params)
            self.shutdown_event.wait(1.0 / 30.0)


class DyadTriadWebController:
    def __init__(self) -> None:
        self.params = DyadTriadParams()
        self.lock = threading.Lock()
        self.running = True
        self.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._reset_locked()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _reset_locked(self) -> None:
        self.robot = Pose2D(-1.05, 0.0, 0.0)
        self.primary = PersonState(
            person_id="P1",
            x=0.0,
            y=0.0,
            yaw=math.pi,
            engagement=1.0,
            speaking_score=1.0,
            affinity=0.8,
        )
        self.newcomer = PersonState(
            person_id="P2",
            x=2.7,
            y=1.9,
            yaw=math.radians(-95.0),
            engagement=1.0,
            speaking_score=0.0,
            affinity=0.45,
        )
        self.last_time = time.monotonic()
        self.event_phase = "dyad"
        self.approach_active = False
        self.approach_target: tuple[float, float] | None = None
        self.latest_plan = compute_dyad_to_triad_plan(self.robot, self.primary, self.newcomer, self.params)

    def reset(self) -> None:
        with self.lock:
            self._reset_locked()

    def toggle_running(self) -> bool:
        with self.lock:
            self.running = not self.running
            return self.running

    def _current_dyad_center_locked(self) -> tuple[float, float]:
        return (
            0.5 * (self.robot.x + self.primary.x),
            0.5 * (self.robot.y + self.primary.y),
        )

    def _current_approach_target_locked(self) -> tuple[float, float]:
        dyad_center = self._current_dyad_center_locked()
        pair_vector = (self.primary.x - self.robot.x, self.primary.y - self.robot.y)
        pair_length = math.hypot(pair_vector[0], pair_vector[1])
        if pair_length < 1e-6:
            pair_dir = (1.0, 0.0)
        else:
            pair_dir = (pair_vector[0] / pair_length, pair_vector[1] / pair_length)
        pair_normal = (-pair_dir[1], pair_dir[0])
        newcomer_side = (
            pair_vector[0] * (self.newcomer.y - dyad_center[1])
            - pair_vector[1] * (self.newcomer.x - dyad_center[0])
        )
        side_sign = 1.0 if newcomer_side >= 0.0 else -1.0
        return (
            dyad_center[0] + 0.22 * pair_dir[0] + side_sign * 0.95 * pair_normal[0],
            dyad_center[1] + 0.22 * pair_dir[1] + side_sign * 0.95 * pair_normal[1],
        )

    def trigger_approach(self) -> None:
        with self.lock:
            self.running = True
            self.newcomer.x = 2.7
            self.newcomer.y = 1.9
            self.newcomer.yaw = math.radians(-95.0)
            self.newcomer.speaking_score = 0.0
            self.primary.speaking_score = 1.0
            self.approach_target = self._current_approach_target_locked()
            self.approach_active = True
            self.event_phase = "p2_approaching"
            self._recompute_locked(time.monotonic())

    def set_actor_position(self, actor_id: str, x: float, y: float) -> None:
        with self.lock:
            actor = self._resolve_actor(actor_id)
            if actor is None:
                return
            actor.x = x
            actor.y = y
            if actor_id == "newcomer":
                self.approach_active = False
                if self.event_phase == "p2_approaching":
                    self.event_phase = "dyad"
            self._recompute_locked(time.monotonic())

    def set_actor_yaw(self, actor_id: str, yaw: float) -> None:
        with self.lock:
            actor = self._resolve_actor(actor_id)
            if actor is None:
                return
            actor.yaw = yaw
            if actor_id == "newcomer" and self.approach_active:
                self.approach_active = False
                if self.event_phase == "p2_approaching":
                    self.event_phase = "dyad"
            self._recompute_locked(time.monotonic())

    def set_affinity(self, actor_id: str, affinity: float) -> None:
        with self.lock:
            actor = self._resolve_actor(actor_id)
            if actor is None or actor_id == "robot":
                return
            actor.affinity = max(0.0, min(1.0, affinity))
            self._recompute_locked(time.monotonic())

    def set_speaker(self, actor_id: str) -> None:
        with self.lock:
            self.primary.speaking_score = 0.0
            self.newcomer.speaking_score = 0.0
            if actor_id == "primary":
                self.primary.speaking_score = 1.0
            elif actor_id == "newcomer":
                self.newcomer.speaking_score = 1.0
            self._recompute_locked(time.monotonic())

    def _resolve_actor(self, actor_id: str) -> Pose2D | PersonState | None:
        if actor_id == "robot":
            return self.robot
        if actor_id == "primary":
            return self.primary
        if actor_id == "newcomer":
            return self.newcomer
        return None

    def _recompute_locked(self, now_s: float) -> None:
        self.latest_plan = compute_dyad_to_triad_plan(
            self.robot,
            self.primary,
            self.newcomer,
            self.params,
        )
        self.last_time = now_s

    def _advance_newcomer_approach_locked(self, dt_s: float) -> None:
        if not self.approach_active or self.approach_target is None:
            return
        dyad_center = self._current_dyad_center_locked()
        target_yaw = math.atan2(dyad_center[1] - self.newcomer.y, dyad_center[0] - self.newcomer.x)
        stepped = step_pose_toward_target(
            Pose2D(self.newcomer.x, self.newcomer.y, self.newcomer.yaw or 0.0),
            self.approach_target,
            target_yaw,
            dt_s,
            move_speed=0.72,
            turn_rate=2.4,
        )
        self.newcomer.x = stepped.x
        self.newcomer.y = stepped.y
        self.newcomer.yaw = stepped.yaw
        if math.hypot(self.newcomer.x - self.approach_target[0], self.newcomer.y - self.approach_target[1]) <= 0.04:
            self.approach_active = False
            self.event_phase = "reconfiguring"

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            plan = self.latest_plan
            dyad_center = (
                (self.robot.x + self.primary.x) * 0.5,
                (self.robot.y + self.primary.y) * 0.5,
            )
            return {
                "running": self.running,
                "summary": format_dyad_event_summary(self.event_phase, plan),
                "robot": {
                    "position": {"x": self.robot.x, "y": self.robot.y},
                    "yaw": self.robot.yaw,
                },
                "primary": {
                    "person_id": self.primary.person_id,
                    "position": {"x": self.primary.x, "y": self.primary.y},
                    "yaw": self.primary.yaw,
                    "speaking_score": self.primary.speaking_score,
                    "affinity": self.primary.affinity,
                },
                "newcomer": {
                    "person_id": self.newcomer.person_id,
                    "position": {"x": self.newcomer.x, "y": self.newcomer.y},
                    "yaw": self.newcomer.yaw,
                    "speaking_score": self.newcomer.speaking_score,
                    "affinity": self.newcomer.affinity,
                },
                "event": {
                    "phase": self.event_phase,
                    "approach_active": self.approach_active,
                    "approach_target_position": (
                        {"x": self.approach_target[0], "y": self.approach_target[1]}
                        if self.approach_target is not None
                        else None
                    ),
                },
                "plan": {
                    "stage": plan.stage,
                    "target_position": {"x": plan.target_position[0], "y": plan.target_position[1]},
                    "body_yaw": plan.body_yaw,
                    "gaze_yaw": plan.gaze_yaw,
                    "focus_person_id": plan.focus_person_id,
                    "joining_score": plan.joining_score,
                    "admitted": plan.admitted,
                    "shared_center": {"x": plan.shared_center[0], "y": plan.shared_center[1]},
                    "dyad_center": {"x": dyad_center[0], "y": dyad_center[1]},
                    "target_primary_distance": plan.target_primary_distance,
                    "target_newcomer_distance": plan.target_newcomer_distance,
                    "same_side_preserved": plan.same_side_preserved,
                    "motion_limit_applied": plan.motion_limit_applied,
                },
                "params": {
                    "admit_distance": self.params.admit_distance,
                    "pre_admit_threshold": self.params.pre_admit_threshold,
                    "admit_threshold": self.params.admit_threshold,
                },
            }

    def _loop(self) -> None:
        while not self.shutdown_event.is_set():
            now_s = time.monotonic()
            with self.lock:
                dt_s = min(0.05, max(0.0, now_s - self.last_time))
                self.last_time = now_s
                if self.running and self.approach_active:
                    self._advance_newcomer_approach_locked(dt_s)
                plan = compute_dyad_to_triad_plan(
                    self.robot,
                    self.primary,
                    self.newcomer,
                    self.params,
                )
                self.latest_plan = plan
                if self.running:
                    if not self.approach_active and self.event_phase in {"reconfiguring", "triad_formed"}:
                        self.robot = step_pose_toward_target(
                            self.robot,
                            plan.target_position,
                            plan.body_yaw,
                            dt_s,
                        )
                        if plan.admitted and plan.stage == "stabilize_triad":
                            self.event_phase = "triad_formed"
            self.shutdown_event.wait(1.0 / 30.0)


class DetourWebController:
    def __init__(self) -> None:
        self.params = SocialNavParams()
        self.lock = threading.Lock()
        self.running = True
        self.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._reset_locked("single_person")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _reset_locked(self, scenario_mode: str | None = None) -> None:
        self.running = True
        self.last_time = time.monotonic()
        self._set_scenario_locked(scenario_mode or getattr(self, "scenario_mode", "single_person"))
        self.latest_plan = self._compute_plan_locked()

    def _set_scenario_locked(self, scenario_mode: str) -> None:
        if scenario_mode == "interactive_group":
            self.scenario_mode = "interactive_group"
            self.scenario_label = "interactive group"
            self.robot = Pose2D(-3.25, 0.0, 0.0)
            self.charge_dock = (3.35, 0.0)
            self.people = demo_people(count=4, speaker_person_id="P2")
            self.speaker_person_id = set_group_speaker(self.people, "P2")
        else:
            self.scenario_mode = "single_person"
            self.scenario_label = "single person"
            self.robot = Pose2D(-3.15, 0.0, 0.0)
            self.charge_dock = (3.25, 0.0)
            self.people = [
                PersonState(
                    person_id="P1",
                    x=0.0,
                    y=0.0,
                    yaw=math.pi,
                    engagement=1.0,
                    speaking_score=0.0,
                    affinity=0.5,
                )
            ]
            self.speaker_person_id = None

        self.robot.yaw = math.atan2(self.charge_dock[1] - self.robot.y, self.charge_dock[0] - self.robot.x)

    def reset(self) -> None:
        with self.lock:
            self._reset_locked(self.scenario_mode)

    def toggle_running(self) -> bool:
        with self.lock:
            self.running = not self.running
            return self.running

    def set_scenario(self, scenario_mode: str) -> None:
        with self.lock:
            self._reset_locked(scenario_mode)

    def _compute_plan_locked(self) -> SocialTransitPlan:
        return compute_social_transit_plan(
            self.robot,
            self.charge_dock,
            self.people,
            self.params,
        )

    def rotate_scene(self) -> None:
        with self.lock:
            if self.scenario_mode == "single_person" and self.people:
                yaw = (self.people[0].yaw or 0.0) + math.radians(45.0)
                while yaw <= -math.pi:
                    yaw += math.tau
                while yaw > math.pi:
                    yaw -= math.tau
                self.people[0].yaw = yaw
            elif self.people:
                person_ids = [person.person_id for person in self.people]
                current_speaker = self.speaker_person_id if self.speaker_person_id in person_ids else person_ids[0]
                next_index = (person_ids.index(current_speaker) + 1) % len(person_ids)
                self.speaker_person_id = set_group_speaker(self.people, person_ids[next_index])
            self.latest_plan = self._compute_plan_locked()

    def move_actor(self, actor_id: str, x: float, y: float) -> None:
        with self.lock:
            if actor_id == "robot":
                self.robot.x = x
                self.robot.y = y
            elif actor_id == "dock":
                self.charge_dock = (x, y)
            else:
                person = next((candidate for candidate in self.people if candidate.person_id == actor_id), None)
                if person is None:
                    return
                person.x = x
                person.y = y
                if self.scenario_mode == "interactive_group":
                    self.speaker_person_id = set_group_speaker(self.people, self.speaker_person_id)
            self.latest_plan = self._compute_plan_locked()

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            plan = self.latest_plan
            return {
                "running": self.running,
                "scenario_mode": self.scenario_mode,
                "scenario_label": self.scenario_label,
                "speaker_person_id": self.speaker_person_id,
                "summary": format_detour_summary(self.scenario_mode, plan),
                "robot": {
                    "position": {"x": self.robot.x, "y": self.robot.y},
                    "yaw": self.robot.yaw,
                },
                "charge_dock": {"x": self.charge_dock[0], "y": self.charge_dock[1]},
                "people": [
                    {
                        "person_id": person.person_id,
                        "position": {"x": person.x, "y": person.y},
                        "yaw": person.yaw,
                        "speaking_score": person.speaking_score,
                        "engagement": person.engagement,
                        "affinity": person.affinity,
                    }
                    for person in self.people
                ],
                "plan": {
                    "stage": plan.stage,
                    "target_position": {"x": plan.target_position[0], "y": plan.target_position[1]},
                    "body_yaw": plan.body_yaw,
                    "gaze_yaw": plan.gaze_yaw,
                    "group_center": {"x": plan.group_center[0], "y": plan.group_center[1]},
                    "group_radius": plan.group_radius,
                    "path_points": [
                        {"x": point[0], "y": point[1]}
                        for point in plan.path_points
                    ],
                    "field_samples": [
                        {"x": sample[0], "y": sample[1], "cost": sample[2]}
                        for sample in plan.field_samples
                    ],
                    "interaction_links": [
                        {"a": link[0], "b": link[1], "probability": link[2]}
                        for link in plan.interaction_links
                    ],
                    "direct_distance": plan.direct_distance,
                    "path_length": plan.path_length,
                    "extra_distance": plan.extra_distance,
                    "distance_to_target": plan.distance_to_target,
                    "min_clearance": plan.min_clearance,
                },
                "params": {
                    "path_grid_resolution": self.params.path_grid_resolution,
                    "path_block_cost": self.params.path_block_cost,
                },
            }

    def _loop(self) -> None:
        while not self.shutdown_event.is_set():
            now_s = time.monotonic()
            with self.lock:
                dt_s = min(0.05, max(0.0, now_s - self.last_time))
                self.last_time = now_s
                self.latest_plan = self._compute_plan_locked()
                if self.running and self.latest_plan.stage != "arrived":
                    self.robot = step_robot_toward_transit_plan(self.robot, self.latest_plan, dt_s, self.params)
                    self.latest_plan = self._compute_plan_locked()
            self.shutdown_event.wait(1.0 / 30.0)


class SocialNavWebRequestHandler(BaseHTTPRequestHandler):
    controller: WebDemoController
    dyad_controller: DyadTriadWebController
    detour_controller: DetourWebController

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path == "/healthz":
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/":
            body = WEB_APP_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/dyad-triad":
            body = DYAD_TRIAD_WEB_APP_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/detour":
            body = DETOUR_WEB_APP_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/state":
            body = json.dumps(self.controller.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/dyad-triad/state":
            body = json.dumps(self.dyad_controller.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/detour/state":
            body = json.dumps(self.detour_controller.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/command", "/dyad-triad/command", "/detour/command"}:
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        data = json.loads(payload or "{}")
        action = data.get("action")

        if path == "/command":
            if action == "toggle_running":
                self.controller.toggle_running()
            elif action == "reset":
                self.controller.reset()
            elif action == "randomize":
                self.controller.randomize_people()
            elif action == "invite_robot":
                self.controller.invite_robot(str(data.get("person_id", "P1")))
            elif action == "set_group_size":
                self.controller.set_group_size(int(data["count"]))
            elif action == "cycle_speaker":
                self.controller.cycle_primary_speaker()
            elif action == "add_person":
                self.controller.add_person(float(data["x"]), float(data["y"]))
            elif action == "move_person":
                self.controller.move_person(str(data["person_id"]), float(data["x"]), float(data["y"]))
            elif action == "remove_person":
                self.controller.remove_person(str(data["person_id"]))
            elif action == "set_affinity":
                self.controller.set_affinity(str(data["person_id"]), float(data["affinity"]))
            else:
                self.send_error(400, "Unknown action")
                return
        elif path == "/dyad-triad/command":
            if action == "toggle_running":
                self.dyad_controller.toggle_running()
            elif action == "trigger_approach":
                self.dyad_controller.trigger_approach()
            elif action == "reset":
                self.dyad_controller.reset()
            elif action == "move_actor":
                self.dyad_controller.set_actor_position(str(data["actor_id"]), float(data["x"]), float(data["y"]))
            elif action == "set_yaw":
                self.dyad_controller.set_actor_yaw(str(data["actor_id"]), float(data["yaw"]))
            elif action == "set_affinity":
                self.dyad_controller.set_affinity(str(data["actor_id"]), float(data["affinity"]))
            elif action == "set_speaker":
                self.dyad_controller.set_speaker(str(data["actor_id"]))
            else:
                self.send_error(400, "Unknown action")
                return
        else:
            if action == "toggle_running":
                self.detour_controller.toggle_running()
            elif action == "reset":
                self.detour_controller.reset()
            elif action == "set_scenario":
                self.detour_controller.set_scenario(str(data["mode"]))
            elif action == "rotate_scene":
                self.detour_controller.rotate_scene()
            elif action == "move_actor":
                self.detour_controller.move_actor(str(data["actor_id"]), float(data["x"]), float(data["y"]))
            else:
                self.send_error(400, "Unknown action")
                return

        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def run_web(port: int = 8000, host: str = "127.0.0.1") -> None:
    controller = WebDemoController()
    dyad_controller = DyadTriadWebController()
    detour_controller = DetourWebController()
    controller.start()
    dyad_controller.start()
    detour_controller.start()
    handler_type = type(
        "ConfiguredSocialNavWebRequestHandler",
        (SocialNavWebRequestHandler,),
        {
            "controller": controller,
            "dyad_controller": dyad_controller,
            "detour_controller": detour_controller,
        },
    )
    server = ReusableThreadingHTTPServer((host, port), handler_type)
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Web demo running at http://{display_host}:{port} (bound to {host})")
    print(f"Dyad -> triad demo at http://{display_host}:{port}/dyad-triad")
    print(f"Detour to dock demo at http://{display_host}:{port}/detour")
    print("Open that address in your browser. Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        controller.stop()
        dyad_controller.stop()
        detour_controller.stop()


def parse_args() -> argparse.Namespace:
    default_port = 8000
    port_from_env = os.environ.get("PORT")
    if port_from_env:
        try:
            default_port = int(port_from_env)
        except ValueError:
            pass

    parser = argparse.ArgumentParser(description="Interactive demo for N-person social navigation.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--headless", action="store_true", help="Run a text-only simulation instead of opening Tk.")
    mode_group.add_argument("--web", action="store_true", help="Run a browser-based real-time visualization.")
    parser.add_argument("--steps", type=int, default=30, help="Number of simulation steps for headless mode.")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host interface used by the browser demo.",
    )
    parser.add_argument("--port", type=int, default=default_port, help="Port used by the browser demo.")
    return parser.parse_args()


def detect_gui_startup_issue() -> str | None:
    if tk is None:
        return "tkinter is unavailable in this Python environment."

    if platform.system() != "Darwin":
        return None

    using_apple_clt_python = sys.executable.startswith("/Library/Developer/CommandLineTools/")
    using_legacy_system_tk = tk.TkVersion < 8.6
    if using_apple_clt_python and using_legacy_system_tk:
        return (
            "This macOS Python build uses the legacy system Tk 8.5 runtime, "
            "which is crashing while opening GUI windows on this machine."
        )

    return None


def main() -> None:
    args = parse_args()
    if args.web:
        run_web(port=args.port, host=args.host)
        return
    if args.headless:
        run_headless(steps=args.steps)
        return
    startup_issue = detect_gui_startup_issue()
    if startup_issue:
        print(startup_issue)
        print("GUI mode is disabled for this interpreter.")
        print("Use `python3 social_navigation_demo.py --web` for a browser-based live view.")
        print("Use `python3 social_navigation_demo.py --headless` for terminal output only.")
        run_headless(steps=args.steps)
        return
    try:
        demo = SocialNavigationDemo()
    except Exception as exc:
        gui_startup_failed = isinstance(exc, RuntimeError) or (tk is not None and isinstance(exc, tk.TclError))
        if not gui_startup_failed:
            raise
        print(f"Unable to open the Tk demo window: {exc}")
        print("Falling back to headless mode. Use `python3 social_navigation_demo.py --web` for a browser-based live view.")
        run_headless(steps=args.steps)
        return
    demo.run()


if __name__ == "__main__":
    main()
