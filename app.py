from dataclasses import dataclass
from typing import Tuple, FrozenSet, List, Dict, Optional, Any
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging

# ============================================================
# CONFIG
# ============================================================

logger = logging.getLogger(__name__)

BEAM_WIDTH = 10
COOLDOWN_ON_USE = 10
INITIAL_HP = 49
MAX_SUCCESSORS_PER_PARENT = 4

PREDICTION_WINDOW = 8
ROLLING_STEP = PREDICTION_WINDOW // 2  # 4

DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}

OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}

WAIT_PENALTY = 0.35
D_HP_COST_PENALTY = 0.10
WALL_COUNT_PENALTY = 0.03
SPAWN_CELL_PENALTY = 6.0
ADJACENT_TO_SPAWN_PENALTY = 1.5

HP_BUDGET_POSITIVE_WEIGHT = 2.0
HP_BUDGET_NEGATIVE_WEIGHT = 8.0

PUSH_MOVES_AGENT = True

DEFAULT_GRID = [
    "#####################",
    "#G......#.....#######",
    "#.#.#.###.#.#.#######",
    "#.#.......#.....#####",
    "#.#.###.#.###.#.#####",
    "#.............#...###",
    "#.###.#.###.#.#.#.###",
    "#...#.#.....#.......#",
    "###.#.#.###.#.#.###.#",
    "#.......#.....#.#...#",
    "#.###.#.#.###.#.#.#.#",
    "#.....#.....#.......#",
    "#.#.#.#.#.#.#.###.#.#",
    "#...#...#.#.......#.#",
    "###.###.#.###.#.#.#.#",
    "###.................#",
    "#####.#.#.#.#.#.###.#",
    "####...#.....#......#",
    "#######.#.###.#.#.#.#",
    "#######............S#",
    "#####################"
]

DEFAULT_START = (19, 19)
DEFAULT_GOAL = (1, 1)
DEFAULT_ENEMY = {
    "N": "O",
    "S": "Se",
    "W": "P",
    "E": "Sa",
}

MIN_TURNS_CACHE: Dict[Tuple[Tuple[int, int], int, FrozenSet[Tuple[int, int]]], int] = {}


MAX_GRID_HEIGHT = 30
MAX_GRID_WIDTH = 30
MAX_MOBILE_WALLS = 50
MAX_HP = 100
MAX_COOLDOWN = 20

ALLOWED_ENEMY_KEYS = {"N", "S", "W", "E"}
ALLOWED_ENEMY_VALUES = {"P", "Se", "Sa", "O"}
ALLOWED_PREDICTION_VALUES = {"Sang", "Sel", "Or"}

BFS_DISTANCE_CACHE = {}
SWAP_CACHE = {}

# ============================================================
# STRUCTURES INTERNES
# ============================================================

@dataclass(frozen=True)
class State:
    pos: Tuple[int, int]
    hp: int
    cd_b: int
    cd_c: int
    walls: FrozenSet[Tuple[int, int]]

# ============================================================
# MODELES API
# ============================================================

class StateModel(BaseModel):
    pos: List[int] = Field(..., example=[18, 18])
    hp: int = Field(..., example=40)
    cd_b: int = Field(..., example=0)
    cd_c: int = Field(..., example=0)
    walls: List[List[int]] = Field(default_factory=list, example=[[17, 18], [15, 14]])

class GameContext(BaseModel):
    grid: List[str] = Field(default_factory=lambda: DEFAULT_GRID.copy())
    goal: List[int] = Field(default_factory=lambda: list(DEFAULT_GOAL))
    state: StateModel
    enemy: Dict[str, str]
    prediction_buffer: List[str]

class SolveOnlyResponse(BaseModel):
    ok: bool
    plan_full: List[str]
    plan_now: List[str]
    debug: Dict[str, Any]

class TurnLog(BaseModel):
    turn_index: int
    info: str
    action: str
    state_before_tick: Dict[str, Any]
    state_after_tick: Dict[str, Any]
    state_after_action: Dict[str, Any]
    enemy_after_swap: Dict[str, str]
    spawn_pos: List[int]
    state_after_spawn: Dict[str, Any]
    budget_after_spawn: int
    need_after_spawn: int

class NextBlockResponse(BaseModel):
    ok: bool
    plan_full: List[str]
    plan_now: List[str]
    turns_played: int
    new_state: StateModel
    new_enemy: Dict[str, str]
    remaining_prediction_buffer: List[str]
    logs: List[TurnLog]
    debug: Dict[str, Any]

# ============================================================
# OUTILS DE CONVERSION
# ============================================================

def model_to_state(m: StateModel) -> State:
    return State(
        pos=(m.pos[0], m.pos[1]),
        hp=m.hp,
        cd_b=m.cd_b,
        cd_c=m.cd_c,
        walls=frozenset((w[0], w[1]) for w in m.walls),
    )

def state_to_model(s: State) -> StateModel:
    return StateModel(
        pos=[s.pos[0], s.pos[1]],
        hp=s.hp,
        cd_b=s.cd_b,
        cd_c=s.cd_c,
        walls=sorted([[r, c] for (r, c) in s.walls]),
    )

def state_to_dict(s: State) -> Dict[str, Any]:
    return {
        "pos": [s.pos[0], s.pos[1]],
        "hp": s.hp,
        "cd_b": s.cd_b,
        "cd_c": s.cd_c,
        "walls": sorted([[r, c] for (r, c) in s.walls]),
    }

# ============================================================
# OUTILS DE BASE
# ============================================================

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def in_bounds(grid, p):
    return 0 <= p[0] < len(grid) and 0 <= p[1] < len(grid[0])

def is_fixed_wall(grid, p):
    return in_bounds(grid, p) and grid[p[0]][p[1]] == "#"

def is_blocked(grid, p, walls):
    return (not in_bounds(grid, p)) or is_fixed_wall(grid, p) or p in walls

def bfs_distance(grid: List[str], start: Tuple[int, int], goal: Tuple[int, int], walls: set[Tuple[int, int]]) -> int:
    key = (start, goal, frozenset(walls))
    if key in BFS_DISTANCE_CACHE:
        return BFS_DISTANCE_CACHE[key]

    if start == goal:
        BFS_DISTANCE_CACHE[key] = 0
        return 0

    q = deque([(start, 0)])
    seen = {start}

    while q:
        p, d = q.popleft()
        for vec in DIRS.values():
            n = add(p, vec)
            if n in seen or is_blocked(grid, n, walls):
                continue
            if n == goal:
                BFS_DISTANCE_CACHE[key] = d + 1
                return d + 1
            seen.add(n)
            q.append((n, d + 1))

    BFS_DISTANCE_CACHE[key] = 999
    return 999

# ============================================================
# ESTIMATION DU BUDGET PV
# ============================================================

def min_turns_to_goal_no_d(grid: List[str], start: Tuple[int, int], goal: Tuple[int, int],
                           walls: FrozenSet[Tuple[int, int]], cd_b: int) -> int:
    key = (start, cd_b, walls)
    if key in MIN_TURNS_CACHE:
        return MIN_TURNS_CACHE[key]

    if start == goal:
        MIN_TURNS_CACHE[key] = 0
        return 0

    q = deque([(start, cd_b, 0)])
    seen = {(start, cd_b)}

    while q:
        pos, cur_cd_b, turns = q.popleft()

        ready_cd_b = max(0, cur_cd_b - 1)

        for vec in DIRS.values():
            n1 = add(pos, vec)
            if not is_blocked(grid, n1, walls):
                nxt = (n1, ready_cd_b)
                if n1 == goal:
                    MIN_TURNS_CACHE[key] = turns + 1
                    return turns + 1
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((n1, ready_cd_b, turns + 1))

        if ready_cd_b == 0:
            for d1 in DIRS.values():
                p1 = add(pos, d1)
                if is_blocked(grid, p1, walls):
                    continue
                for d2 in DIRS.values():
                    p2 = add(p1, d2)
                    if is_blocked(grid, p2, walls):
                        continue
                    nxt = (p2, COOLDOWN_ON_USE)
                    if p2 == goal:
                        MIN_TURNS_CACHE[key] = turns + 1
                        return turns + 1
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append((p2, COOLDOWN_ON_USE, turns + 1))

    MIN_TURNS_CACHE[key] = 999
    return 999

def hp_needed_to_finish(grid: List[str], state: State, goal: Tuple[int, int]) -> int:
    return min_turns_to_goal_no_d(grid, state.pos, goal, state.walls, state.cd_b)

def hp_budget_margin(grid: List[str], state: State, goal: Tuple[int, int]) -> int:
    needed = hp_needed_to_finish(grid, state, goal)
    if needed >= 999:
        return -999
    return state.hp - needed

# ============================================================
# ADVERSAIRE
# ============================================================

def swap(enemy: Dict[str, str], info: Optional[str]) -> Dict[str, str]:
    key = (tuple(sorted(enemy.items())), info)
    if key in SWAP_CACHE:
        return SWAP_CACHE[key]

    if info in (None, ".", ""):
        res = enemy.copy()
        SWAP_CACHE[key] = res
        return res

    mapping = {"Sang": "Sa", "Sel": "Se", "Or": "O"}
    if info not in mapping:
        raise ValueError(f"Info adverse invalide: {info!r}")

    target = mapping[info]

    d = enemy.copy()
    p_pos = next(k for k, v in d.items() if v == "P")
    t_pos = next(k for k, v in d.items() if v == target)

    d[p_pos], d[t_pos] = d[t_pos], d[p_pos]

    SWAP_CACHE[key] = d
    return d

def get_principal_dir(enemy: Dict[str, str]) -> str:
    return next(k for k, v in enemy.items() if v == "P")

def get_principal_abs_pos(state: State, enemy: Dict[str, str]) -> Tuple[int, int]:
    return add(state.pos, DIRS[get_principal_dir(enemy)])

def get_spawn_pos(state: State, enemy: Dict[str, str]) -> Tuple[int, int]:
    p_dir = get_principal_dir(enemy)
    spawn_dir = OPPOSITE[p_dir]
    return add(state.pos, DIRS[spawn_dir])

def spawn_wall(state: State, enemy: Dict[str, str], grid: List[str]) -> State:
    target = get_spawn_pos(state, enemy)

    if is_blocked(grid, target, state.walls):
        return state

    new_walls = set(state.walls)
    new_walls.add(target)

    return State(
        pos=state.pos,
        hp=state.hp,
        cd_b=state.cd_b,
        cd_c=state.cd_c,
        walls=frozenset(new_walls),
    )

# ============================================================
# ACTIONS
# ============================================================

def enumerate_actions(grid: List[str], state: State):
    res = []

    candidate_reductions = [0]
    if state.cd_b > 0:
        candidate_reductions.append(state.cd_b)
    if state.cd_c > 0:
        candidate_reductions.append(state.cd_c)

    for reduce_n in sorted(set(candidate_reductions)):
        if reduce_n >= state.hp:
            continue
        hp2 = state.hp - reduce_n
        cd_b2 = max(0, state.cd_b - reduce_n)
        cd_c2 = max(0, state.cd_c - reduce_n)

        prefix = f"D x{reduce_n} + " if reduce_n > 0 else ""
        d_penalty = reduce_n * D_HP_COST_PENALTY

        base_state = State(
            pos=state.pos,
            hp=hp2,
            cd_b=cd_b2,
            cd_c=cd_c2,
            walls=state.walls
        )

        res.append((
            prefix + "WAIT",
            base_state,
            d_penalty + WAIT_PENALTY
        ))

        for d, vec in DIRS.items():
            n = add(base_state.pos, vec)
            if not is_blocked(grid, n, base_state.walls):
                res.append((
                    prefix + f"MOVE {d}",
                    State(n, base_state.hp, base_state.cd_b, base_state.cd_c, base_state.walls),
                    d_penalty
                ))

        if base_state.cd_b == 0:
            for d1 in DIRS:
                p1 = add(base_state.pos, DIRS[d1])
                if is_blocked(grid, p1, base_state.walls):
                    continue
                for d2 in DIRS:
                    p2 = add(p1, DIRS[d2])
                    if is_blocked(grid, p2, base_state.walls):
                        continue
                    res.append((
                        prefix + f"MOVE2 {d1}{d2}",
                        State(p2, base_state.hp, COOLDOWN_ON_USE, base_state.cd_c, base_state.walls),
                        d_penalty
                    ))

        if base_state.cd_c == 0:
            for d, vec in DIRS.items():
                wall_pos = add(base_state.pos, vec)
                if wall_pos not in base_state.walls:
                    continue

                behind = add(wall_pos, vec)
                if is_blocked(grid, behind, base_state.walls):
                    continue

                new_walls = set(base_state.walls)
                new_walls.remove(wall_pos)
                new_walls.add(behind)

                new_pos = wall_pos if PUSH_MOVES_AGENT else base_state.pos

                if new_pos in new_walls or is_fixed_wall(grid, new_pos):
                    continue

                res.append((
                    prefix + f"PUSH {d}",
                    State(new_pos, base_state.hp, base_state.cd_b, COOLDOWN_ON_USE, frozenset(new_walls)),
                    d_penalty
                ))

    return res

# ============================================================
# VALIDATION / APPLICATION D'ACTION
# ============================================================

def validate_action(grid: List[str], state: State, action: str) -> None:
    label = action.strip()

    reduce_n = 0
    if label.startswith("D x"):
        left, right = label.split("+", 1)
        reduce_n = int(left.strip()[3:])
        label = right.strip()

        if reduce_n > max(state.cd_b, state.cd_c):
            raise ValueError(f"Réduction D invalide: {action} depuis cdB={state.cd_b}, cdC={state.cd_c}")
        if reduce_n >= state.hp:
            raise ValueError(f"Réduction D invalide: pas assez de PV pour {action}")

    hp2 = state.hp - reduce_n
    cd_b2 = max(0, state.cd_b - reduce_n)
    cd_c2 = max(0, state.cd_c - reduce_n)

    tmp = State(state.pos, hp2, cd_b2, cd_c2, state.walls)

    if label == "WAIT":
        return

    if label.startswith("MOVE2 "):
        if tmp.cd_b != 0:
            raise ValueError(f"MOVE2 impossible: cooldown B non prêt pour {action}")

        seq = label.split()[1]
        if len(seq) != 2 or any(ch not in DIRS for ch in seq):
            raise ValueError(f"MOVE2 invalide: {action}")

        pos = tmp.pos
        for d in seq:
            pos = add(pos, DIRS[d])
            if is_blocked(grid, pos, tmp.walls):
                raise ValueError(f"MOVE2 traverse une case bloquée: {action}, blocage sur {pos}")
        return

    if label.startswith("MOVE "):
        d = label.split()[1]
        if d not in DIRS:
            raise ValueError(f"MOVE invalide: {action}")
        nxt = add(tmp.pos, DIRS[d])
        if is_blocked(grid, nxt, tmp.walls):
            raise ValueError(f"MOVE sur case bloquée: {action}, cible={nxt}")
        return

    if label.startswith("PUSH "):
        if tmp.cd_c != 0:
            raise ValueError(f"PUSH impossible: cooldown C non prêt pour {action}")

        d = label.split()[1]
        if d not in DIRS:
            raise ValueError(f"PUSH invalide: {action}")

        vec = DIRS[d]
        wall_pos = add(tmp.pos, vec)
        if wall_pos not in tmp.walls:
            raise ValueError(f"PUSH invalide: pas de mur mobile à pousser sur {wall_pos}")

        behind = add(wall_pos, vec)
        if is_blocked(grid, behind, tmp.walls):
            raise ValueError(f"PUSH impossible: destination du mur bloquée sur {behind}")
        return

    raise ValueError(f"Action inconnue: {action}")

def apply_action(grid: List[str], state: State, action: str) -> State:
    validate_action(grid, state, action)

    label = action.strip()

    reduce_n = 0
    if label.startswith("D x"):
        left, right = label.split("+", 1)
        reduce_n = int(left.strip()[3:])
        label = right.strip()

    hp2 = state.hp - reduce_n
    cd_b2 = max(0, state.cd_b - reduce_n)
    cd_c2 = max(0, state.cd_c - reduce_n)

    cur = State(state.pos, hp2, cd_b2, cd_c2, state.walls)

    if label == "WAIT":
        return cur

    if label.startswith("MOVE2 "):
        seq = label.split()[1]
        pos = cur.pos
        for d in seq:
            pos = add(pos, DIRS[d])
        return State(pos, cur.hp, COOLDOWN_ON_USE, cur.cd_c, cur.walls)

    if label.startswith("MOVE "):
        d = label.split()[1]
        return State(add(cur.pos, DIRS[d]), cur.hp, cur.cd_b, cur.cd_c, cur.walls)

    if label.startswith("PUSH "):
        d = label.split()[1]
        vec = DIRS[d]
        wall_pos = add(cur.pos, vec)
        behind = add(wall_pos, vec)

        new_walls = set(cur.walls)
        new_walls.remove(wall_pos)
        new_walls.add(behind)

        new_pos = wall_pos if PUSH_MOVES_AGENT else cur.pos
        return State(new_pos, cur.hp, cur.cd_b, COOLDOWN_ON_USE, frozenset(new_walls))

    raise ValueError(f"Action non reconnue: {action!r}")

# ============================================================
# HEURISTIQUES / ANTI-PIÈGE
# ============================================================

def future_spawn_penalty(state: State, enemy: Dict[str, str], grid: List[str]) -> float:
    spawn = get_spawn_pos(state, enemy)
    penalty = 0.0

    if in_bounds(grid, spawn):
        if state.pos == spawn:
            penalty -= SPAWN_CELL_PENALTY

        for vec in DIRS.values():
            if add(state.pos, vec) == spawn:
                penalty -= ADJACENT_TO_SPAWN_PENALTY

    return penalty

def budget_score(grid: List[str], state: State, goal: Tuple[int, int]) -> Tuple[int, float]:
    margin = hp_budget_margin(grid, state, goal)
    feasible = 1 if margin >= 0 else 0

    if margin >= 0:
        weighted = margin * HP_BUDGET_POSITIVE_WEIGHT
    else:
        weighted = margin * HP_BUDGET_NEGATIVE_WEIGHT

    return feasible, weighted

def score_state(grid: List[str], state: State, goal: Tuple[int, int], enemy: Dict[str, str]) -> Tuple[float, float, float, float, float]:
    dist = bfs_distance(grid, state.pos, goal, state.walls)
    budget_feasible, budget_weighted = budget_score(grid, state, goal)

    return (
        float(budget_feasible),
        budget_weighted,
        -float(dist),
        future_spawn_penalty(state, enemy, grid),
        -len(state.walls) * WALL_COUNT_PENALTY,
    )

# ============================================================
# SOLVER
# ============================================================

def solve(grid: List[str], init_state: State, goal: Tuple[int, int], enemy_infos: List[str], enemy_init: Dict[str, str]):
    beam = [(init_state, enemy_init, [])]

    for t in range(len(enemy_infos)):
        new_beam = []

        for state, enemy, path in beam:
            state = State(
                state.pos,
                state.hp - 1,
                max(0, state.cd_b - 1),
                max(0, state.cd_c - 1),
                state.walls
            )

            if state.hp <= 0:
                continue

            if hp_needed_to_finish(grid, state, goal) > state.hp:
                continue

            parent_candidates = []

            for action_label, after_action_state, local_penalty in enumerate_actions(grid, state):
                enemy2 = swap(enemy, enemy_infos[t])
                after_spawn_state = spawn_wall(after_action_state, enemy2, grid)

                if hp_needed_to_finish(grid, after_spawn_state, goal) > after_spawn_state.hp:
                    continue

                base_score = score_state(grid, after_spawn_state, goal, enemy2)

                combined_score = (
                    base_score[0],
                    base_score[1] - local_penalty,
                    base_score[2],
                    base_score[3],
                    base_score[4],
                )

                parent_candidates.append((combined_score, after_spawn_state, enemy2, path + [action_label]))

            parent_candidates.sort(key=lambda x: x[0], reverse=True)
            new_beam.extend(parent_candidates[:MAX_SUCCESSORS_PER_PARENT])

        if not new_beam:
            return None

        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = [(s, e, p) for (_, s, e, p) in new_beam[:BEAM_WIDTH]]

        best_state = beam[0][0]
        if best_state.pos == goal:
            return beam[0]

    return beam[0]

# ============================================================
# EXECUTION D'UN BLOC
# ============================================================

def play_turns_api(grid: List[str],
                   state: State,
                   enemy: Dict[str, str],
                   enemy_infos: List[str],
                   plan: List[str],
                   goal: Tuple[int, int],
                   turns_to_play: int):
    turns_to_play = min(turns_to_play, len(plan), len(enemy_infos))
    logs: List[TurnLog] = []

    for i in range(turns_to_play):
        state_before_tick = state

        state = State(
            state.pos,
            state.hp - 1,
            max(0, state.cd_b - 1),
            max(0, state.cd_c - 1),
            state.walls
        )

        if state.hp <= 0:
            break

        state_after_tick = state

        action = plan[i]
        state = apply_action(grid, state, action)
        state_after_action = state

        info = enemy_infos[i]
        enemy = swap(enemy, info)
        enemy_after_swap = enemy.copy()

        spawn_pos = get_spawn_pos(state, enemy)
        state = spawn_wall(state, enemy, grid)
        state_after_spawn = state

        logs.append(TurnLog(
            turn_index=i + 1,
            info=info,
            action=action,
            state_before_tick=state_to_dict(state_before_tick),
            state_after_tick=state_to_dict(state_after_tick),
            state_after_action=state_to_dict(state_after_action),
            enemy_after_swap=enemy_after_swap,
            spawn_pos=[spawn_pos[0], spawn_pos[1]],
            state_after_spawn=state_to_dict(state_after_spawn),
            budget_after_spawn=hp_budget_margin(grid, state_after_spawn, goal),
            need_after_spawn=hp_needed_to_finish(grid, state_after_spawn, goal),
        ))

        if state.pos == goal:
            break

    return state, enemy, logs



# ============================================================
# VALIDATION CONTEXTE API
# ============================================================

def validate_coord_list(name: str, value: List[int]) -> Tuple[int, int]:
    if len(value) != 2:
        raise HTTPException(status_code=400, detail=f"{name} doit contenir exactement 2 entiers")
    r, c = value[0], value[1]
    if not isinstance(r, int) or not isinstance(c, int):
        raise HTTPException(status_code=400, detail=f"{name} doit contenir exactement 2 entiers")
    return (r, c)

def validate_grid(grid: List[str]) -> None:
    if not grid:
        raise HTTPException(status_code=400, detail="grid ne doit pas être vide")

    if len(grid) > MAX_GRID_HEIGHT:
        raise HTTPException(
            status_code=400,
            detail=f"grid dépasse la hauteur maximale autorisée ({MAX_GRID_HEIGHT})"
        )

    width = len(grid[0])
    if width == 0:
        raise HTTPException(status_code=400, detail="grid ne doit pas contenir de ligne vide")

    if width > MAX_GRID_WIDTH:
        raise HTTPException(
            status_code=400,
            detail=f"grid dépasse la largeur maximale autorisée ({MAX_GRID_WIDTH})"
        )

    for i, row in enumerate(grid):
        if len(row) != width:
            raise HTTPException(
                status_code=400,
                detail=f"toutes les lignes de grid doivent avoir la même longueur (ligne 0={width}, ligne {i}={len(row)})"
            )

def validate_enemy(enemy: Dict[str, str]) -> None:
    keys = set(enemy.keys())
    values = set(enemy.values())

    if keys != ALLOWED_ENEMY_KEYS:
        raise HTTPException(
            status_code=400,
            detail="enemy doit contenir exactement les clés N, S, W, E"
        )

    if values != ALLOWED_ENEMY_VALUES or len(enemy) != 4:
        raise HTTPException(
            status_code=400,
            detail="enemy doit contenir exactement les valeurs P, Se, Sa, O"
        )

def validate_prediction_buffer(prediction_buffer: List[str]) -> None:
    if len(prediction_buffer) != PREDICTION_WINDOW:
        raise HTTPException(
            status_code=400,
            detail=f"prediction_buffer doit contenir exactement {PREDICTION_WINDOW} éléments"
        )

    invalid = [x for x in prediction_buffer if x not in ALLOWED_PREDICTION_VALUES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail="prediction_buffer ne doit contenir que Sang, Sel ou Or"
        )

def validate_state_and_goal(grid: List[str], state: State, goal: Tuple[int, int]) -> None:
    if state.hp < 0 or state.hp > MAX_HP:
        raise HTTPException(
            status_code=400,
            detail=f"state.hp doit être compris entre 0 et {MAX_HP}"
        )

    if state.cd_b < 0 or state.cd_b > MAX_COOLDOWN:
        raise HTTPException(
            status_code=400,
            detail=f"state.cd_b doit être compris entre 0 et {MAX_COOLDOWN}"
        )

    if state.cd_c < 0 or state.cd_c > MAX_COOLDOWN:
        raise HTTPException(
            status_code=400,
            detail=f"state.cd_c doit être compris entre 0 et {MAX_COOLDOWN}"
        )

    if len(state.walls) > MAX_MOBILE_WALLS:
        raise HTTPException(
            status_code=400,
            detail=f"state.walls ne doit pas contenir plus de {MAX_MOBILE_WALLS} murs mobiles"
        )

    if not in_bounds(grid, state.pos):
        raise HTTPException(status_code=400, detail="state.pos doit être dans la grille")

    if not in_bounds(grid, goal):
        raise HTTPException(status_code=400, detail="goal doit être dans la grille")

    if is_fixed_wall(grid, state.pos):
        raise HTTPException(status_code=400, detail="state.pos ne peut pas être sur un mur fixe")

    if is_fixed_wall(grid, goal):
        raise HTTPException(status_code=400, detail="goal ne peut pas être sur un mur fixe")

    for wall in state.walls:
        if not in_bounds(grid, wall):
            raise HTTPException(
                status_code=400,
                detail=f"un mur mobile est hors grille: {list(wall)}"
            )
        if is_fixed_wall(grid, wall):
            raise HTTPException(
                status_code=400,
                detail=f"un mur mobile ne peut pas être sur un mur fixe: {list(wall)}"
            )
        if wall == state.pos:
            raise HTTPException(
                status_code=400,
                detail=f"un mur mobile ne peut pas être sur state.pos: {list(wall)}"
            )
        if wall == goal:
            raise HTTPException(
                status_code=400,
                detail=f"un mur mobile ne peut pas être sur goal: {list(wall)}"
            )

def validate_game_context(ctx: GameContext) -> Tuple[State, Tuple[int, int]]:
    validate_grid(ctx.grid)
    validate_enemy(ctx.enemy)
    validate_prediction_buffer(ctx.prediction_buffer)

    goal = validate_coord_list("goal", ctx.goal)
    state = model_to_state(ctx.state)

    if len(ctx.state.pos) != 2:
        raise HTTPException(status_code=400, detail="state.pos doit contenir exactement 2 entiers")

    for i, wall in enumerate(ctx.state.walls):
        if len(wall) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"state.walls[{i}] doit contenir exactement 2 entiers"
            )

    validate_state_and_goal(ctx.grid, state, goal)
    return state, goal

# ============================================================
# APP FASTAPI
# ============================================================

app = FastAPI(title="Mino Solver API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"
#        "https://emmathie.fr",
#        "https://www.emmathie.fr",
#        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["emmathie.fr", "www.emmathie.fr", "127.0.0.1", "localhost"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/config")
def config():
    return {
        "ok": True,
        "beam_width": BEAM_WIDTH,
        "cooldown_on_use": COOLDOWN_ON_USE,
        "initial_hp": INITIAL_HP,
        "prediction_window": PREDICTION_WINDOW,
        "rolling_step": ROLLING_STEP,
        "default_grid": DEFAULT_GRID,
        "default_start": list(DEFAULT_START),
        "default_goal": list(DEFAULT_GOAL),
        "default_enemy": DEFAULT_ENEMY,
    }

@app.post("/solve", response_model=SolveOnlyResponse)
def solve_endpoint(ctx: GameContext):
    try:
        state, goal = validate_game_context(ctx)

        result = solve(ctx.grid, state, goal, ctx.prediction_buffer, ctx.enemy)
        if result is None:
            raise HTTPException(status_code=400, detail="Aucun plan trouvé")

        best_state, best_enemy, plan = result

        return SolveOnlyResponse(
            ok=True,
            plan_full=plan,
            plan_now=plan[:ROLLING_STEP],
            debug={
                "best_state": state_to_dict(best_state),
                "best_enemy": best_enemy,
                "best_need": hp_needed_to_finish(ctx.grid, best_state, goal),
                "best_budget": hp_budget_margin(ctx.grid, best_state, goal),
                "best_distance": bfs_distance(ctx.grid, best_state.pos, goal, best_state.walls),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /next-block: " + str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/next-block", response_model=NextBlockResponse)
def next_block_endpoint(ctx: GameContext):
    try:
        state, goal = validate_game_context(ctx)

        result = solve(ctx.grid, state, goal, ctx.prediction_buffer, ctx.enemy)
        if result is None:
            raise HTTPException(status_code=400, detail="Aucun plan trouvé")

        best_state, best_enemy, plan = result

        new_state, new_enemy, logs = play_turns_api(
            grid=ctx.grid,
            state=state,
            enemy=ctx.enemy,
            enemy_infos=ctx.prediction_buffer,
            plan=plan,
            goal=goal,
            turns_to_play=ROLLING_STEP,
        )

        remaining_prediction_buffer = ctx.prediction_buffer[ROLLING_STEP:]

        return NextBlockResponse(
            ok=True,
            plan_full=plan,
            plan_now=plan[:ROLLING_STEP],
            turns_played=min(ROLLING_STEP, len(plan)),
            new_state=state_to_model(new_state),
            new_enemy=new_enemy,
            remaining_prediction_buffer=remaining_prediction_buffer,
            logs=logs,
            debug={
                "best_state": state_to_dict(best_state),
                "best_enemy": best_enemy,
                "best_need": hp_needed_to_finish(ctx.grid, best_state, goal),
                "best_budget": hp_budget_margin(ctx.grid, best_state, goal),
                "best_distance": bfs_distance(ctx.grid, best_state.pos, goal, best_state.walls),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /next-block: " + str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
