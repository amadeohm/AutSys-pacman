# my_team.py
# Improved Capture the Flag team

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

###############################################################################
# Constants
###############################################################################
GHOST_THREAT_RADIUS  = 7
SCARED_CHASE_RADIUS  = 8
CARRY_THRESHOLD      = 4
PATROL_DEPTH         = 3
DEADEND_DEPTH        = 6
ENDGAME_MARGIN       = 8

###############################################################################
# Team creation
###############################################################################

def create_team(first_index, second_index, is_red,
                first='ImprovedOffensiveAgent', second='ImprovedDefensiveAgent',
                num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]

###############################################################################
# Base agent
###############################################################################

class BaseAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        walls = game_state.get_walls()
        self.width  = walls.width
        self.height = walls.height
        mid = self.width // 2
        self.home_x  = mid - 1 if self.red else mid
        self.enemy_x = mid     if self.red else mid - 1
        self._precompute_border(game_state)
        self._precompute_deadends(game_state)

    # ── Border helpers ──────────────────────────────────────────────────

    def _precompute_border(self, game_state):
        walls = game_state.get_walls()
        self.border_cells = [(self.home_x, y)
                             for y in range(self.height)
                             if not walls[self.home_x][y]]

    def dist_to_home(self, pos):
        return min(self.get_maze_distance(pos, p) for p in self.border_cells)

    def nearest_home_cell(self, pos):
        return min(self.border_cells, key=lambda p: self.get_maze_distance(pos, p))

    # ── Dead-end detection ──────────────────────────────────────────────

    def _precompute_deadends(self, game_state):
        walls = game_state.get_walls()
        all_cells = [(x, y) for x in range(self.width) for y in range(self.height)
                     if not walls[x][y]]
        self.deadend = set()
        for cell in all_cells:
            if len(self._bfs_limited(cell, walls, DEADEND_DEPTH)) <= DEADEND_DEPTH:
                self.deadend.add(cell)

    def _bfs_limited(self, start, walls, max_depth):
        visited = {start}
        queue = util.Queue()
        queue.push((start, 0))
        while not queue.is_empty():
            (x, y), d = queue.pop()
            if d >= max_depth:
                continue
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and not walls[nx][ny] and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.push(((nx, ny), d+1))
        return visited

    # ── A* pathfinding ──────────────────────────────────────────────────

    def astar_action(self, game_state, start, targets, forbidden=None):
        """
        A* from start to the nearest cell in targets.
        Heuristic: min Manhattan distance to any target.
        forbidden: set of positions to avoid (e.g. ghost cells).
        Returns the first action to take, or None if unreachable.
        """
        if forbidden is None:
            forbidden = set()
        walls = game_state.get_walls()

        if start in targets:
            return Directions.STOP

        def heuristic(pos):
            return min(abs(pos[0]-t[0]) + abs(pos[1]-t[1]) for t in targets)

        # Priority queue: (f, g, pos, first_action)
        pq = util.PriorityQueue()
        pq.push((start, None), heuristic(start))
        visited = {}   # pos -> best g seen

        g_map = {start: 0}

        while not pq.is_empty():
            (pos, first_action) = pq.pop()
            g = g_map.get(pos, 0)

            if pos in visited and visited[pos] <= g:
                continue
            visited[pos] = g

            if pos in targets:
                return first_action

            x, y = pos
            for dx, dy, action in [(1,0,Directions.EAST),(-1,0,Directions.WEST),
                                    (0,1,Directions.NORTH),(0,-1,Directions.SOUTH)]:
                nx, ny = int(x+dx), int(y+dy)
                npos = (nx, ny)
                if (0 <= nx < self.width and 0 <= ny < self.height
                        and not walls[nx][ny]
                        and npos not in forbidden):
                    ng = g + 1
                    if npos not in visited or g_map.get(npos, 9999) > ng:
                        g_map[npos] = ng
                        fa = first_action if first_action else action
                        pq.push((npos, fa), ng + heuristic(npos))

        return None  # unreachable — caller should fall back

    # ── Common enemy helpers ────────────────────────────────────────────

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def get_ghosts(self, state):
        enemies = [state.get_agent_state(i) for i in self.get_opponents(state)]
        return [a for a in enemies if not a.is_pacman and a.get_position() is not None]

    def get_scared_ghosts(self, state):
        return [g for g in self.get_ghosts(state) if g.scared_timer > 0]

    def get_dangerous_ghosts(self, state):
        return [g for g in self.get_ghosts(state) if g.scared_timer == 0]

    def get_invaders(self, state):
        enemies = [state.get_agent_state(i) for i in self.get_opponents(state)]
        return [a for a in enemies if a.is_pacman and a.get_position() is not None]


###############################################################################
# Offensive agent — A* with persistent target (no oscillation)
###############################################################################

class ImprovedOffensiveAgent(BaseAgent):
    """
    Uses A* with a PERSISTENT TARGET to avoid oscillation.

    Key insight: oscillation happens when the agent recalculates its target
    every turn and flip-flops between two equidistant options.
    Fix: lock onto a target until it is reached, eaten, or invalidated.

    Priority order (checked each turn to decide if target needs updating):
      1. End-game: return home if time is running out
      2. Ghost threat: flee home (A* avoiding ghost cells) or grab capsule
      3. Carrying enough food: return home to bank points
      4. Chase scared ghost
      5. Persistent food target (only pick new one when current is gone)
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.target = None   # persistent target position
        self.mode   = None   # 'food', 'home', 'flee', 'capsule', 'scared'

    def _target_valid(self, game_state, my_pos, my_state):
        """Check if the current target is still worth pursuing."""
        if self.target is None:
            return False
        # Once we crossed back home (not pacman anymore), home/flee targets are done
        if self.mode in ('home', 'flee'):
            return my_state.is_pacman  # invalid once we crossed the border
        if self.mode == 'food':
            return self.target in self.get_food(game_state).as_list()
        if self.mode == 'capsule':
            return self.target in self.get_capsules(game_state)
        if self.mode == 'scared':
            scared_pos = {g.get_position() for g in self.get_scared_ghosts(game_state)}
            return self.target in scared_pos
        return True

    def _set_target(self, pos, mode):
        self.target = pos
        self.mode   = mode

    def choose_action(self, game_state):
        my_pos   = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)

        legal     = [a for a in game_state.get_legal_actions(self.index)
                     if a != Directions.STOP]
        if not legal:
            return Directions.STOP

        dangerous       = self.get_dangerous_ghosts(game_state)
        scared          = self.get_scared_ghosts(game_state)
        ghost_positions = {g.get_position() for g in dangerous}
        time_left       = game_state.data.timeleft
        dist_home       = self.dist_to_home(my_pos)

        # ── 0. JUST CROSSED HOME — keep running if ghost is still close ────
        # A chasing ghost can be 1-2 steps away right after we cross.
        # While we are NOT pacman but a dangerous ghost is within SAFE_BUFFER,
        # immediately return the action that moves us furthest from it.
        # This bypasses the target system entirely — pure reactive escape.
        SAFE_BUFFER = 7
        if not my_state.is_pacman and dangerous:
            min_gd = min(self.get_maze_distance(my_pos, g.get_position())
                         for g in dangerous)
            if min_gd < SAFE_BUFFER:
                # Pick the legal action that maximises distance to nearest ghost
                def min_ghost_dist_after(a):
                    succ_pos = self.get_successor(game_state, a).get_agent_state(self.index).get_position()
                    return min(self.get_maze_distance(succ_pos, g.get_position()) for g in dangerous)
                return max(legal, key=min_ghost_dist_after)

        # ── 1. END-GAME (overrides everything) ───────────────────────────
        if (my_state.is_pacman and my_state.num_carrying > 0
                and time_left <= (dist_home + ENDGAME_MARGIN) * 2):
            self._set_target(self.nearest_home_cell(my_pos), 'home')

        # ── 2. GHOST THREAT — fully reactive, no target system ──────────
        elif my_state.is_pacman and dangerous:
            min_gd = min(self.get_maze_distance(my_pos, g.get_position())
                         for g in dangerous)
            if min_gd <= GHOST_THREAT_RADIUS:
                # Capsule: grab it if reachable before ghost
                caps = self.get_capsules(game_state)
                if caps:
                    best_cap = min(caps, key=lambda c: self.get_maze_distance(my_pos, c))
                    cap_dist = self.get_maze_distance(my_pos, best_cap)
                    if cap_dist < min_gd:
                        self._set_target(best_cap, 'capsule')
                        # fall through to execute block
                    else:
                        return self._flee(game_state, my_pos, legal, dangerous)
                else:
                    return self._flee(game_state, my_pos, legal, dangerous)

        # ── 3. RETURN HOME (carrying threshold) ──────────────────────────
        elif my_state.is_pacman and my_state.num_carrying >= CARRY_THRESHOLD:
            if self.mode not in ('home',):
                self._set_target(self.nearest_home_cell(my_pos), 'home')

        # ── 4. CHASE SCARED GHOSTS ───────────────────────────────────────
        elif scared and self.mode not in ('food',):
            close = [g for g in scared
                     if self.get_maze_distance(my_pos, g.get_position()) <= SCARED_CHASE_RADIUS]
            if close:
                best = min(close, key=lambda g: self.get_maze_distance(my_pos, g.get_position()))
                self._set_target(best.get_position(), 'scared')

        # ── 5. FOOD — only pick new target if current one is gone ─────────
        # This is the key anti-oscillation mechanism: we do NOT recalculate
        # the food target every turn, only when the current one was eaten.
        if not self._target_valid(game_state, my_pos, my_state):
            food_list = self.get_food(game_state).as_list()
            if food_list:
                # Avoid dead-ends when ghosts are nearby
                if dangerous and min(self.get_maze_distance(my_pos, g.get_position())
                                     for g in dangerous) <= GHOST_THREAT_RADIUS + 3:
                    safe = [f for f in food_list if f not in self.deadend]
                    pool = safe if safe else food_list
                else:
                    pool = food_list
                best_food = min(pool, key=lambda f: self.get_maze_distance(my_pos, f))
                self._set_target(best_food, 'food')

        # ── Execute: A* toward current target ────────────────────────────
        if self.target is not None:
            action = self.astar_action(game_state, my_pos, {self.target})
            if action and action in legal:
                if my_pos == self.target:
                    self.target = None
                    self.mode   = None
                return action

        # Fallback: maximise distance from ghosts if any, else random
        if dangerous:
            return max(legal,
                       key=lambda a: min(
                           self.get_maze_distance(
                               self.get_successor(game_state, a)
                                   .get_agent_state(self.index).get_position(),
                               g.get_position())
                           for g in dangerous))
        return random.choice(legal)

    def _flee(self, game_state, my_pos, legal, dangerous):
        """
        Reactive flee: score each legal action by a weighted combination of
          - distance to home (want HIGH  → safe)
          - distance to nearest ghost (want HIGH → safe)
        Uses only maze distances so it works correctly in any corridor.
        No forbidden zones, no A* — pure per-action scoring avoids loops.
        """
        def score(a):
            succ_pos = (self.get_successor(game_state, a)
                            .get_agent_state(self.index).get_position())
            d_ghost = min(self.get_maze_distance(succ_pos, g.get_position())
                          for g in dangerous)
            d_home  = self.dist_to_home(succ_pos)
            # Strong ghost avoidance; home as secondary objective
            return 8 * d_ghost - d_home
        return max(legal, key=score)


###############################################################################
# Defensive agent
###############################################################################

class ImprovedDefensiveAgent(BaseAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self._compute_patrol_points(game_state)

    def _compute_patrol_points(self, game_state):
        our_food = self.get_food_you_are_defending(game_state).as_list()
        scored = [(sum(1 for fd in our_food
                       if self.get_maze_distance(cell, fd) <= PATROL_DEPTH), cell)
                  for cell in self.border_cells]
        scored.sort(reverse=True)
        self.patrol_points = [c for _, c in scored[:max(3, len(scored)//3)]]
        self._patrol_idx   = 0

    def next_patrol(self, pos):
        target = self.patrol_points[self._patrol_idx % len(self.patrol_points)]
        if self.get_maze_distance(pos, target) <= 1:
            self._patrol_idx += 1
        return self.patrol_points[self._patrol_idx % len(self.patrol_points)]

    def choose_action(self, game_state):
        actions = [a for a in game_state.get_legal_actions(self.index)
                   if a != Directions.STOP]
        if not actions:
            return Directions.STOP
        values  = [self.get_features(game_state, a) * self.get_weights(game_state, a)
                   for a in actions]
        best = max(zip(values, actions))[1]
        return best

    def get_features(self, game_state, action):
        f = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state  = successor.get_agent_state(self.index)
        my_pos    = my_state.get_position()

        f['on_defense'] = 1 if not my_state.is_pacman else 0

        invaders = self.get_invaders(successor)
        f['num_invaders'] = len(invaders)
        scared = my_state.scared_timer > 0

        if invaders:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            min_d = min(dists)
            if scared:
                f['scared_invader_dist'] = -min_d
            else:
                f['invader_dist'] = min_d
        else:
            f['patrol_dist'] = self.get_maze_distance(my_pos, self.next_patrol(my_pos))

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            f['reverse'] = 1

        return f

    def get_weights(self, game_state, action):
        if game_state.get_agent_state(self.index).scared_timer > 0:
            return {'num_invaders': -800, 'on_defense': 100,
                    'scared_invader_dist': -15, 'patrol_dist': -5, 'reverse': -2}
        return {'num_invaders': -1000, 'on_defense': 100,
                'invader_dist': -12, 'patrol_dist': -4, 'reverse': -2}