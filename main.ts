// sokoban-solver: a push-optimal Sokoban solver for the make.fun sokoban-solver-01 set.
//
// How it works, in short. The levels this contest draws are a hallway with small rooms off
// it, and a box can never travel between rooms, so the level splits into independent
// rooms: cut the player's row (the hallway) at the squares a box could never be pushed
// through toward a goal, and each floor component that still holds boxes is one room.
// Every room is then solved on its own with an A* search over pushes -- never over single
// steps -- which is what the score rewards: a solution in the room's optimal push count
// earns full points, one push more earns under half.
//
// The search's estimate is the cheapest assignment of boxes to goals, where the cost of
// one box to one goal is its true push distance with the other boxes ignored (precomputed
// per goal by pulling a lone box backwards off it). That estimate never overstates the
// pushes still needed, so the first solution the search reaches is push-optimal. Two
// deadlock prunes keep the search out of dead ends: a box resting on a square no push can
// bring to any goal, and a box frozen inside a solid 2 x 2 block of walls and boxes.
// If a room's A* ever ran past its work budget, a plain breadth-first search over pushes
// (the same optimum, slower) takes over; if a level ever refused to split into rooms, the
// same two searches run on the whole level instead. All budgets are counted in expanded
// states, never in seconds, so two runs of this program write byte-identical answers.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Directions, always in this order: up, down, left, right. */
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP_CH = ['u', 'd', 'l', 'r'];
const PUSH_CH = ['U', 'D', 'L', 'R'];
/** "No way there" for distances: far above any real push count on these boards. */
const FAR = 1e9;

/** Work budgets, in expanded states: deterministic, so the answers never depend on speed. */
const ROOM_ASTAR_BUDGET = 400_000;
const ROOM_BFS_BUDGET = 1_200_000;
const LEVEL_BUDGET = 1_500_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/**
 * A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player,
 * + player on a goal), one row a line. Short rows are padded with walls, and the board's
 * outer frame counts as wall whatever it shows: nothing steps or slides off the board.
 */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goal[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goal, goals, boxes, player };
}

/**
 * The squares a lone box can be pushed from to reach some goal, found by pulling a box
 * backwards off every goal over open floor: from the square the box sits on, step back to
 * where it stood before the push, which needs the square behind that -- where the player
 * stood -- to be floor as well. A box on any other square can never finish, so a state
 * with one is dead.
 */
function liveSquares(w: number, h: number, wall: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(w * h);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const y = queue[qi];
    const yx = y % w;
    const yy = (y / w) | 0;
    for (let d = 0; d < 4; d++) {
      const xx = yx - DX[d];
      const xy = yy - DY[d];
      const sx = xx - DX[d];
      const sy = xy - DY[d];
      if (xx < 0 || xy < 0 || xx >= w || xy >= h || sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
      const x = xy * w + xx;
      if (wall[x] || wall[sy * w + sx] || live[x]) continue;
      live[x] = 1;
      queue.push(x);
    }
  }
  return live;
}

/**
 * One distance table per goal: dist[g][c] is the fewest pushes that bring a lone box from
 * square c onto goal g, other boxes ignored (the same backward pull as liveSquares, but
 * counting layers). FAR where no push sequence gets there.
 */
function pushDistances(w: number, h: number, wall: Uint8Array, goals: number[]): Int32Array[] {
  return goals.map((g) => {
    const dist = new Int32Array(w * h).fill(FAR);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const y = queue[qi];
      const yx = y % w;
      const yy = (y / w) | 0;
      for (let d = 0; d < 4; d++) {
        const xx = yx - DX[d];
        const xy = yy - DY[d];
        const sx = xx - DX[d];
        const sy = xy - DY[d];
        if (xx < 0 || xy < 0 || xx >= w || xy >= h || sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        const x = xy * w + xx;
        if (wall[x] || wall[sy * w + sx] || dist[x] !== FAR) continue;
        dist[x] = dist[y] + 1;
        queue.push(x);
      }
    }
    return dist;
  });
}

/**
 * The cheapest way to pair every box with its own goal, summed over the pairs, from the
 * per-goal push distances. Each box still needs at least its paired distance in pushes,
 * and pushes of different boxes never help each other, so this never overstates the
 * pushes a state still needs: an admissible estimate for the push search. FAR when some
 * box reaches no goal at all. Few boxes (the rooms here hold at most four) are paired by
 * trying every bijection; larger counts go through the Hungarian method.
 */
function matchCost(boxes: number[], dist: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  for (let i = 0; i < n; i++) {
    let any = false;
    for (let g = 0; g < n; g++) if (dist[g][boxes[i]] < FAR) any = true;
    if (!any) return FAR;
  }
  if (n <= 6) {
    let best = FAR;
    const used = new Uint8Array(n);
    const walk = (i: number, acc: number): void => {
      if (acc >= best) return;
      if (i === n) {
        best = acc;
        return;
      }
      for (let g = 0; g < n; g++) {
        if (used[g]) continue;
        const c = dist[g][boxes[i]];
        if (c >= FAR) continue;
        used[g] = 1;
        walk(i + 1, acc + c);
        used[g] = 0;
      }
    };
    walk(0, 0);
    return best;
  }
  // Hungarian method on the n x n cost matrix, potentials form, O(n^3).
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1).fill(-1);
  const way = new Int32Array(n + 1).fill(-1);
  for (let i = 0; i < n; i++) {
    p[n] = i;
    let j0 = n;
    const minv = new Float64Array(n + 1).fill(Infinity);
    const usedJ = new Uint8Array(n + 1);
    do {
      usedJ[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = -1;
      for (let j = 0; j < n; j++) {
        if (usedJ[j]) continue;
        const cur = dist[j][boxes[i0]] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (usedJ[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== -1);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== -1 && j0 !== n);
  }
  let total = 0;
  for (let j = 0; j < n; j++) {
    if (p[j] < 0) return FAR;
    total += dist[j][boxes[p[j]]];
    if (total >= FAR) return FAR;
  }
  return total;
}

/**
 * Mark the squares the player can walk to from `from` around walls and the boxes in
 * `occupied`, stamping them with `stamp` in the shared `seen` array, and return the
 * region's lowest square: two player positions in one region play exactly alike, so the
 * lowest square names the region in a search state. `stack` is caller scratch; when
 * `cells` is given it is filled with the region's squares, which the caller needs
 * because a later flood (a child's region) re-stamps squares of this one.
 */
function flood(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
  cells?: number[],
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  if (cells) {
    cells.length = 0;
    cells.push(from);
  }
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const up = p - w;
    const down = p + w;
    const left = p - 1;
    const right = p + 1;
    if (!wall[up] && !occupied[up] && seen[up] !== stamp) {
      seen[up] = stamp;
      stack[n++] = up;
      if (cells) cells.push(up);
    }
    if (!wall[down] && !occupied[down] && seen[down] !== stamp) {
      seen[down] = stamp;
      stack[n++] = down;
      if (cells) cells.push(down);
    }
    if (!wall[left] && !occupied[left] && seen[left] !== stamp) {
      seen[left] = stamp;
      stack[n++] = left;
      if (cells) cells.push(left);
    }
    if (!wall[right] && !occupied[right] && seen[right] !== stamp) {
      seen[right] = stamp;
      stack[n++] = right;
      if (cells) cells.push(right);
    }
  }
  return low;
}

/**
 * Whether the box just pushed onto `t` is frozen inside a solid 2 x 2 block: all four
 * squares wall or box, with at least one box among them off its goal. Such boxes can
 * never move again, so the state is dead. Only blocks containing t can newly freeze.
 */
function frozenBlock(
  w: number,
  h: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  const tx = t % w;
  const ty = (t / w) | 0;
  for (let oy = -1; oy <= 0; oy++) {
    for (let ox = -1; ox <= 0; ox++) {
      const ax = tx + ox;
      const ay = ty + oy;
      if (ax < 0 || ay < 0 || ax + 1 >= w || ay + 1 >= h) continue;
      const a = ay * w + ax;
      const cells = [a, a + 1, a + w, a + w + 1];
      let solid = true;
      let offGoal = false;
      for (const c of cells) {
        if (!wall[c] && !occupied[c]) {
          solid = false;
          break;
        }
        if (occupied[c] && !goal[c]) offGoal = true;
      }
      if (solid && offGoal) return true;
    }
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, as lower-case steps. */
function walkPath(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Int8Array(wall.length).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let d = 0; d < 4; d++) {
      const q = p + DY[d] * w + DX[d];
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkPath: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = STEP_CH[how[p]] + s;
  return s;
}

/** A board a push search runs on: static walls, the goals in play, and their tables. */
interface Board {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  goals: number[];
  dist: Int32Array[];
  dead: Uint8Array;
}

/** Build a search board from walls and the goals its boxes must fill. */
function makeBoard(w: number, h: number, wall: Uint8Array, goals: number[]): Board {
  const goal = new Uint8Array(w * h);
  for (const g of goals) goal[g] = 1;
  const dist = pushDistances(w, h, wall, goals);
  const dead = new Uint8Array(w * h);
  for (let c = 0; c < w * h; c++) {
    if (wall[c]) continue;
    let reachable = false;
    for (let g = 0; g < goals.length; g++) if (dist[g][c] < FAR) reachable = true;
    if (!reachable) dead[c] = 1;
  }
  return { w, h, wall, goal, goals, dist, dead };
}

/** One push of a solution: the box that stood on `boxFrom` was pushed along `dir`. */
interface PushStep {
  boxFrom: number;
  dir: number;
}

/**
 * Search over pushes from (startBoxes, startPlayer) until every box rests on a goal.
 * With `guided`, an A* ordered by pushes made plus the matching estimate, so the first
 * solution found is push-optimal; without, a plain breadth-first search over pushes,
 * which is optimal too and differently exhaustive. States are the sorted box squares
 * plus the player's region (named by its lowest square); a state already reached in no
 * more pushes is never expanded again. Gives up -- returns null -- after `budget`
 * expansions, a count of work, never of time.
 */
function solvePushes(
  board: Board,
  startBoxes: number[],
  startPlayer: number,
  budget: number,
  guided: boolean,
): PushStep[] | null {
  const { w, wall, goal, dist, dead } = board;
  const size = wall.length;
  const n = startBoxes.length;
  if (startBoxes.some((b) => dead[b])) return null;
  if (startBoxes.every((b) => goal[b])) return [];

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const stack = new Int32Array(size);
  const regionCells: number[] = [];
  const inRegion = new Uint8Array(size);
  let stamp = 0;

  // The states found, in the order found. A state's boxes are kept as the characters of
  // its key (the region's lowest square first, then the box squares, ascending).
  const keys: string[] = [];
  const nodePlayer: number[] = [];
  const nodeG: number[] = [];
  const nodeParent: number[] = [];
  const nodeDir: number[] = [];
  const nodeBoxFrom: number[] = [];
  const nodeF: number[] = [];
  const best = new Map<string, number>();

  // The open list: a binary heap of state indexes, cheapest estimate first; ties go to
  // the state with more pushes made (it is closer to done), then to the earlier found.
  const heap: number[] = [];
  const before = (a: number, b: number): boolean => {
    if (nodeF[a] !== nodeF[b]) return nodeF[a] < nodeF[b];
    if (nodeG[a] !== nodeG[b]) return nodeG[a] > nodeG[b];
    return a < b;
  };
  const heapPush = (i: number): void => {
    heap.push(i);
    let c = heap.length - 1;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (!before(heap[c], heap[p])) break;
      [heap[c], heap[p]] = [heap[p], heap[c]];
      c = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1], heap[c])) c++;
        if (!before(heap[c], heap[i])) break;
        [heap[c], heap[i]] = [heap[i], heap[c]];
        i = c;
      }
    }
    return top;
  };

  const addState = (
    boxes: number[],
    rep: number,
    player: number,
    g: number,
    parent: number,
    dir: number,
    boxFrom: number,
  ): number => {
    const key = String.fromCharCode(rep, ...boxes);
    const idx = keys.length;
    keys.push(key);
    nodePlayer.push(player);
    nodeG.push(g);
    nodeParent.push(parent);
    nodeDir.push(dir);
    nodeBoxFrom.push(boxFrom);
    nodeF.push(g + (guided ? matchCost(boxes, dist) : 0));
    best.set(key, idx);
    heapPush(idx);
    return idx;
  };

  for (const b of startBoxes) occupied[b] = 1;
  const rootRep = flood(w, wall, occupied, startPlayer, seen, ++stamp, stack);
  for (const b of startBoxes) occupied[b] = 0;
  addState(startBoxes, rootRep, startPlayer, 0, -1, -1, -1);

  let expanded = 0;
  while (heap.length > 0) {
    const cur = heapPop();
    if (best.get(keys[cur]) !== cur) continue; // a cheaper route to this state exists
    if (++expanded > budget) return null;
    const key = keys[cur];
    const boxes: number[] = [];
    for (let i = 1; i < key.length; i++) boxes.push(key.charCodeAt(i));
    for (const b of boxes) occupied[b] = 1;
    flood(w, wall, occupied, nodePlayer[cur], seen, ++stamp, stack, regionCells);
    for (const c of regionCells) inRegion[c] = 1;
    const g = nodeG[cur];
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi];
      for (let d = 0; d < 4; d++) {
        const step = DY[d] * w + DX[d];
        const stand = b - step;
        const dest = b + step;
        if (!inRegion[stand] || wall[dest] || occupied[dest] || dead[dest]) continue;
        occupied[b] = 0;
        occupied[dest] = 1;
        const frozen = frozenBlock(w, board.h, wall, occupied, goal, dest);
        let rep = -1;
        if (!frozen) rep = flood(w, wall, occupied, b, seen, ++stamp, stack);
        occupied[dest] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const next = boxes.slice();
        next[bi] = dest;
        next.sort((x, y) => x - y);
        const childKey = String.fromCharCode(rep, ...next);
        const known = best.get(childKey);
        if (known !== undefined && nodeG[known] <= g + 1) continue;
        const child = addState(next, rep, b, g + 1, cur, d, b);
        if (next.every((x) => goal[x])) {
          const steps: PushStep[] = [];
          for (let s = child; s > 0; s = nodeParent[s]) {
            steps.push({ boxFrom: nodeBoxFrom[s], dir: nodeDir[s] });
          }
          steps.reverse();
          return steps;
        }
      }
    }
    for (const c of regionCells) inRegion[c] = 0;
    for (const b of boxes) occupied[b] = 0;
  }
  return null;
}

/**
 * The level's rooms, or null when the level is no set of rooms. The player's row is the
 * hallway; its squares a box could never be pushed through toward a goal are cut away,
 * and the floor components left are the rooms -- a box pushed out of a room into the
 * hallway can never come back, so each component's boxes belong to its own goals. A
 * component counts as a room only when its boxes and goals balance; a box starting in
 * the hallway, or a goal sitting in it, means the level is no set of rooms at all.
 */
function splitRooms(level: Level): { boxes: number[]; goals: number[] }[] | null {
  const { w, h, wall, goals, boxes, player } = level;
  const live = liveSquares(w, h, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  if (goals.some((g) => Math.floor(g / w) === row)) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const comps: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const idx = comps.length;
    const squares = [s];
    part[s] = idx;
    for (let i = 0; i < squares.length; i++) {
      const cx = squares[i] % w;
      const cy = (squares[i] / w) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d];
        const ny = cy + DY[d];
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q = ny * w + nx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = idx;
          squares.push(q);
        }
      }
    }
    comps.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    comps[part[b]].boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    comps[part[g]].goals.push(g);
  }
  const rooms = comps.filter((c) => c.boxes.length > 0);
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  for (const r of rooms) {
    r.boxes.sort((a, b) => a - b);
    r.goals.sort((a, b) => a - b);
  }
  rooms.sort((a, b) => a.boxes[0] - b.boxes[0]);
  return rooms;
}

/**
 * Turn a push sequence into the player's full move string on a board with the given
 * walls: before each push, walk the player (shortest way, boxes standing) onto the
 * square behind the box, then push. Returns the moves and the square the player ends
 * on -- the last box's old square.
 */
function renderMoves(
  w: number,
  wall: Uint8Array,
  startBoxes: number[],
  startPlayer: number,
  steps: PushStep[],
): { moves: string; endPlayer: number } {
  const occupied = new Uint8Array(wall.length);
  for (const b of startBoxes) occupied[b] = 1;
  let player = startPlayer;
  let out = '';
  for (const { boxFrom, dir } of steps) {
    const step = DY[dir] * w + DX[dir];
    out += walkPath(w, wall, occupied, player, boxFrom - step);
    out += PUSH_CH[dir];
    occupied[boxFrom] = 0;
    occupied[boxFrom + step] = 1;
    player = boxFrom;
  }
  return { moves: out, endPlayer: player };
}

/**
 * Solve room by room: each room in turn, the other rooms' boxes standing as walls --
 * on their goals once their room is done -- and the player walking on from wherever the
 * last room left them. The rooms' push counts add up to the level's, and each room is
 * solved push-optimally, so the whole is too. Null when a room defeats both searches.
 */
function solveByRooms(level: Level, rooms: { boxes: number[]; goals: number[] }[]): string | null {
  const { w, h } = level;
  let player = level.player;
  let moves = '';
  const done = rooms.map(() => false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    const wall = level.wall.slice();
    rooms.forEach((other, oi) => {
      if (oi === ri) return;
      for (const b of done[oi] ? other.goals : other.boxes) wall[b] = 1;
    });
    const board = makeBoard(w, h, wall, room.goals);
    const steps =
      solvePushes(board, room.boxes, player, ROOM_ASTAR_BUDGET, true) ??
      solvePushes(board, room.boxes, player, ROOM_BFS_BUDGET, false);
    if (steps === null) return null;
    const rendered = renderMoves(w, wall, room.boxes, player, steps);
    moves += rendered.moves;
    player = rendered.endPlayer;
    done[ri] = true;
  }
  return moves;
}

/** The moves that solve one level: by rooms when it splits, whole otherwise. */
function solveLevel(level: Level): string {
  const rooms = splitRooms(level);
  if (rooms !== null) {
    const byRooms = solveByRooms(level, rooms);
    if (byRooms !== null) return byRooms;
  }
  const board = makeBoard(level.w, level.h, level.wall, level.goals);
  const steps =
    solvePushes(board, level.boxes, level.player, LEVEL_BUDGET, true) ??
    solvePushes(board, level.boxes, level.player, LEVEL_BUDGET, false);
  if (steps === null) return '';
  return renderMoves(level.w, level.wall, level.boxes, level.player, steps).moves;
}

/** Answer every instance under the input directory, each in its own output folder. */
function main(): void {
  let ids: string[];
  try {
    ids = readdirSync(INPUT_DIR).sort();
  } catch {
    return;
  }
  for (const id of ids) {
    let moves = '';
    try {
      const text = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
      moves = solveLevel(parseLevel(text));
    } catch {
      moves = '';
    }
    mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
    writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
  }
}

main();
