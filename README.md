# sokoban-solver

Built by an AI agent for make.fun's sokoban-solver-01-run-20261121 bounty.

## Approach

The drawn levels are a hallway with small rooms off it, and a box can never travel
between rooms, so the solver splits each level into its rooms: cut the player's row at
the squares a box could never be pushed through toward a goal, and each remaining floor
component that holds boxes is one room. Each room is solved push-optimally with an A*
search over pushes (never single steps), guided by the cheapest box-to-goal assignment
over precomputed push-distance tables, with dead-square and frozen-2x2 deadlock
pruning. A plain push-BFS stands behind the A* if it ever outruns its work budget, and
the same pair of searches runs on the whole level if a level ever refuses to split.
All budgets are counted in expanded states, never in seconds, so runs are
byte-for-byte deterministic.
