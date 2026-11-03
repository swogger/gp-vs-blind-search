"""GP configurations for Study 2 (tool T19; H8; docs/design/algorithms.md, robustness arm R1-R6).

Each entry: run_gp keyword arguments, population size, and generations per E0 = 2x10^4 evaluations (pop x gens = E0).
"base" is regime A's configuration (and B's and HC's)."""
CONFIGS = {
    "base": dict(kwargs={}, pop_size=100, generations_per_E0=200),
    "R1": dict(kwargs={}, pop_size=25, generations_per_E0=800),                       # small population, long run
    "R2": dict(kwargs={}, pop_size=1000, generations_per_E0=20),                      # large population, short run
    "R3": dict(kwargs=dict(tournament_size=2), pop_size=100, generations_per_E0=200),  # weak selection
    "R4": dict(kwargs=dict(selection="lexicase"), pop_size=100, generations_per_E0=200),
    "R5": dict(kwargs=dict(init="uniform"), pop_size=100, generations_per_E0=200),    # uniform-size initialisation
    "R6": dict(kwargs=dict(max_depth=17), pop_size=100, generations_per_E0=200),      # Koza's depth limit
}
