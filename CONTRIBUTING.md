# Contributing

Contributions should preserve the paper-defined state, action, reward, safety-cost, and aggregation interfaces. New algorithms should reuse the shared environment and expose all experiment settings through JSON configuration.

Before submitting a change, run Python compilation, static analysis, formatting checks, the environment smoke test, and at least one short trainer update. Generated results, checkpoints, local datasets, credentials, IDE metadata, and caches must not be committed.

Bug reports should include the command, configuration file, Python version, dependency versions, and the shortest reproducible error trace.
