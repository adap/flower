# Simulating Client Dropouts

This example provides a simple solution to simulating dropouts.
We simulate dropouts with `sys.exit()` in the client-side logic.
Call `check_dropout` to dropout under a certain probabililty in `fit`, `evaluate` or `get_parameters`.