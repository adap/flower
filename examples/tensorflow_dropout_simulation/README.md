# Simulating Client Dropouts

This example provides a simple solution to simulating dropouts.
We call `check_dropout` to dropout under a certain probabililty in `fit`.
An ErrorRes message will be passed back to the server. The failure exception will be shown as "Forced Dropout"
Clients which have dropped out in one round can still join in in the next round