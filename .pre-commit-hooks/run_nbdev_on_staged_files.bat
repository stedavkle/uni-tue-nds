@echo off
for %%A in (%*) do (
    nbdev_clean --fname "%%A"
    REM nbdev_clean --clear_all --fname "%%A"
)