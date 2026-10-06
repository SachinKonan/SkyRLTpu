# Reviewed runtime inventories

No complete legacy environment inventory is available in this checkout. Do not
label inferred package pins as a captured working baseline.

Run `runtime_inventory.py` with the actual legacy role's Python and `--source`
pointing to its deployed source. Capture each model/role independently. Compare
the resulting JSON against an independently built replacement using `--expect`.
Review source/SDK/native-completion differences explicitly; compare source hashes
as well as package names and versions.

Once validated, place an approved JSON inventory here and select it with:

```json
{"runtime_baselines": {"trainer": "qwen-trainer.json", "serving": "qwen-serving.json", "client": "qwen-client.json"}}
```

Build rejects missing/invalid inventories. Host setup checks the full selected
inventory before allowing a model process to start, including when reusing a
venv. The inventory hash is part of that environment's identity. A profile with
no selected baseline only records packages and detects subsequent local drift;
it is **not** certified as matching legacy.

This is an acceptance gate, not a dependency lock or installer. A separate full
lock must be produced from the validated environment and used by both launchers.
