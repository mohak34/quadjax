# volatus

## Scripts

- `uv run python scripts/fly.py`: flies one drone with fixed motor commands. Correct looks like: hover stays at z=1.0000, climb rises about 0.46 m in 1 s, and with the front-right motor at +5% it rolls and pitches negative (front-right side lifts) and drifts toward back-left (x negative, y positive).
- `uv run python scripts/compare.py`: gives our sim and MuJoCo the same thrust and torques for 3 s and saves `out/compare.png`. Correct looks like: the two lines overlap in every panel, the position gap stays well under 1 mm and the attitude gap under 0.01 deg.
