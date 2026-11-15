# volatus

## Scripts

- `uv run python scripts/fly.py`: flies one drone with fixed motor commands. Correct looks like: hover stays at z=1.0000, climb rises about 0.46 m in 1 s, and with the front-right motor at +5% it rolls and pitches negative (front-right side lifts) and drifts toward back-left (x negative, y positive).
- `uv run python scripts/compare.py`: gives our sim and MuJoCo the same thrust and torques for 3 s and saves `out/compare.png`. Correct looks like: the two lines overlap in every panel, the position gap stays well under 1 mm and the attitude gap under 0.01 deg.
- `uv run python scripts/bench.py`: steps 1 to 65536 drones at once for 2 s of flight and prints the speed. On an RTX 3080 Laptop, 4096 drones run about 600,000 flight-seconds per real second.
- `uv run python scripts/train_hover.py`: trains the hover policy with PPO (about 20 s on the GPU), then tests it on 1000 new random starts. Saves `out/hover_curve.png`, `out/hover_curve.csv` and `out/hover_params.pkl`. Correct looks like: return climbs from near 0 to about 225 (250 is the max, the gap is time spent flying in from the start point), 0 crashes, and under 1 cm from the target after 5 s.
