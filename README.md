# AI Animal Evolution Environment (v1)

Deterministic, extensible artificial life sandbox featuring emergent evolution of autonomous agents (herbivores & carnivores) with genomes, neural brains, mutation, and ecological resource dynamics.

## Key Features (Current Stage)
- Deterministic runs via central RNG (replayable)
- Spatial hashing for neighbor queries (near O(k))
- Toroidal world (no edge clustering)
- Modular genome + evolution engine
- Safe batch birth/death processing
- Energy economy with bounded flows
- Simple neural network decision layer (weight mutation)
- Foundational tests: determinism / energy sanity / selection signal
- Streamlit observer UI (heatmap + agents + live stats)

## Roadmap (Next Stages)
1. Mate-based reproduction with sexual selection / compatibility.
2. Trait-driven speciation (distance in trait-space clustering).
3. Reinforcement learning brain plug-in interface (policy hot-swap).
4. Biomes / seasonal cycles.
5. Persistence (run serialization + replay viewer).
6. Advanced visualization (Godot/Unity bridge).

## Quickstart

```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\\Scripts\\activate)
pip install -r requirements.txt
python main.py --steps 1000
pytest
streamlit run ui/streamlit_app.py
```

## Tests

Implemented with `pytest`. Determinism and evolutionary pressure checks included. See `tests/`.

## Configuration

Edit `config.py` or override via CLI:
```bash
python main.py --seed 999 --width 160 --height 160 --plant_growth_rate 0.08
```

## License
MIT
