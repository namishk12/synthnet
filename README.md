# SynthNet

SynthNet is the standalone local-first telecom synthetic-data workspace. It
contains the CDR/IPDR upload and identity-validation workflow, synthetic
generation, IPDR correlation, relationship learning, the data visualizer, and
the Nexus India dashboard.

## Included

- `cgan_gui.html` and `cgan_gui_server.py`: upload, identity evidence review,
  generation, correlation, and output downloads.
- `Synthetic_data_cgan.py`, `identity_validation.py`, `ipdr_correlation.py`,
  and `synthnet_validate.py`: core processing and validation.
- `friend_interaction_nn/`: supervised relationship-learning workflow.
- `visualizer.html`: the standalone CSV dashboard.
- `india-agent-visualizer/`: the Nexus India dashboard and its synthetic demo
  data.
- `tests/`: focused identity-workflow regression tests.

Generated local datasets, model artifacts, presentations, caches, virtual
environments, and dependency directories from the original workspace are not
included. The dashboard's checked-in demo data is synthetic and exists so the
dashboard can render immediately after checkout.

## Run locally

From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\start_cgan_gui.ps1
```

This opens SynthNet Studio at `http://127.0.0.1:8780/`.

The standalone visualizer is available with:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_visualizer.ps1
```

It opens at `http://127.0.0.1:8765/visualizer.html`.

To run the Nexus India dashboard:

```powershell
Set-Location .\india-agent-visualizer
npm ci
npm run dev -- --host 127.0.0.1
```

Open `http://localhost:3000/`.

## Tests

```powershell
python -m pytest -q tests friend_interaction_nn/tests
```

The browser workflows are local-first: uploaded files are processed by the
local service, and CDR/IPDR identity evidence is kept with the run metadata.
