# Nexus India

Interactive India telecom-intelligence dashboard built from the latest
100-agent synthetic CGAN rerun.

## Included data

- 100 complete synthetic subscriber profiles
- 100,000 CDR events
- 100,000 IPDR sessions
- 1,709 observed and CGAN-inferred friendship edges
- Every incoming and outgoing contact for each core agent
- Full raw CDR/IPDR record inspection and latitude/longitude activity points

## Local development

```powershell
npm.cmd run dev
```

Open `http://localhost:3000/`.

## Rebuild the dashboard data

```powershell
& '..\.venv\Scripts\python.exe' tools\build_dashboard_data.py `
  --input-dir 'C:\path\to\extracted\cgan_clean_outputs' `
  --output-dir '.\public\data'
```

All records shown by the site are synthetic.
