# PDE Solver dashboard

A Streamlit frontend over `optpricing`. Lives outside the `optpricing/`
package on purpose — it only imports from `optpricing`'s public API, the same
way an external user of the published library would.

## Run locally

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo.
3. Set **Main file path** to `dashboard/app.py`.
4. Leave the requirements path as default — Streamlit Cloud picks up `requirements.txt` at the repo root, which installs `optpricing` itself (via `-e .`) plus `streamlit`/`plotly`.
