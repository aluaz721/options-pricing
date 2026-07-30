# PDE Solver dashboard

A Streamlit frontend over `optpricing`. Lives outside the `optpricing/`
package on purpose — it only imports from `optpricing`'s public API, the same
way an external user of the published library would.

Live demo (Streamlit): **https://options-pricing-aluaz721.streamlit.app/**

## Run locally

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py
```