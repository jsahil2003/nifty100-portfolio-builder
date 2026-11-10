# Nifty 100 Portfolio Builder

For educational purposes only. Not investment advice.

Enter your capital, pick a strategy, and see which Nifty 100 stocks to buy, how many shares,
and the expected return, volatility and drawdown.

Run locally:

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free (Streamlit Community Cloud)

1. Create an empty repository on GitHub (e.g. `nifty100-portfolio-builder`). Public is simplest;
   the free plan also allows one private app.
2. Push this folder to it:
   ```
   cd webapp
   git init
   git add .
   git commit -m "Nifty 100 Portfolio Builder"
   git branch -M main
   git remote add origin https://github.com/<your-username>/nifty100-portfolio-builder.git
   git push -u origin main
   ```
3. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, choose the
   repository, branch `main` and file `app.py`, and click **Deploy**. Under *Advanced settings*,
   choose Python 3.12. The first build takes a few minutes.

The app sleeps after 12 hours without visitors; the next visitor wakes it (about half a minute).

## Updating the data

The prices end on 31 July 2026. After refreshing the data and results in the main project, run
`sh make_webapp.sh` there (it copies the new files into this folder), then:

```
git add .
git commit -m "Update data"
git push
```

Streamlit Cloud redeploys automatically after every push.
