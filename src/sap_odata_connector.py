import requests
import pandas as pd
import urllib3
from dotenv import load_dotenv
from urllib.parse import unquote
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv(override=True)

BASE_URL = os.getenv("SAP_BASE_URL")

COOKIES = {
    "JSESSIONID": unquote(os.getenv("SAP_SESSION_ID", "")),
    "__VCAP_ID__": os.getenv("SAP_VCAP_ID", ""),
}

def get_session():
    session = requests.Session()
    session.verify = False
    session.max_redirects = 0
    session.cookies.update(COOKIES)
    try:
        r = session.get(
            BASE_URL,
            headers={"x-csrf-token": "Fetch", "Accept": "application/json"},
            allow_redirects=False
        )
        token = r.headers.get("x-csrf-token", "")
    except Exception:
        token = ""
    session.headers.update({
        "x-csrf-token": token,
        "Accept": "application/json"
    })
    return session

def fetch_entity(entity_set: str, filters: str = None, top: int = None) -> pd.DataFrame:
    session = get_session()
    params = {
        "$format": "json",
        "sap-language": "EN",
    }
    if top:
        params["$top"] = top
    if filters:
        params["$filter"] = filters

    all_results = []
    url = f"{BASE_URL}/{entity_set}"

    while url:
        r = session.get(url, params=params, allow_redirects=False)
        print(f"  Status: {r.status_code} | Content-Type: {r.headers.get('content-type','')[:30]}")

        if r.status_code in (301, 302, 307, 308):
            print(f"  Redirect blocked to: {r.headers.get('location','')[:80]}")
            break
        if r.status_code == 403:
            print("  403 Forbidden — session cookie may be expired")
            break
        if r.status_code == 400:
            print(f"  400 Bad Request: {r.text[:200]}")
            break
        if "text/html" in r.headers.get("content-type", ""):
            print("  Got HTML — session expired, get a new JSESSIONID from browser")
            break

        data = r.json().get("d", {})
        results = data.get("results", [])
        all_results.extend(results)
        print(f"  Fetched {len(results)} records (total: {len(all_results)})")

        next_url = data.get("__next", None)
        if next_url:
            next_url = next_url.replace(
                "https://hwpflpsrv.priv.stage.fin.purolator.com:123",
                "https://purostage.launchpad.cfapps.ca10.hana.ondemand.com/dynamic_dest/hs4-s4hana-rt"
            )
        url = next_url
        params = {}

    return pd.DataFrame(all_results)

def pull_due_credit() -> pd.DataFrame:
    return fetch_entity("DueCreditSet", filters="CompanyCode eq 'PR01'")

if __name__ == "__main__":
    print("Pulling DueCreditSet — rolling 12 months...")
    df_due =  pull_due_credit()
    print(f"\nTotal DueCreditSet rows: {len(df_due):,}")

    print("\nPulling BusinessPartnerListSet...")
    df_bp = fetch_entity("BusinessPartnerListSet")
    print(f"  → {len(df_bp):,} rows, {len(df_bp.columns)} columns")

    os.makedirs("data/raw", exist_ok=True)

    df_due = df_due.drop(columns=["__metadata"], errors="ignore")
    df_bp  = df_bp.drop(columns=["__metadata"], errors="ignore")

    df_due.to_csv("data/raw/due_credit.csv", index=False)
    df_bp.to_csv("data/raw/business_partners.csv", index=False)
    print("\nSaved to data/raw/")