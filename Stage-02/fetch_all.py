#!/usr/bin/env python3
"""Stage 1 (rebuilt). Two files/day: BRDM00DLR nav + JAX0MGXFIN SP3.

JAX0MGXFIN is the ONLY final MGEX product containing GEO satellites (J07, J08);
COD/GRG/IAC/WUM/GFZ all start BeiDou at C06 and carry no GEO at all.
CLK files are NOT downloaded: SP3 clocks match the 30S CLK product to 1.6 ps,
which is 1000x below the ns-level signal. Saves ~5 MB/day.
"""
import argparse, datetime as dt, gzip, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

GPS_EPOCH=dt.date(1980,1,6)
def gw(d): return (d-GPS_EPOCH).days//7
def doy(d): return d.timetuple().tm_yday
S=requests.Session(); S.headers["User-Agent"]="gnss-fetch/1.0"
ORB_CHAIN=["JAX0MGXFIN","GFZ0MGXRAP","COD0MGXFIN"]

def gz_ok(p):
    try:
        with gzip.open(p,"rb") as f: return len(f.read(4096))>0
    except Exception: return False

def grab(urls,dest):
    if dest.exists() and gz_ok(dest): return "cached"
    dest.parent.mkdir(parents=True,exist_ok=True)
    for url in urls:
        for a in range(3):
            try: r=S.get(url,timeout=300)
            except Exception: time.sleep(2*(a+1)); continue
            if r.status_code==404: break
            if r.status_code!=200: time.sleep(3*(a+1)); continue
            if b"<html" in r.content[:600].lower(): return "LOGINPAGE"
            dest.write_bytes(r.content)
            if gz_ok(dest): return url.rsplit("/",1)[-1]
            dest.unlink(missing_ok=True); break
    return None

def one(d,out):
    y,dd,yy,wk=d.year,f"{doy(d):03d}",f"{d.year%100:02d}",gw(d)
    nav=grab([f"https://cddis.nasa.gov/archive/gnss/data/daily/{y}/{dd}/{yy}p/BRDM00DLR_S_{y}{dd}0000_01D_MN.rnx.gz",
              f"https://cddis.nasa.gov/archive/gnss/data/daily/{y}/{dd}/{yy}p/BRD400DLR_S_{y}{dd}0000_01D_MN.rnx.gz"],
             out/"nav"/f"{d:%Y%m%d}_nav.rnx.gz")
    orb=grab([f"https://cddis.nasa.gov/archive/gnss/products/{wk}/{ac}_{y}{dd}0000_01D_05M_ORB.SP3.gz" for ac in ORB_CHAIN],
             out/"orb"/f"{d:%Y%m%d}_orb.sp3.gz")
    return d,nav,orb

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",required=True); ap.add_argument("--end",required=True)
    ap.add_argument("--out",default="../raw"); ap.add_argument("--workers",type=int,default=6)
    a=ap.parse_args()
    s=dt.date.fromisoformat(a.start); e=dt.date.fromisoformat(a.end); out=Path(a.out)
    days=[s+dt.timedelta(days=k) for k in range((e-s).days+1)]
    miss=[]; n=0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for d,nav,orb in ex.map(lambda x:one(x,out),days):
            n+=1
            if nav is None or orb is None or "LOGIN" in str(nav)+str(orb):
                miss.append((d,nav,orb)); print(f"{d} MISSING nav={nav} orb={orb}",flush=True)
            elif n%20==0: print(f"{d} ok ({n}/{len(days)})",flush=True)
    print(f"\ndone {n} days, {len(miss)} incomplete")
    for d,nv,ob in miss[:30]: print("  ",d,nv,ob)

if __name__=="__main__": main()
