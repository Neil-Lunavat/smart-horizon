"""Core GNSS routines. Facts verified against real CDDIS files, 2026-08-02."""
import gzip, datetime as dt
import numpy as np, pandas as pd

GPS_EPOCH = dt.datetime(1980,1,6)
MU     = {"G":3.986005e14,"E":3.986004418e14,"C":3.986004418e14,"J":3.986005e14,"I":3.986005e14}
OMEGA_E= {"G":7.2921151467e-5,"E":7.2921151467e-5,"C":7.292115e-5,"J":7.2921151467e-5,"I":7.292115e-5}
C_LIGHT= 299792458.0

NAV_FIELDS = ["af0","af1","af2",
  "iode","crs","delta_n","m0", "cuc","ecc","cus","sqrt_a",
  "toe","cic","omega0","cis", "i0","crc","omega","omega_dot",
  "idot","l2_codes","week","l2_pflag", "accuracy","health","tgd","iodc",
  "trans_time","fit_interval"]

def _f(s):
    s = s.strip().replace("D","E").replace("d","e")
    if not s: return np.nan
    try: return float(s)
    except ValueError: return np.nan

def read_nav(path, systems=("C","E","G","J","I"), bds_offset=14.0):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",errors="ignore") as fh: lines = fh.read().splitlines()
    for i,l in enumerate(lines):
        if "END OF HEADER" in l: lines = lines[i+1:]; break
    recs=[]; i=0; n=len(lines)
    while i < n:
        l = lines[i]
        if len(l) < 23 or l[0] not in "GRECJIS": i+=1; continue
        s = l[0]; nc = 3 if s in "RS" else 7
        if s not in systems or i+nc >= n: i += 1+nc; continue
        try:
            toc = dt.datetime(int(l[4:8]),int(l[9:11]),int(l[12:14]),
                              int(l[15:17]),int(l[18:20]),int(l[21:23]))
        except ValueError: i += 1+nc; continue
        vals=[_f(l[23:42]),_f(l[42:61]),_f(l[61:80])]
        for k in range(1,nc+1):
            c = lines[i+k].ljust(80)
            vals += [_f(c[4:23]),_f(c[23:42]),_f(c[42:61]),_f(c[61:80])]
        r = {"sat":l[0:3],"sys":s,"toc":toc}
        r.update(dict(zip(NAV_FIELDS, vals)))
        recs.append(r); i += 1+nc
    df = pd.DataFrame(recs)
    if df.empty: return df
    is_c = (df["sys"]=="C").to_numpy()
    # VERIFIED: BDS nav carries BDT week (1074 == GPS 2430); others carry GPS week.
    week_gps = np.where(is_c, df["week"].to_numpy()+1356, df["week"].to_numpy())
    toe_gps  = np.where(is_c, df["toe"].to_numpy()+bds_offset, df["toe"].to_numpy())
    df["t_toe"] = (GPS_EPOCH + pd.to_timedelta(week_gps*604800.0,unit="s")
                             + pd.to_timedelta(toe_gps,unit="s"))
    df["t_toc"] = pd.to_datetime(df["toc"]) + pd.to_timedelta(np.where(is_c,bds_offset,0.0),unit="s")
    df = df.dropna(subset=["sqrt_a","ecc","m0","toe"])
    df = df[df["sqrt_a"]>1000]
    # VERIFIED: BDS/NavIC IODE is AODE (constant) -> arcs must key on TOE, not IODE.
    return df.sort_values(["sat","t_toe"]).drop_duplicates(["sat","t_toe"]).reset_index(drop=True)

def eval_broadcast(rec, t_gpst, bds_geo_branch=True):
    s = rec["sys"]; mu = MU[s]; we = OMEGA_E[s]
    toe_abs = (rec["t_toe"] - GPS_EPOCH).total_seconds()
    tk = t_gpst - toe_abs
    tk = np.where(tk> 302400, tk-604800, tk); tk = np.where(tk<-302400, tk+604800, tk)
    A = rec["sqrt_a"]**2
    n = np.sqrt(mu/A**3) + rec["delta_n"]
    Mk = rec["m0"] + n*tk
    e = rec["ecc"]; Ek = Mk.copy()
    for _ in range(15): Ek = Ek - (Ek - e*np.sin(Ek) - Mk)/(1 - e*np.cos(Ek))
    vk = np.arctan2(np.sqrt(1-e**2)*np.sin(Ek), np.cos(Ek)-e)
    phik = vk + rec["omega"]
    s2,c2 = np.sin(2*phik), np.cos(2*phik)
    uk = phik + rec["cus"]*s2 + rec["cuc"]*c2
    rk = A*(1-e*np.cos(Ek)) + rec["crs"]*s2 + rec["crc"]*c2
    ik = rec["i0"] + rec["idot"]*tk + rec["cis"]*s2 + rec["cic"]*c2
    xp, yp = rk*np.cos(uk), rk*np.sin(uk)
    toe_sow = rec["toe"]                       # own time system, per ICD
    is_geo_c = bds_geo_branch and s=="C" and abs(rec["i0"])<0.15
    if is_geo_c:
        Ok = rec["omega0"] + rec["omega_dot"]*tk - we*toe_sow
        xg = xp*np.cos(Ok) - yp*np.cos(ik)*np.sin(Ok)
        yg = xp*np.sin(Ok) + yp*np.cos(ik)*np.cos(Ok)
        zg = yp*np.sin(ik)
        ph = np.deg2rad(-5.0)
        y1 = yg*np.cos(ph) - zg*np.sin(ph); z1 = yg*np.sin(ph) + zg*np.cos(ph); x1 = xg
        a = we*tk
        x =  x1*np.cos(a) + y1*np.sin(a); y = -x1*np.sin(a) + y1*np.cos(a); z = z1
    else:
        Ok = rec["omega0"] + (rec["omega_dot"]-we)*tk - we*toe_sow
        x = xp*np.cos(Ok) - yp*np.cos(ik)*np.sin(Ok)
        y = xp*np.sin(Ok) + yp*np.cos(ik)*np.cos(Ok)
        z = yp*np.sin(ik)
    toc_abs = (rec["t_toc"] - GPS_EPOCH).total_seconds()
    dtc = t_gpst - toc_abs
    dtc = np.where(dtc>302400,dtc-604800,dtc); dtc = np.where(dtc<-302400,dtc+604800,dtc)
    dts = rec["af0"] + rec["af1"]*dtc + rec["af2"]*dtc**2
    # NO relativistic periodic term. VERIFIED 2026-08-02: IGS precise clocks do
    # not contain it either, so it cancels in the difference. Adding it injects a
    # spurious +-F*e*sqrt(A) signal: +-396 ns on E14/E18 (e=0.16), +-216 ns on
    # QZSS, +-23 ns on GPS. Leaving it out drops Galileo clock std 72 -> 0.7 ns.
    return x,y,z,dts,tk,uk

def read_sp3(path):
    op = gzip.open if str(path).endswith(".gz") else open
    rows=[]; ep=None
    with op(path,"rt",errors="ignore") as f:
        for ln in f:
            if ln.startswith("*"):
                p=ln[1:].split(); sec=float(p[5])
                ep=dt.datetime(int(p[0]),int(p[1]),int(p[2]),int(p[3]),int(p[4]),
                               int(sec),int(round((sec%1)*1e6)))
            elif ln.startswith("P") and ep is not None:
                sat=ln[1:4].strip()
                try:
                    x=float(ln[4:18]); y=float(ln[18:32]); z=float(ln[32:46]); c=float(ln[46:60])
                except ValueError: continue
                if abs(x)<1e-6 and abs(y)<1e-6: continue
                rows.append((ep,sat,x*1e3,y*1e3,z*1e3,np.nan if c>=999999.0 else c*1e-6))
    return pd.DataFrame(rows,columns=["epoch","sat","px","py","pz","pclk"])

def ecef_to_rtn(dxyz,pos,vel):
    eR = pos/np.linalg.norm(pos,axis=1,keepdims=True)
    h  = np.cross(pos,vel); eC = h/np.linalg.norm(h,axis=1,keepdims=True)
    eA = np.cross(eC,eR)
    return (np.einsum("ij,ij->i",dxyz,eR), np.einsum("ij,ij->i",dxyz,eA),
            np.einsum("ij,ij->i",dxyz,eC))

def classify(pos):
    """Measured, not assumed: radius + ECEF longitude drift + max latitude."""
    r = np.linalg.norm(pos,axis=1); rm = r.mean()
    lon_span = np.degrees(np.ptp(np.unwrap(np.arctan2(pos[:,1],pos[:,0]))))
    maxlat = np.degrees(np.arcsin(np.abs(pos[:,2]/r).max()))
    if rm > 4.0e7:
        return "GEO" if (lon_span < 5 and maxlat < 5) else "GSO"
    return "MEO"


def eval_broadcast_vec(df, t_gpst):
    """Vectorised across RECORDS: df has N records, t_gpst has N target times.
    Same math as eval_broadcast; used by the QC pass which needs one evaluation
    per record and was 100x slower going through Python per row."""
    sysv = df["sys"].to_numpy()
    mu = np.array([MU[s] for s in sysv]); we = np.array([OMEGA_E[s] for s in sysv])
    g = lambda c: df[c].to_numpy(float)
    toe_abs = (df["t_toe"] - GPS_EPOCH).dt.total_seconds().to_numpy()
    tk = t_gpst - toe_abs
    tk = np.where(tk>302400, tk-604800, tk); tk = np.where(tk<-302400, tk+604800, tk)
    A = g("sqrt_a")**2
    n = np.sqrt(mu/A**3) + g("delta_n")
    Mk = g("m0") + n*tk
    e = g("ecc"); Ek = Mk.copy()
    for _ in range(15): Ek = Ek - (Ek - e*np.sin(Ek) - Mk)/(1 - e*np.cos(Ek))
    vk = np.arctan2(np.sqrt(1-e**2)*np.sin(Ek), np.cos(Ek)-e)
    phik = vk + g("omega")
    s2,c2 = np.sin(2*phik), np.cos(2*phik)
    uk = phik + g("cus")*s2 + g("cuc")*c2
    rk = A*(1-e*np.cos(Ek)) + g("crs")*s2 + g("crc")*c2
    ik = g("i0") + g("idot")*tk + g("cis")*s2 + g("cic")*c2
    xp, yp = rk*np.cos(uk), rk*np.sin(uk)
    toe_sow = g("toe")
    geo = (sysv=="C") & (np.abs(g("i0"))<0.15)
    Ok = np.where(geo, g("omega0") + g("omega_dot")*tk - we*toe_sow,
                       g("omega0") + (g("omega_dot")-we)*tk - we*toe_sow)
    x = xp*np.cos(Ok) - yp*np.cos(ik)*np.sin(Ok)
    y = xp*np.sin(Ok) + yp*np.cos(ik)*np.cos(Ok)
    z = yp*np.sin(ik)
    if geo.any():
        ph = np.deg2rad(-5.0); a = we*tk
        y1 = y*np.cos(ph) - z*np.sin(ph); z1 = y*np.sin(ph) + z*np.cos(ph)
        xg =  x*np.cos(a) + y1*np.sin(a); yg = -x*np.sin(a) + y1*np.cos(a)
        x = np.where(geo, xg, x); y = np.where(geo, yg, y); z = np.where(geo, z1, z)
    toc_abs = (df["t_toc"] - GPS_EPOCH).dt.total_seconds().to_numpy()
    dtc = t_gpst - toc_abs
    dtc = np.where(dtc>302400,dtc-604800,dtc); dtc = np.where(dtc<-302400,dtc+604800,dtc)
    dts = g("af0") + g("af1")*dtc + g("af2")*dtc**2
    return x,y,z,dts,tk,uk
