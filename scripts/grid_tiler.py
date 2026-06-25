#!/usr/bin/env python3
"""
Grid-anchored drawing tiler for engineering-drawing QA.
- Detects each sheet's border frame (grey-line aware).
- Pairs Rev1 page N with Rev2 page N.
- Registers orientation+scale (auto-rotate Rev1 to match Rev2) so zones are like-for-like.
- Snaps tile boundaries to the printed grid ticks when detectable; else even division.
- Emits zone-labelled Rev1/Rev2 tile pairs as base64 PNG JSON for the LLM step.
Usage: grid_tiler.py REV1.pdf REV2.pdf [--cols 4] [--rows 3] [--dpi 170] [--out tiles.json]
"""
import cv2, numpy as np, subprocess, os, glob, json, sys, argparse, base64

def render(pdf,page,dpi):
    pre=f"_gt{os.getpid()}_{page}_{dpi}"
    subprocess.run(["pdftoppm","-png","-r",str(dpi),"-f",str(page),"-l",str(page),pdf,pre],
                   check=True,stderr=subprocess.DEVNULL)
    f=sorted(glob.glob(pre+"*.png"))[0]; img=cv2.imread(f); os.remove(f); return img

def npages(pdf):
    out=subprocess.run(["pdfinfo",pdf],capture_output=True,text=True).stdout
    for ln in out.splitlines():
        if ln.startswith("Pages:"): return int(ln.split()[1])
    return 1

def has_text(pdf):
    out=subprocess.run(["pdffonts",pdf],capture_output=True,text=True).stdout.strip().splitlines()
    return len(out)>2   # header is 2 lines; >2 means at least one font

def page_size(pdf,page):
    out=subprocess.run(["pdfinfo","-f",str(page),"-l",str(page),pdf],capture_output=True,text=True).stdout
    for ln in out.splitlines():
        if "size:" in ln.lower():
            parts=ln.split()
            try:
                w=float(parts[parts.index("size:")+1]); h=float(parts[parts.index("size:")+3]); return (w,h)
            except Exception: pass
    return None

def preflight(rev1,rev2):
    p1,p2=npages(rev1),npages(rev2)
    t1,t2=has_text(rev1),has_text(rev2)
    s1=page_size(rev1,1); s2=page_size(rev2,1)
    notes=[]
    if p1!=p2: notes.append(f"Page count differs: Rev1 has {p1}, Rev2 has {p2}.")
    if not t1: notes.append("Rev1 has NO extractable text layer (outlined/rasterized) - its dimensions are image-only and read at lower confidence.")
    if not t2: notes.append("Rev2 has NO extractable text layer (outlined/rasterized) - its dimensions are image-only and read at lower confidence.")
    def aspect(s): return (max(s)/min(s)) if s else None
    if s1 and s2:
        a1,a2=aspect(s1),aspect(s2)
        if abs((s1[0]*s1[1]) - (s2[0]*s2[1]))/max(s1[0]*s1[1],s2[0]*s2[1])>0.15:
            notes.append(f"Sheet sizes differ (Rev1 {int(s1[0])}x{int(s1[1])}pt vs Rev2 {int(s2[0])}x{int(s2[1])}pt) - scale normalised by registration.")
        port1=s1[1]>s1[0]; port2=s2[1]>s2[0]
        if port1!=port2: notes.append("Page orientation differs between revisions - corrected by registration.")
    return {"rev1_pages":p1,"rev2_pages":p2,"rev1_has_text":t1,"rev2_has_text":t2,
            "rev1_size_pt":s1,"rev2_size_pt":s2,"notes":notes}

def frame_box(gray):
    bw=(gray<200).astype(np.uint8)*255; H,W=gray.shape
    ci=(bw>0).sum(0); ri=(bw>0).sum(1)
    cs=np.where(ci>0.3*H)[0]; rs=np.where(ri>0.3*W)[0]
    if len(cs)<2 or len(rs)<2:   # relax
        cs=np.where(ci>0.15*H)[0]; rs=np.where(ri>0.15*W)[0]
    if len(cs)<2 or len(rs)<2: return (0,0,W,H)
    return (int(cs.min()),int(rs.min()),int(cs.max()),int(rs.max()))

def edge_norm(img,size=(700,500)):
    e=cv2.Canny(img,50,150); e=cv2.resize(e,size,interpolation=cv2.INTER_AREA)
    return (e.astype(np.float32)-e.mean())/(e.std()+1e-6)

def best_rotation(src_gray, ref_gray):
    ref=edge_norm(ref_gray); best=(0,None,-1)
    for r,code in [(0,None),(90,cv2.ROTATE_90_CLOCKWISE),(180,cv2.ROTATE_180),(270,cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        im=src_gray if code is None else cv2.rotate(src_gray,code)
        s=float((edge_norm(im)*ref).mean())
        if s>best[2]: best=(r,code,s)
    return best  # (deg, cv2code, score)

def grid_ticks(gray_frame, axis, n_expect_min=5, n_expect_max=20):
    """Detect grid tick boundaries near the frame edge. axis=0 columns(x), 1 rows(y)."""
    H,W=gray_frame.shape
    bw=(gray_frame<160).astype(np.uint8)
    strip=max(6,int(0.05*(H if axis==1 else W)))
    band=bw[:strip,:] if axis==0 else bw[:,:strip]
    prof=band.sum(0 if axis==0 else 1).astype(float)
    if prof.max()<=0: return None
    prof=cv2.GaussianBlur(prof.reshape(-1,1),(1,1),0).ravel()
    prof=prof/prof.max()
    th=0.45; peaks=[]; i=0; n=len(prof)
    while i<n:
        if prof[i]>th:
            j=i
            while j<n and prof[j]>th: j+=1
            peaks.append((i+j)//2); i=j
        else: i+=1
    # need plausible count
    if not (n_expect_min<=len(peaks)<=n_expect_max): return None
    return sorted(peaks)

def boundaries(length, n_cells, ticks):
    """Return n_cells+1 boundary positions, snapped to ticks if available."""
    even=[round(k*length/n_cells) for k in range(n_cells+1)]
    if not ticks: return even
    snapped=[]
    for e in even:
        nearest=min(ticks,key=lambda t:abs(t-e))
        snapped.append(nearest if abs(nearest-e)<length/(2*n_cells) else e)
    snapped[0]=0; snapped[-1]=length
    return sorted(set(snapped)) if len(set(snapped))==n_cells+1 else even

def zone_name(r,c,nrows,ncols):
    letters="ABCDEFGHJKLMNPQRSTUV"
    L=letters[min(nrows-1-r,len(letters)-1)]   # bottom row = A
    return f"{L}{c+1}"

def b64png(img):
    ok,buf=cv2.imencode(".png",img); return base64.b64encode(buf).decode()

def process(rev1,rev2,cols,rows,dpi):
    p1,p2=npages(rev1),npages(rev2)
    pages=min(p1,p2)
    pf=preflight(rev1,rev2)
    result={"meta":{"rev1_pages":p1,"rev2_pages":p2,"compared_pages":pages,
                    "grid_cols":cols,"grid_rows":rows,"dpi":dpi},"preflight":pf,"pages":[],
            "warnings":list(pf["notes"])}
    if p1!=p2: result["warnings"].append(f"Page count differs (Rev1={p1}, Rev2={p2}); compared first {pages}.")
    for pg in range(1,pages+1):
        a_img=render(rev1,pg,dpi); b_img=render(rev2,pg,dpi)
        ga=cv2.cvtColor(a_img,cv2.COLOR_BGR2GRAY); gb=cv2.cvtColor(b_img,cv2.COLOR_BGR2GRAY)
        ax0,ay0,ax1,ay1=frame_box(ga); bx0,by0,bx1,by1=frame_box(gb)
        A=a_img[ay0:ay1,ax0:ax1]; B=b_img[by0:by1,bx0:bx1]
        gA=cv2.cvtColor(A,cv2.COLOR_BGR2GRAY); gB=cv2.cvtColor(B,cv2.COLOR_BGR2GRAY)
        deg,code,score=best_rotation(gA,gB)
        if code is not None: A=cv2.rotate(A,code)
        A=cv2.resize(A,(B.shape[1],B.shape[0]),interpolation=cv2.INTER_AREA)
        gB=cv2.cvtColor(B,cv2.COLOR_BGR2GRAY)
        cticks=grid_ticks(gB,0); rticks=grid_ticks(gB,1)
        xs=boundaries(B.shape[1],cols,cticks); ys=boundaries(B.shape[0],rows,rticks)
        zones=[]
        for r in range(len(ys)-1):
            for c in range(len(xs)-1):
                x0,x1=xs[c],xs[c+1]; y0,y1=ys[r],ys[r+1]
                za=A[y0:y1,x0:x1]; zb=B[y0:y1,x0:x1]
                zones.append({"zone":f"S{pg}:"+zone_name(r,c,rows,cols),"page":pg,"row":r,"col":c,
                              "media_type":"image/png","rev1":b64png(za),"rev2":b64png(zb)})
        result["pages"].append({"page":pg,"rotation_applied_to_rev1":deg,"registration_score":round(score,4),
            "grid_source":{"cols":"ticks" if cticks else "even","rows":"ticks" if rticks else "even"},
            "frame_rev1":[ax0,ay0,ax1,ay1],"frame_rev2":[bx0,by0,bx1,by1],"zone_count":len(zones)})
        if score<0.15: result["warnings"].append(f"Page {pg}: low registration score ({score:.2f}); revisions may not align well.")
        result["pages"][-1]["zones_inline"]=zones  # held separately below
    # flatten zones to top level for the payload step, keep page meta clean
    allzones=[]
    for pinfo in result["pages"]:
        allzones+=pinfo.pop("zones_inline")
    scores=[p["registration_score"] for p in result["pages"]]
    result["meta"]["min_registration_score"]=round(min(scores),4) if scores else None
    result["meta"]["comparable"]= bool(scores) and min(scores)>=0.15
    if not result["meta"]["comparable"]:
        result["warnings"].append("Low registration score on at least one page - revisions may not align; treat as NEEDS_REVIEW.")
    result["zones"]=allzones
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("rev1"); ap.add_argument("rev2")
    ap.add_argument("--cols",type=int,default=4); ap.add_argument("--rows",type=int,default=3)
    ap.add_argument("--dpi",type=int,default=170); ap.add_argument("--out",default="-")
    a=ap.parse_args()
    res=process(a.rev1,a.rev2,a.cols,a.rows,a.dpi)
    js=json.dumps(res)
    if a.out=="-": sys.stdout.write(js)
    else: open(a.out,"w").write(js)
