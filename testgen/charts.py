# ruff: noqa: E501  (embedded SVG render code mirrors the gallery templates line for line)
"""Editorial charts for the write-up, in the Lieflat Charts visual grammar.

    uv run python -m testgen.charts            # -> docs/charts/*.html (+ .png if Brave is installed)

Six single-file HTML charts, one conclusion each, every number computed from
the committed run artifacts (no hand-typed results). Templates are the real
gallery implementations from https://github.com/larashero3-dotcom/lieflat-charts
(MIT), skinned with the "wire" preset: greys carry the data, one orange
element per chart is the hero. Selection, per the skill's rule of auditing
Lupi Editorial and Lupi Basics first:

  results     F5 Tick Rows       grounded score per arm, one tick = one point, CI as a hairline range
  harness     F12 Dumbbell Queue unaided -> grounded validity per arm, one bead = one point gained
  discordant  F6 Paired Rungs    functions only the fine-tune / only the base gets valid
  styles      F7 Stacked Rungs   assertion mix per arm, one rung = two percent of asserts
  devcurves   F2 Hairline Line   dev-171 grounded score per checkpoint, two adapters
  funnel      L13 Hourglass      KodCode curation, 169K rows poured down to 12,000

PNG snapshots use Brave/Chrome headless when available, so the markdown
write-up can embed them; the HTML files are the interactive originals.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from config import ROOT, RUNS_DIR
from testgen import stats
from testgen.train.oracle import assertion_styles

OUT = ROOT / "docs" / "charts"
BASE = "baselines-test-bf16base-grounded-20260910T203314Z"
ARMS = [  # label, run dir, hero?
    ("BASE ZERO-SHOT", BASE, False),
    ("BASE FEW-SHOT", "baselines-test-bf16base-few-grounded-20260910T205151Z", False),
    ("SFT · 388 SELF", "baselines-test-finetune-grounded-20260910T211110Z", False),
    ("DPO · 645 SELF", "baselines-test-finetune-20260911T075927Z", False),
    ("SFT · 4K KODCODE", "baselines-test-finetune-20260912T130009Z", False),
    ("SFT · 12K KODCODE", "baselines-test-finetune-20260914T115239Z", False),
    ("12K + STYLE PROMPT", "baselines-test-ext12k-teacher-20260914T152109Z", False),
    ("BASE · STYLE PROMPT", "baselines-test-control-teacher-20260914T143758Z", True),
]
DEV = {
    "4K": ROOT / "models/adapters/lora-4b-ext/devcurve.json",
    "12K": ROOT / "models/adapters/lora-4b-ext12k/devcurve.json",
}
DEV_BASE = 0.621
FUNNEL = [  # KodCode curation (data/train/ext*/yield.json, stage_a.json, stage_b.json)
    ("ROWS, FIVE SUBSETS", 169_000),
    ("ONE PURE FN · 8+ MUTANTS", 44_527),
    ("SAMPLED TO EXECUTE", 24_000),
    ("PASS + KILL A MUTANT", 21_085),
    ("FIT 1,024 TOK · DEDUPED", 13_429),
    ("KEPT FOR TRAINING", 12_000),
]
BROWSERS = [
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

# ── wire preset + card css (inlined per the skill's single-file rule) ──
HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{{--bg:#F0F0EE;--ink:#1F1E1C;--muted:rgba(31,30,28,.60);--faint:rgba(31,30,28,.32);--grid:rgba(31,30,28,.16)}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);font-family:'Inter',sans-serif;color:var(--ink);padding:40px;-webkit-font-smoothing:antialiased}}
.card{{background:var(--bg);border-radius:24px;padding:28px 28px 20px;max-width:{width}px;margin:0 auto}}
h2{{font-weight:700;font-size:16.5px;letter-spacing:-.02em;margin-bottom:3px}}
.sub{{font-size:11.5px;color:var(--muted);margin-bottom:14px}}
.src{{font-size:9.5px;color:var(--faint);margin-top:10px;letter-spacing:.08em;font-weight:500}}
svg{{width:100%;max-height:{svgh}px;display:block;margin:0 auto}}
svg text{{font-family:'Inter',sans-serif}}
.pop{{transform-box:fill-box;transform-origin:center;animation:pop .5s cubic-bezier(.2,.7,.3,1.3) both}}
@keyframes pop{{from{{transform:scale(0)}}to{{transform:none}}}}
.fade{{animation:fade .9s ease both}}
@keyframes fade{{from{{opacity:0}}}}
.draw{{stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1s cubic-bezier(.4,0,.2,1) both}}
@keyframes draw{{to{{stroke-dashoffset:0}}}}
@media (prefers-reduced-motion:reduce){{.pop,.fade{{animation:none}}.draw{{animation:none;stroke-dasharray:none;stroke-dashoffset:0}}}}
</style></head><body>
<div class="card">
<h2>{h2}</h2>
<div class="sub">{sub}</div>
<svg id="ch" viewBox="0 0 {vw} {vh}" preserveAspectRatio="xMidYMid meet"></svg>
<div class="src">{src}</div>
</div>
<script>
const BG='#F0F0EE',TXT='#1F1E1C',MUT='rgba(31,30,28,.60)',LAB='rgba(31,30,28,.72)',FAINT='rgba(31,30,28,.32)';
const FLOOR='rgba(31,30,28,.24)',GRID='rgba(31,30,28,.16)',DATA='#22211F',DATA2='#8F8E86',HERO='#F5572F';
const FAINTDATA='#C0BFB7',BEAD='#8F8E86',PAPER=BG,INK=DATA;
const NS='http://www.w3.org/2000/svg';
const el=(p,t,a)=>{{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);p.appendChild(n);return n}};
const txt=(p,a,s)=>{{const n=el(p,'text',a);n.textContent=s;return n}};
const tip=(n,s)=>{{const t=document.createElementNS(NS,'title');t.textContent=s;n.appendChild(t)}};
const rnd=(i,k)=>Math.abs(((i*73856093)^(k*19349663))%1000)/1000;
const obsReveal=(id,fn)=>{{const n=document.getElementById(id);const go=()=>{{n.innerHTML='';fn(n)}};
  const io=new IntersectionObserver(es=>{{if(es[0].isIntersecting){{go();io.disconnect()}}}},{{threshold:.3}});
  io.observe(n);n.style.cursor='pointer';n.addEventListener('click',go)}};
const D={data};
{script}
</script></body></html>
"""


def _records(run: str) -> dict[str, dict]:
    path = RUNS_DIR / run / "outputs.jsonl"
    return {json.loads(ln)["id"]: json.loads(ln) for ln in path.read_text().splitlines() if ln}


def _page(name: str, h2: str, sub: str, src: str, data, script: str, vh: int = 320) -> Path:
    html = HEAD.format(
        title=h2,
        width=720,
        svgh=int(vh * 1.9),
        h2=h2,
        sub=sub,
        vw=400,
        vh=vh,
        src=src,
        data=json.dumps(data),
        script=script,
    )
    path = OUT / f"{name}.html"
    path.write_text(html)
    return path


def arm_stats() -> list[dict]:
    """Per arm vs base zero-shot: grounded score + CI, validities, discordant counts, styles."""
    base = _records(BASE)
    out = []
    for label, run, hero in ARMS:
        recs = _records(run)
        cmp = stats.compare(recs, base, "arm", "base", grounded=True)
        d = cmp["grounded_score_diff_all_functions"]
        g = cmp["grounded_score"]["arm"]
        unaided = sum(
            1 for r in recs.values() if r["parsed"] and r["score"] and r["score"]["valid"]
        )
        styles = {}
        for r in recs.values():
            if r["parsed"]:
                for k, v in assertion_styles(r["suite"]).items():
                    styles[k] = styles.get(k, 0) + v
        tot = sum(styles.values())
        out.append(
            {
                "label": label,
                "hero": hero,
                "score": g,
                "lo": g - (d["mean"] - d["ci95"][0]),
                "hi": g + (d["ci95"][1] - d["mean"]),
                "unaided": unaided / len(recs),
                "grounded": cmp["validity"]["arm"],
                "only_arm": cmp["discordant"]["only_arm"],
                "only_base": cmp["discordant"]["only_base"],
                "literal": 100 * styles.get("literal_eq", 0) / tot,
                "membership": 100 * styles.get("membership", 0) / tot,
                "other": 100
                * (tot - styles.get("literal_eq", 0) - styles.get("membership", 0))
                / tot,
            }
        )
    return out


def chart_results(arms: list[dict]) -> Path:
    """F5 Tick Rows: one tick = one point of grounded score; CI drawn as a hairline range."""
    data = [
        [
            a["label"],
            round(a["score"] * 100, 1),
            round(a["lo"] * 100, 1),
            round(a["hi"] * 100, 1),
            a["hero"],
        ]
        for a in arms
    ]
    script = """
obsReveal('ch',s=>{
  const y0=i=>36+i*34,X0=118,PX=3.7,ORIGIN=40; // ticks start at 40 points: nothing scored below it
  const xOf=v=>X0+(v-ORIGIN)*PX;
  D.forEach(([name,v,lo,hi,hero],i)=>{
    const y=y0(i),n=Math.round(v-ORIGIN);
    txt(s,{x:108,y:y+3,'font-size':7.5,'font-weight':700,fill:LAB,'text-anchor':'end',
      'letter-spacing':'.06em',class:'fade',style:`animation-delay:${i*.08}s`},name);
    el(s,'line',{x1:X0,y1:y+9,x2:X0+32*PX,y2:y+9,stroke:GRID,'stroke-width':.6,class:'fade',style:`animation-delay:${i*.08}s`});
    const rowInk=hero?HERO:DATA;
    for(let k=0;k<n;k++){
      const x=X0+k*PX+PX/2,h=9+rnd(k+1,i+2)*6;
      el(s,'line',{x1:x,y1:y+9,x2:x,y2:y+9-h,stroke:rowInk,'stroke-width':1.5,
        opacity:.85+rnd(k+3,i+5)*.15,class:'fade',style:`animation-delay:${i*.08+k*.012}s`});
      if(k%5===4)el(s,'circle',{cx:x,cy:y+13,r:.8,fill:FAINT,class:'fade',style:`animation-delay:${i*.08+k*.012}s`});
    }
    // paired 95% interval vs base zero-shot, as a hairline range with end ticks
    if(i>0){
      el(s,'line',{x1:xOf(lo),y1:y+17,x2:xOf(hi),y2:y+17,stroke:FLOOR,'stroke-width':.8,class:'fade',style:`animation-delay:${.6+i*.08}s`});
      [lo,hi].forEach(v=>el(s,'line',{x1:xOf(v),y1:y+14.5,x2:xOf(v),y2:y+19.5,stroke:FLOOR,'stroke-width':.8,class:'fade',style:`animation-delay:${.6+i*.08}s`}));
    }
    const lab=txt(s,{x:xOf(v)+8,y:y+4,'font-size':11,'font-weight':800,fill:hero?HERO:TXT,
      class:'fade',style:`animation-delay:${.4+i*.08}s`},(v/100).toFixed(3));
    tip(lab,`${name} — grounded score ${(v/100).toFixed(3)}, 95% CI vs base [${(lo/100).toFixed(3)}, ${(hi/100).toFixed(3)}]`);
  });
  el(s,'line',{x1:xOf(D[0][1]),y1:32,x2:xOf(D[0][1]),y2:y0(7)+22,stroke:FLOOR,'stroke-width':.7,'stroke-dasharray':'2 3',class:'fade',style:'animation-delay:.8s'});
  txt(s,{x:200,y:308,'font-size':7,'font-weight':600,fill:FAINT,'text-anchor':'middle','letter-spacing':'.12em',class:'fade',style:'animation-delay:.9s'},
    'ONE TICK = ONE POINT ABOVE 0.40 · RANGE = PAIRED 95% CI VS BASE');
});"""
    return _page(
        "results",
        "The prompt for the teacher's style beat the fine-tune",
        "grounded score = mutants caught per function, 0 if the suite fails after the harness fills values · test split, 315 functions, Qwen3.5-4B · orange = the base, prompted for short exact-value suites",
        "TICK ROWS · WIRE · RUNS/BASELINES-TEST-* · PAIRED BOOTSTRAP, 2,000 RESAMPLES",
        data,
        script,
    )


def chart_harness(arms: list[dict]) -> Path:
    """F12 Dumbbell Queue: hollow = as written, ink = after the harness fills values; one bead = one point."""
    data = [
        [a["label"], round(a["unaided"] * 100, 1), round(a["grounded"] * 100, 1), a["hero"]]
        for a in arms
    ]
    script = """
obsReveal('ch',s=>{
  const y0=i=>42+i*34,X0=126,X1=372,mapX=v=>X0+(v-35)/55*(X1-X0);
  D.forEach(([name,was,now,hero],i)=>{
    const y=y0(i),xa=mapX(was),xb=mapX(now),ink=hero?HERO:DATA;
    txt(s,{x:116,y:y+3,'font-size':7.5,'font-weight':700,fill:LAB,'text-anchor':'end','letter-spacing':'.06em',class:'fade',style:`animation-delay:${i*.08}s`},name);
    el(s,'line',{x1:X0-6,y1:y,x2:X1+6,y2:y,stroke:GRID,'stroke-width':.7,class:'fade',style:`animation-delay:${i*.08}s`});
    const n=Math.round(now-was);
    for(let k=0;k<n;k++){
      const t=(k+.5)/n,x=xa+t*(xb-xa),yy=y+(rnd(k+1,i+3)-.5)*2.6;
      el(s,'circle',{cx:x,cy:yy,r:1.5+rnd(k+2,i+4)*.9,fill:hero?HERO:BEAD,opacity:.85,class:'pop',style:`animation-delay:${.3+i*.08+k*.03}s`});
    }
    el(s,'circle',{cx:xa,cy:y,r:4.2,fill:PAPER,stroke:ink,'stroke-width':1.5,class:'pop',style:`animation-delay:${.2+i*.08}s`});
    const after=el(s,'circle',{cx:xb,cy:y,r:4.6,fill:ink,class:'pop',style:`animation-delay:${.6+i*.08}s`});
    tip(after,`${name} — validity ${was}% as written → ${now}% after the harness fills expected values`);
    txt(s,{x:xa-9,y:y+3.5,'font-size':8.5,'font-weight':700,fill:FAINT,'text-anchor':'end',class:'fade',style:`animation-delay:${.3+i*.08}s`},was.toFixed(0));
    txt(s,{x:xb+9,y:y+3.5,'font-size':10,'font-weight':800,fill:hero?HERO:TXT,class:'fade',style:`animation-delay:${.7+i*.08}s`},now.toFixed(0));
  });
  txt(s,{x:X0,y:300,'font-size':7,'font-weight':600,fill:FAINT,class:'fade'},'VALIDITY, % OF 315 SUITES →');
  txt(s,{x:200,y:314,'font-size':7,'font-weight':600,fill:FAINT,'text-anchor':'middle','letter-spacing':'.12em',class:'fade',style:'animation-delay:1s'},
    'HOLLOW = AS WRITTEN · INK = AFTER FILLING · BEAD = ONE POINT');
});"""
    return _page(
        "harness",
        "Execution fills the values: 28 points before any fine-tune",
        "share of suites that pass on the correct function, as written and after the harness rewrites literal expected values from execution · every arm gets the same harness · orange = base, style prompt",
        "DUMBBELL QUEUE · WIRE · RUNS/BASELINES-TEST-* · TESTGEN/TRAIN/ORACLE.PY",
        data,
        script,
    )


def chart_discordant(arms: list[dict]) -> Path:
    """F6 Paired Rungs: per arm, functions grounded-valid only for the arm (ink) vs only for the base (faint)."""
    data = [
        [
            a["label"].replace("BASE ", "").replace(" KODCODE", ""),
            a["only_base"],
            a["only_arm"],
            a["hero"],
        ]
        for a in arms[1:]
    ]
    script = """
obsReveal('ch',s=>{
  const x0=i=>42+i*52,base=250,step=3.4,HW=8;
  D.forEach(([name,was,now,hero],i)=>{
    const xa=x0(i)-13,xb=x0(i)+13,ink=hero?HERO:DATA;
    for(let k=0;k<was;k++){const y=base-k*step,w=HW-1.2+rnd(k+1,i+2)*2.4;
      el(s,'line',{x1:xa-w,y1:y,x2:xa+w,y2:y,stroke:FAINTDATA,'stroke-width':1.8,opacity:.85+rnd(k+2,i+3)*.15,class:'fade',style:`animation-delay:${i*.08+k*.01}s`});}
    for(let k=0;k<now;k++){const y=base-k*step,w=HW-1.2+rnd(k+1,i+7)*2.4;
      el(s,'line',{x1:xb-w,y1:y,x2:xb+w,y2:y,stroke:ink,'stroke-width':1.8,opacity:.85+rnd(k+2,i+8)*.15,class:'fade',style:`animation-delay:${.15+i*.08+k*.01}s`});}
    const num=txt(s,{x:xb,y:base-(now-1)*step-9,'font-size':10.5,'font-weight':800,fill:hero?HERO:TXT,'text-anchor':'middle',class:'fade',style:`animation-delay:${.5+i*.08}s`},now);
    tip(num,`${name} — ${now} functions valid only for this arm, ${was} only for the base`);
    txt(s,{x:xa,y:base-(was-1)*step-9,'font-size':8.5,'font-weight':700,fill:FAINT,'text-anchor':'middle',class:'fade',style:`animation-delay:${.5+i*.08}s`},was);
    name.split(' · ').forEach((part,li)=>txt(s,{x:x0(i),y:base+18+li*10,'font-size':6.8,'font-weight':700,fill:MUT,'text-anchor':'middle','letter-spacing':'.06em',class:'fade',style:`animation-delay:${i*.08}s`},part));
  });
  el(s,'line',{x1:24,y1:base+4,x2:376,y2:base+4,stroke:GRID,'stroke-width':.8,class:'fade'});
  txt(s,{x:200,y:306,'font-size':7,'font-weight':600,fill:FAINT,'text-anchor':'middle','letter-spacing':'.12em',class:'fade',style:'animation-delay:1s'},
    'FAINT = ONLY THE BASE · INK = ONLY THIS ARM · RUNG = ONE FUNCTION');
});"""
    return _page(
        "discordant",
        "The functions that changed hands",
        "of 315 held-out functions, how many become grounded-valid only under this arm versus only under base zero-shot · McNemar runs on exactly these two counts",
        "PAIRED RUNGS · WIRE · RUNS/BASELINES-TEST-* · GROUNDED VALIDITY",
        data,
        script,
    )


def chart_styles(arms: list[dict]) -> Path:
    """F7 Stacked Rungs: assertion mix per arm; one rung = 2% of asserts; darkest = exact-value literals."""
    pick = [arms[0], arms[4], arms[5], arms[7]]
    data = [
        [
            a["label"].replace("BASE ", "").replace(" KODCODE", ""),
            [round(a["literal"] / 2), round(a["membership"] / 2), round(a["other"] / 2)],
            [round(a["literal"]), round(a["membership"]), round(a["other"])],
            a["hero"],
        ]
        for a in pick
    ]
    script = """
const SEG=['EXACT-VALUE LITERAL','MEMBERSHIP','OTHER'];
obsReveal('ch',s=>{
  const x0=i=>70+i*80,base=262,step=4.4,HW=13;
  D.forEach(([name,segs,pct,hero],i)=>{
    const x=x0(i);let k0=0;const SHADE=[hero?HERO:DATA,DATA2,FAINTDATA];
    segs.forEach((v,si)=>{
      for(let k=0;k<v;k++){const y=base-(k0+k+si)*step,w=HW-1.4+rnd(k+1,i*3+si+2)*2.8;
        el(s,'line',{x1:x-w,y1:y,x2:x+w,y2:y,stroke:SHADE[si],'stroke-width':1.8,opacity:.85+rnd(k+2,i+si+4)*.15,class:'fade',style:`animation-delay:${i*.09+(k0+k)*.012}s`});}
      const midY=base-(k0+v/2+si)*step;
      const lab=txt(s,{x:x+HW+7,y:midY+2.5,'font-size':8,'font-weight':800,fill:si===2?DATA2:SHADE[si],class:'fade',style:`animation-delay:${.5+i*.09+si*.06}s`},pct[si]+'%');
      tip(lab,`${name} — ${SEG[si]}: ${pct[si]}% of asserts`);
      k0+=v;
    });
    name.split(' · ').forEach((part,li)=>txt(s,{x,y:base+18+li*10,'font-size':6.8,'font-weight':700,fill:MUT,'text-anchor':'middle','letter-spacing':'.06em',class:'fade',style:`animation-delay:${i*.09}s`},part));
  });
  el(s,'line',{x1:36,y1:base+4,x2:364,y2:base+4,stroke:GRID,'stroke-width':.8,class:'fade'});
  txt(s,{x:200,y:306,'font-size':7,'font-weight':600,fill:FAINT,'text-anchor':'middle','letter-spacing':'.12em',class:'fade',style:'animation-delay:1.1s'},
    'DARK = EXACT-VALUE LITERAL · MID = MEMBERSHIP · PALE = OTHER · RUNG = 2%');
});"""
    return _page(
        "styles",
        "The style the harness rewards, learned or asked for",
        "share of asserts by kind across all 315 generated suites · exact-value literals are the ones the harness can fill · orange = the shipped adapter",
        "STACKED RUNGS · WIRE · TESTGEN/TRAIN/ORACLE.ASSERTION_STYLES",
        data,
        script,
    )


def chart_devcurves() -> Path:
    """F2 Hairline Line, two series: dev-171 grounded score per checkpoint, base as a dashed floor."""
    series = []
    for label, path in DEV.items():
        curve = json.loads(path.read_text())
        pts = sorted((int(k), v["grounded_score"]) for k, v in curve.items() if k != "best")
        series.append([label, [[k, round(v, 3)] for k, v in pts], int(curve["best"])])
    data = {"series": series, "base": DEV_BASE}
    script = """
obsReveal('ch',s=>{
  const x=k=>34+k/10000*340,base=262,map=v=>base-(v-.58)/.08*200;
  el(s,'line',{x1:24,y1:base,x2:376,y2:base,stroke:GRID,'stroke-width':.8,class:'fade'});
  [.58,.60,.62,.64,.66].forEach(v=>{el(s,'line',{x1:28,y1:map(v),x2:34,y2:map(v),stroke:FLOOR,'stroke-width':.6,class:'fade'});
    txt(s,{x:26,y:map(v)+2.5,'font-size':6.5,'font-weight':600,fill:FAINT,'text-anchor':'end',class:'fade'},v.toFixed(2));});
  el(s,'line',{x1:34,y1:map(D.base),x2:376,y2:map(D.base),stroke:FLOOR,'stroke-width':.8,'stroke-dasharray':'2 3',class:'fade',style:'animation-delay:.3s'});
  txt(s,{x:372,y:map(D.base)-4,'font-size':6.5,'font-weight':700,fill:MUT,'text-anchor':'end','letter-spacing':'.08em',class:'fade'},'BASE '+D.base.toFixed(3));
  D.series.forEach(([label,pts,best],si)=>{
    const hero=si===1,ink=hero?HERO:DATA;
    el(s,'path',{d:'M'+pts.map(([k,v])=>`${x(k)} ${map(v)}`).join(' L '),fill:'none',stroke:ink,'stroke-width':1.6,pathLength:1,class:'draw',style:`animation-duration:1.2s;animation-delay:${si*.3}s`});
    pts.forEach(([k,v],j)=>{
      const big=k===best;
      const dot=el(s,'circle',{cx:x(k),cy:map(v),r:big?4.4:2.4,fill:big?ink:PAPER,stroke:ink,'stroke-width':1.4,class:'pop',style:`animation-delay:${.4+si*.3+j*.08}s`});
      tip(dot,`${label} adapter, checkpoint ${k} — dev-171 grounded score ${v.toFixed(3)}${big?' (chosen)':''}`);
      if(big)txt(s,{x:x(k),y:map(v)-11,'font-size':9.5,'font-weight':800,fill:ink,'text-anchor':'middle',style:`paint-order:stroke;stroke:${PAPER};stroke-width:3px`,class:'fade'},v.toFixed(3));
    });
    txt(s,{x:x(pts[pts.length-1][0])+8,y:map(pts[pts.length-1][1])+3,'font-size':7.5,'font-weight':700,fill:ink,'letter-spacing':'.06em',class:'fade',style:'animation-delay:1.4s'},label);
  });
  [0,2500,5000,7500,10000].forEach(k=>txt(s,{x:x(k),y:base+16,'font-size':7,'font-weight':600,fill:MUT,'text-anchor':'middle','letter-spacing':'.08em',class:'fade'},k.toLocaleString()));
  txt(s,{x:200,y:306,'font-size':7,'font-weight':600,fill:FAINT,'text-anchor':'middle','letter-spacing':'.12em',class:'fade',style:'animation-delay:1.1s'},
    'ONE DOT = ONE CHECKPOINT · SOLID = CHOSEN · X = TRAINING ITERATION');
});"""
    return _page(
        "devcurves",
        "Dev curves picked the checkpoint; the test split judged it",
        "dev-171 grounded score per saved checkpoint for the two KodCode adapters · dashed = base zero-shot on the same 171 functions · noise at this size is about ±0.05",
        "HAIRLINE LINE · WIRE · MODELS/ADAPTERS/*/DEVCURVE.JSON",
        data,
        script,
    )


def chart_funnel() -> Path:
    """L13 Hourglass Stream: one tick ~ 1,600 rows; threads trickle to the next stage."""
    script = """
obsReveal('ch',s=>{
  const CXm=150,sy=k=>30+k*56,W0=D[0][1],w=c=>c/W0*250;
  D.forEach(([name,c],k)=>{
    const y=sy(k),hw=w(c)/2,last=k===D.length-1,ink=last?HERO:DATA;
    const n=Math.max(6,Math.round(c/1600));
    for(let t=0;t<n;t++){
      const x=CXm-hw+(t+.5)/n*hw*2+(rnd(t+1,k+3)-.5)*3;
      el(s,'line',{x1:x,y1:y-6,x2:x,y2:y+6,stroke:ink,'stroke-width':1.2,opacity:.55+rnd(t+2,k+5)*.45,class:'fade',style:`animation-delay:${k*.12+t*.004}s`});
    }
    if(k<D.length-1){
      const hw1=w(D[k+1][1])/2;
      for(let t=0;t<34;t++){
        const xt=CXm+(rnd(t+1,k*7+1)-.5)*2*hw*.94,xb=CXm+(rnd(t+3,k*7+5)-.5)*2*hw1*.94;
        el(s,'path',{d:`M${xt} ${y+8} C${xt} ${y+30} ${xb} ${sy(k+1)-30} ${xb} ${sy(k+1)-8}`,fill:'none',stroke:DATA2,'stroke-width':.8,opacity:.4,pathLength:1,class:'draw',style:`animation-delay:${.2+k*.15+t*.008}s;animation-duration:.8s`});
      }
      const pct=Math.round(D[k+1][1]/c*100);
      txt(s,{x:14,y:(y+sy(k+1))/2+3,'font-size':8.5,'font-weight':800,fill:DATA2,class:'fade',style:`animation-delay:${.5+k*.15}s`},pct+'%');
      txt(s,{x:14,y:(y+sy(k+1))/2+13,'font-size':5.6,'font-weight':600,fill:FAINT,'letter-spacing':'.08em',class:'fade',style:`animation-delay:${.5+k*.15}s`},'GET THROUGH');
    }
    el(s,'line',{x1:CXm+hw+6,y1:y,x2:290,y2:y,stroke:GRID,'stroke-width':.8,class:'fade',style:`animation-delay:${.3+k*.12}s`});
    txt(s,{x:294,y:y-1,'font-size':6.4,'font-weight':700,fill:LAB,'letter-spacing':'.06em',class:'fade',style:`animation-delay:${.35+k*.12}s`},name);
    txt(s,{x:294,y:y+10,'font-size':9.5,'font-weight':800,fill:last?HERO:TXT,class:'fade',style:`animation-delay:${.4+k*.12}s`},c.toLocaleString());
  });
  txt(s,{x:200,y:352,'font-size':7,'font-weight':600,fill:FAINT,'text-anchor':'middle','letter-spacing':'.12em',class:'fade',style:'animation-delay:1.2s'},
    'ONE TICK ≈ 1,600 ROWS · WIDTH = ROWS AT THAT STAGE · SAMPLING IS A SEEDED DRAW');
});"""
    return _page(
        "funnel",
        "The funnel, poured: 169,000 rows to 12,000 examples",
        "KodCode-V1 through the project's own harness · purity and mutant floor, then execution against the solution and its mutants, then decontamination and the token cap",
        "HOURGLASS STREAM · WIRE · TESTGEN/TRAIN/CURATE_EXT.PY · DATA/TRAIN/EXT12K/YIELD.JSON",
        FUNNEL,
        script,
        vh=360,
    )


def snapshot(paths: list[Path]) -> int:
    """Headless Brave/Chrome screenshot of each chart; animations are advanced by the virtual-time budget."""
    browser = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("chromium")
    if not browser:
        print("no headless browser found; HTML only")
        return 0
    for p in paths:
        png = p.with_suffix(".png")
        subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                "--window-size=800,720",
                "--force-device-scale-factor=2",
                "--virtual-time-budget=6000",
                f"--screenshot={png}",
                p.resolve().as_uri(),
            ],
            capture_output=True,
            timeout=120,
        )
        print(png.relative_to(ROOT) if png.exists() else f"snapshot failed: {p.name}")
    return len(paths)


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = arm_stats()
    paths = [
        chart_results(arms),
        chart_harness(arms),
        chart_discordant(arms),
        chart_styles(arms),
        chart_devcurves(),
        chart_funnel(),
    ]
    for p in paths:
        print(p.relative_to(ROOT))
    if "--no-png" not in argv:
        snapshot(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
