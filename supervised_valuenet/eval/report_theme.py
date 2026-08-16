"""Design tokens, page CSS and page JS for the report.

Palette: the dataviz reference palette (validated 2026-07-23 with the skill's
validate_palette.js, all-pairs, both modes):

    subgoal planner, full language    blue   #2a78d6 light / #3987e5 dark
    subgoal planner, original lang.   aqua   #1baf7a / #199e70
    move-by-move planner              orange #eb6834 / #d95926

The light-mode aqua sits at 2.74:1 contrast (validator WARN) — relief is
provided everywhere by direct labels and table twins, per the skill's rule.
Robot colors on board/plan art are the robots' literal names (red, blue,
green, yellow) and are a separate namespace from the series colors.

The CSS variable names --surface --ink --muted --baseline --line --grid
--mono --r-red --r-blue --r-green --r-yellow are required by the read-only
plan-structure assets (eval/plan_viz_core.py) and must not be renamed.
"""

# both theme blocks share these var definitions -------------------------------

_LIGHT_VARS = """
  color-scheme: light;
  --bg:#f9f9f7; --surface:#fcfcfb;
  --ink:#0b0b0b; --muted:#52514e; --faint:#898781;
  --grid:#e1e0d9; --baseline:#c3c2b7; --line:rgba(11,11,11,.10);
  --c-bwd:#2a78d6; --c-bwd-old:#1baf7a; --c-fwd:#eb6834;
  --c-bwd-wash:rgba(42,120,214,.10); --c-bwd-old-wash:rgba(27,175,122,.12);
  --c-fwd-wash:rgba(235,104,52,.10);
  --seq1:#cde2fb; --seq2:#9ec5f4; --seq3:#6da7ec; --seq4:#3987e5;
  --seq5:#256abf; --seq6:#184f95; --seq7:#0d366b;
  --warn-wash:rgba(250,178,25,.16); --run-wash:rgba(42,120,214,.10);
  --ok-ink:#006300; --bad-ink:#b02f2f;
  --r-red:#e34948; --r-blue:#2a78d6; --r-green:#008300; --r-yellow:#c98500;
  --shadow:0 1px 2px rgba(11,11,11,.05), 0 4px 14px rgba(11,11,11,.05);
"""

_DARK_VARS = """
  color-scheme: dark;
  --bg:#0d0d0d; --surface:#1a1a19;
  --ink:#f2f1ec; --muted:#c3c2b7; --faint:#898781;
  --grid:#2c2c2a; --baseline:#383835; --line:rgba(255,255,255,.10);
  --c-bwd:#3987e5; --c-bwd-old:#199e70; --c-fwd:#d95926;
  --c-bwd-wash:rgba(57,135,229,.16); --c-bwd-old-wash:rgba(25,158,112,.16);
  --c-fwd-wash:rgba(217,89,38,.16);
  --seq1:#0d366b; --seq2:#104281; --seq3:#1c5cab; --seq4:#256abf;
  --seq5:#3987e5; --seq6:#6da7ec; --seq7:#9ec5f4;
  --warn-wash:rgba(250,178,25,.13); --run-wash:rgba(57,135,229,.16);
  --ok-ink:#0ca30c; --bad-ink:#e66767;
  --r-red:#e66767; --r-blue:#3987e5; --r-green:#0ca30c; --r-yellow:#c98500;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 4px 14px rgba(0,0,0,.35);
"""

CSS = """
:root{
""" + _LIGHT_VARS + """
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
""" + _DARK_VARS + """
  }
}
:root[data-theme="dark"]{
""" + _DARK_VARS + """
}

*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:15.5px; line-height:1.62;
  -webkit-text-size-adjust:100%;
}
.wrap{max-width:1080px; margin:0 auto; padding:0 22px 90px}

/* ------------------------------------------------------------------ header */
header.masthead{padding:54px 0 8px}
.masthead .doctag{font-size:11.5px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--faint); margin-bottom:14px}
.masthead h1{font-size:33px; line-height:1.16; margin:0 0 12px; font-weight:700;
  letter-spacing:-.015em; max-width:24em}
.masthead .subtitle{font-size:16.5px; color:var(--muted); max-width:56em; margin:0 0 18px}
.masthead .meta{font-size:12.5px; color:var(--faint)}
.masthead .meta .sep{margin:0 .6em; opacity:.6}

/* ------------------------------------------------------------------ tabbar */
.tabwrap{position:sticky; top:0; z-index:40; background:var(--bg);
  border-bottom:1px solid var(--line); margin:26px -22px 0; padding:0 22px}
.tabbar{display:flex; gap:4px; max-width:1080px; margin:0 auto;
  overflow-x:auto; scrollbar-width:none}
.tabbar::-webkit-scrollbar{display:none}
.maintab{appearance:none; background:none; border:none; cursor:pointer;
  font:inherit; font-size:14px; color:var(--muted); padding:13px 13px 11px;
  border-bottom:2px solid transparent; white-space:nowrap}
.maintab:hover{color:var(--ink)}
.maintab.on{color:var(--ink); font-weight:600; border-bottom-color:var(--c-bwd)}
.maintab:focus-visible{outline:2px solid var(--c-bwd); outline-offset:-2px;
  border-radius:6px}
.tabnum{display:inline-block; font-size:11px; color:var(--faint);
  margin-right:6px; font-variant-numeric:tabular-nums}
.tabpanel{padding-top:8px}
.tabpanel:focus{outline:none}

/* -------------------------------------------------------------- structure */
section{margin:52px 0 0}
.sechead{margin-bottom:18px}
.kicker{font-size:11.5px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--faint); margin-bottom:6px}
h2{font-size:23px; line-height:1.25; margin:0; font-weight:700;
  letter-spacing:-.012em; max-width:30em}
h3{font-size:16.5px; margin:26px 0 8px; font-weight:650}
p{max-width:72ch; margin:.65em 0}
.lede{color:var(--muted); font-size:15.5px; max-width:72ch; margin-top:8px}
.small{font-size:13px}
.muted{color:var(--muted)}
.faint{color:var(--faint)}
b,strong{font-weight:650}
code,.mono{font-family:var(--mono); font-size:.92em}
a{color:inherit; text-decoration:underline; text-decoration-color:var(--baseline);
  text-underline-offset:2px}
a:hover{text-decoration-color:var(--ink)}
hr{border:none; border-top:1px solid var(--line); margin:40px 0}

/* ------------------------------------------------------------------- cards */
.panel{background:var(--surface); border:1px solid var(--line);
  border-radius:12px; padding:18px 20px; margin:16px 0}
.panel > :first-child{margin-top:0}
.panel > :last-child{margin-bottom:0}
.cols2{display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:stretch}
.cols3{display:grid; grid-template-columns:repeat(3,1fr); gap:16px}
@media (max-width:860px){.cols2,.cols3{grid-template-columns:1fr}}
.panel h3.ph{margin:0 0 8px; font-size:15px}
.panel .cellnote{margin-top:10px}

/* ---------------------------------------------------------------- KPI row */
.kpirow{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:20px 0}
@media (max-width:960px){.kpirow{grid-template-columns:repeat(2,1fr)}}
@media (max-width:520px){.kpirow{grid-template-columns:1fr}}
.tile{background:var(--surface); border:1px solid var(--line);
  border-radius:12px; padding:15px 17px}
.tile .tlabel{font-size:12px; color:var(--muted); margin-bottom:6px; line-height:1.4}
.tile .tvalue{font-size:31px; font-weight:650; letter-spacing:-.01em; line-height:1.1}
.tile .tvalue .vs{font-size:16px; color:var(--faint); font-weight:400; margin:0 3px}
.tile .tsub{font-size:12px; color:var(--faint); margin-top:7px; line-height:1.45}

/* ------------------------------------------------------------------ tables */
.scroll{overflow-x:auto; margin:14px 0; -webkit-overflow-scrolling:touch}
table{border-collapse:collapse; font-size:13.5px; min-width:100%;
  font-variant-numeric:tabular-nums}
th{font-size:11px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--faint); font-weight:600; text-align:left; padding:7px 10px;
  border-bottom:1px solid var(--baseline); vertical-align:bottom; line-height:1.35}
th.num{text-align:right}
td{padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top}
td.num{text-align:right; white-space:nowrap}
td.sep, th.sep{border-left:1px solid var(--grid)}
tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--c-bwd-wash)}
.rowgroup td{padding-top:16px; border-bottom:1px solid var(--baseline);
  font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--faint)}
td.win{font-weight:650}
td.win .windot{display:inline-block; width:7px; height:7px; border-radius:50%;
  margin-left:6px; vertical-align:1px}
.win-bwd .windot{background:var(--c-bwd)}
.win-fwd .windot{background:var(--c-fwd)}
.cellnote{font-size:12px; color:var(--faint); line-height:1.45; margin-top:3px;
  font-variant-numeric:normal; max-width:44em}
.syscell{min-width:15em}

/* ------------------------------------------------------------ chips & dots */
.dot{display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:7px; vertical-align:0}
.dot-bwd{background:var(--c-bwd)} .dot-fwd{background:var(--c-fwd)}
.dot-bwd-old{background:var(--c-bwd-old)}
.chip{display:inline-block; font-size:11px; font-weight:600; letter-spacing:.06em;
  text-transform:uppercase; padding:2px 8px; border-radius:20px;
  border:1px solid var(--line); color:var(--muted); vertical-align:1px}
.chip.warn{background:var(--warn-wash)}
.chip.run{background:var(--run-wash)}
.chip.okc{background:var(--c-bwd-old-wash)}
.pending{font-size:13px; color:var(--muted); background:var(--warn-wash);
  border:1px solid var(--line); border-radius:9px; padding:9px 13px; max-width:64em}
.pendtag{font-size:10.5px; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; margin-right:8px; color:var(--ink)}
.pendtag.run{color:var(--ink)}

/* ---------------------------------------------------------------- figures */
figure.chart{background:var(--surface); border:1px solid var(--line);
  border-radius:12px; padding:18px 18px 12px; margin:18px 0}
figure.chart svg{display:block; width:100%; height:auto}
figure.chart figcaption{font-size:12.5px; color:var(--muted); line-height:1.5;
  margin-top:10px; max-width:76em}
figure.chart figcaption .src{font-family:var(--mono); font-size:10.5px;
  color:var(--faint); display:block; margin-top:4px; word-break:break-all}
.charthead{margin:0 0 4px; font-size:14.5px; font-weight:650}
.chartsub{margin:0 0 12px; font-size:12.5px; color:var(--muted)}
.legend{display:flex; flex-wrap:wrap; gap:6px 18px; font-size:12.5px;
  color:var(--muted); margin:2px 0 10px}
.legend .key{display:inline-flex; align-items:center; gap:7px}
.legend .swatch{width:11px; height:11px; border-radius:3px}
.legend .lswatch{width:16px; height:2.5px; border-radius:2px}

/* board + plan art */
.boardrow{display:flex; gap:26px; flex-wrap:wrap; align-items:flex-start}
.boardrow .boardcol{flex:0 1 330px; min-width:270px}
.boardrow .textcol{flex:1 1 340px; min-width:280px}

/* ---------------------------------------------------------------- tooltip */
#tt{position:fixed; z-index:99; pointer-events:none; display:none;
  background:var(--surface); border:1px solid var(--line); border-radius:9px;
  box-shadow:var(--shadow); padding:8px 11px; font-size:12.5px; line-height:1.5;
  max-width:320px; font-variant-numeric:tabular-nums}
#tt .t0{font-weight:650; margin-bottom:2px}
#tt .tr{color:var(--muted)}
#tt .tr b{color:var(--ink); font-weight:650}
[data-tt]{cursor:default}
svg [data-tt]:hover{opacity:.92}

/* ------------------------------------------------------------- theme button */
.themebtn{position:fixed; top:10px; right:12px; z-index:60; appearance:none;
  font:inherit; font-size:11.5px; letter-spacing:.05em; color:var(--muted);
  background:var(--surface); border:1px solid var(--line); border-radius:20px;
  padding:4px 12px; cursor:pointer}
.themebtn:hover{color:var(--ink)}

/* ---------------------------------------------------------------- footnote */
.fnref{font-size:.72em; vertical-align:super; text-decoration:none;
  color:var(--c-bwd); font-weight:650}
ol.fnlist{font-size:13px; color:var(--muted); padding-left:1.4em}
ol.fnlist li{margin:.5em 0; max-width:78ch}

/* --------------------------------------------------------------- glossary */
dl.gloss{display:grid; grid-template-columns:max-content 1fr; gap:8px 22px;
  font-size:13.5px; max-width:80em}
dl.gloss dt{font-weight:650; white-space:nowrap}
dl.gloss dd{margin:0; color:var(--muted); max-width:66ch}
@media (max-width:640px){dl.gloss{grid-template-columns:1fr}
  dl.gloss dd{margin-bottom:8px}}

/* --------------------------------------------------------------- verdict */
.verdict{border-left:3px solid var(--c-bwd); padding:2px 0 2px 18px; margin:22px 0}
.verdict p{max-width:76ch}

.checkok{color:var(--ok-ink); font-weight:650}
.checkbad{color:var(--bad-ink); font-weight:650}

/* ------------------------------------------------------- table extras */
tr.hl td{background:var(--c-bwd-wash)}
tr.hl:hover td{background:var(--c-bwd-wash)}
sup.mnote{font-size:9px; color:var(--faint); margin-left:2px; letter-spacing:.04em}

/* details / disclosure */
details{margin:10px 0}
details summary{cursor:pointer; padding:6px 2px; list-style-position:inside}
details summary:hover{color:var(--ink)}
details.tabletwin{margin:8px 0 0}
details.tabletwin summary{font-size:12px}
details.rungdetail{background:var(--surface); border:1px solid var(--line);
  border-radius:12px; padding:6px 18px; margin:12px 0}
details.rungdetail summary{padding:10px 2px; font-size:15px}
details.rungdetail h3{font-size:13.5px; margin:16px 0 4px; color:var(--muted);
  font-weight:600}

/* worked example move list */
ol.movelist{list-style:none; counter-reset:mv; padding-left:0; font-size:13.5px;
  max-width:34em}
ol.movelist li{counter-increment:mv; margin:.35em 0; padding-left:2em;
  position:relative}
ol.movelist li::before{content:counter(mv); position:absolute; left:0; top:.1em;
  width:1.45em; height:1.45em; border:1px solid var(--baseline); color:var(--muted);
  border-radius:50%; font-size:.82em; display:flex; align-items:center;
  justify-content:center; font-variant-numeric:tabular-nums}
.rdot{display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:7px; border:1px solid var(--surface)}

/* plan diagram rows */
.planrow{display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap}
.planscroll{flex:1 1 420px; min-width:300px}
.planinset{flex:0 0 180px}
.planinset svg{max-width:170px !important}
@media (max-width:700px){.planinset{flex-basis:100%}}

/* plan-viz legend (markup emitted by eval/plan_viz_core.py legend()) */
.legend-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
  gap:6px 22px; margin:10px 0 18px; font-size:12.5px; color:var(--muted)}
.legend-grid > div{display:flex; align-items:center; gap:10px}
.legend-grid svg{flex:0 0 auto}

/* ------------------------------------------------------------------- print */
@media print{
  .tabwrap,.themebtn{display:none}
  .tabpanel[hidden]{display:block !important}
  .tabpanel{page-break-before:always}
  body{background:#fff}
}
@media (prefers-reduced-motion: reduce){
  html{scroll-behavior:auto}
}
"""

JS = """
(function(){
"use strict";
/* ---- theme toggle: auto -> light -> dark, persisted ---- */
var order=["auto","light","dark"];
var btn=document.getElementById("themebtn");
function applyTheme(t){
  if(t==="auto"){document.documentElement.removeAttribute("data-theme");}
  else{document.documentElement.setAttribute("data-theme",t);}
  if(btn){btn.textContent="theme: "+t;}
}
var saved="auto";
try{saved=localStorage.getItem("rr-theme")||"auto";}catch(e){}
applyTheme(saved);
if(btn){btn.addEventListener("click",function(){
  saved=order[(order.indexOf(saved)+1)%order.length];
  try{localStorage.setItem("rr-theme",saved);}catch(e){}
  applyTheme(saved);
});}

/* ---- tabs with hash deep-linking ---- */
var tabs=[].slice.call(document.querySelectorAll(".maintab"));
function panelOf(tab){return document.getElementById(tab.id.replace("tab-","panel-"));}
function select(tab,push){
  tabs.forEach(function(t){
    var on=(t===tab);
    t.classList.toggle("on",on);
    t.setAttribute("aria-selected",on?"true":"false");
    t.tabIndex=on?0:-1;
    panelOf(t).hidden=!on;
  });
  if(push!==false){
    try{history.replaceState(null,"","#"+tab.id.replace("tab-",""));}catch(e){}
  }
}
tabs.forEach(function(t,i){
  t.addEventListener("click",function(){select(t);});
  t.addEventListener("keydown",function(e){
    var d=(e.key==="ArrowRight")?1:(e.key==="ArrowLeft")?-1:0;
    if(!d)return;
    e.preventDefault();
    var n=tabs[(i+d+tabs.length)%tabs.length];
    n.focus(); select(n);
  });
});
function route(){
  var h=(location.hash||"").replace("#","");
  if(!h)return;
  var direct=document.getElementById("tab-"+h);
  if(direct){select(direct,false);return;}
  var el=document.getElementById(h);
  if(!el)return;
  var panel=el.closest(".tabpanel");
  if(panel){
    var tab=document.getElementById(panel.id.replace("panel-","tab-"));
    if(tab){select(tab,false);
      setTimeout(function(){el.scrollIntoView();},0);}
  }
}
window.addEventListener("hashchange",route);
route();
/* in-page links to sections in other tabs */
document.addEventListener("click",function(e){
  var a=e.target.closest&&e.target.closest("a[href^='#']");
  if(!a)return;
  var h=a.getAttribute("href").slice(1);
  var el=document.getElementById(h)||document.getElementById("tab-"+h);
  if(!el)return;
  e.preventDefault();
  try{history.replaceState(null,"","#"+h);}catch(err){}
  route();
});

/* ---- tooltip layer (data-tt carries newline-separated rows) ---- */
var tt=document.createElement("div");
tt.id="tt";
document.body.appendChild(tt);
var ttOn=false;
function fill(payload){
  while(tt.firstChild)tt.removeChild(tt.firstChild);
  payload.split("\\n").forEach(function(line,i){
    var row=document.createElement("div");
    row.className=i===0?"t0":"tr";
    var j=line.indexOf("\\t");
    if(i>0&&j>-1){
      row.appendChild(document.createTextNode(line.slice(0,j)+" "));
      var bnode=document.createElement("b");
      bnode.textContent=line.slice(j+1);
      row.appendChild(bnode);
    }else{
      row.textContent=line;
    }
    tt.appendChild(row);
  });
}
function move(e){
  var pad=14;
  var x=e.clientX+pad, y=e.clientY+pad;
  var r=tt.getBoundingClientRect();
  if(x+r.width>window.innerWidth-8)x=e.clientX-r.width-pad;
  if(y+r.height>window.innerHeight-8)y=e.clientY-r.height-pad;
  tt.style.left=x+"px"; tt.style.top=y+"px";
}
document.addEventListener("pointerover",function(e){
  var el=e.target.closest&&e.target.closest("[data-tt]");
  if(!el){if(ttOn){tt.style.display="none";ttOn=false;}return;}
  fill(el.getAttribute("data-tt"));
  tt.style.display="block"; ttOn=true; move(e);
});
document.addEventListener("pointermove",function(e){if(ttOn)move(e);});
document.addEventListener("pointerout",function(e){
  if(ttOn&&!(e.relatedTarget&&e.relatedTarget.closest&&
      e.relatedTarget.closest("[data-tt]"))){
    tt.style.display="none"; ttOn=false;
  }
});
/* keyboard: focusable marks show the same details */
document.addEventListener("focusin",function(e){
  var el=e.target.closest&&e.target.closest("[data-tt]");
  if(!el)return;
  fill(el.getAttribute("data-tt"));
  var r=el.getBoundingClientRect();
  tt.style.display="block"; ttOn=true;
  tt.style.left=Math.min(r.left,window.innerWidth-340)+"px";
  tt.style.top=(r.bottom+10)+"px";
});
document.addEventListener("focusout",function(){
  if(ttOn){tt.style.display="none";ttOn=false;}
});
})();
"""
