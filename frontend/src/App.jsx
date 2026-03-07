import { useState, useRef, useCallback, useEffect, useMemo } from "react";

/* ── FONTS ───────────────────────────────────────────────────────────────────── */
const _fl = document.createElement("link");
_fl.rel = "stylesheet";
_fl.href = "https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&family=DM+Sans:wght@400;500;700&display=swap";
document.head.appendChild(_fl);

/* ── TOKENS ──────────────────────────────────────────────────────────────────── */
const C = {
  ink:"#04040a", paper:"#080810", panel:"#0d0d18", card:"#11111e",
  rule:"#1a1a2e", dim:"#2a2a44", muted:"#505070", body:"#9898b8",
  bright:"#d8d8f0", white:"#f0f0ff",
  lime:"#a8ff3e", red:"#ff2d55", amber:"#ff9500", blue:"#0af", purple:"#bf5fff",
  pink:"#ff2d8f",
  D:"'Bebas Neue',sans-serif", M:"'Space Mono',monospace", S:"'DM Sans',sans-serif",
};

/* ── GLOBAL CSS ──────────────────────────────────────────────────────────────── */
const _gs = document.createElement("style");
_gs.textContent = `
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html,body{background:${C.ink};color:${C.body};font-family:${C.S};min-height:100vh;overflow-x:hidden;cursor:none}
  ::selection{background:${C.lime};color:${C.ink}}
  ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:${C.dim}}
  select option{background:${C.panel};color:${C.bright}}

  @keyframes ticker  {from{transform:translateX(0)}to{transform:translateX(-50%)}}
  @keyframes blink   {0%,49%,100%{opacity:1}50%,99%{opacity:0}}
  @keyframes fadeUp  {from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:none}}
  @keyframes fadeIn  {from{opacity:0}to{opacity:1}}
  @keyframes popIn   {from{opacity:0;transform:scale(.82) translateY(12px)}to{opacity:1;transform:none}}
  @keyframes countUp {from{opacity:0;transform:translateY(14px) scale(.9)}to{opacity:1;transform:none}}
  @keyframes shimmer {0%{left:-80%}100%{left:160%}}
  @keyframes pulse   {0%,100%{opacity:.5;transform:scale(1)}50%{opacity:1;transform:scale(1.06)}}
  @keyframes spinSlow{from{transform:rotate(0)}to{transform:rotate(360deg)}}
  @keyframes glitch  {
    0%,90%,100%{transform:none;clip-path:none}
    91%{transform:translate(-4px,2px);clip-path:inset(10% 0 75% 0)}
    93%{transform:translate(4px,-2px);clip-path:inset(70% 0 10% 0)}
    95%{transform:translate(-2px,4px);clip-path:inset(42% 0 42% 0)}
    96%{transform:none;clip-path:none}
  }
  @keyframes float   {0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
  @keyframes ripple  {0%{transform:scale(0);opacity:.5}100%{transform:scale(4);opacity:0}}
  @keyframes borderGlow{0%,100%{box-shadow:0 0 0 1px ${C.lime}00}50%{box-shadow:0 0 0 1px ${C.lime}44,0 0 30px ${C.lime}11}}
  @keyframes slideIn {from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:none}}
  @keyframes scanDown{0%{transform:translateY(-100%)}100%{transform:translateY(100vh)}}
  @keyframes waveX   {0%,100%{transform:scaleX(1) translateY(0)}50%{transform:scaleX(1.04) translateY(-3px)}}
  @keyframes aurora  {
    0%  {background-position:0% 50%}
    50% {background-position:100% 50%}
    100%{background-position:0% 50%}
  }
  @keyframes typeIn  {from{width:0}to{width:100%}}
  @keyframes cursorBlink{0%,100%{opacity:1}50%{opacity:0}}
  @keyframes dataRain{0%{transform:translateY(-120px);opacity:0}10%{opacity:1}85%{opacity:.8}100%{transform:translateY(110vh);opacity:0}}

  .fu  {animation:fadeUp  .55s cubic-bezier(.22,1,.36,1) both}
  .fi  {animation:fadeIn  .4s ease both}
  .pop {animation:popIn   .5s cubic-bezier(.34,1.56,.64,1) both}
  .cu  {animation:countUp .75s cubic-bezier(.22,1,.36,1) both}
  .si  {animation:slideIn .45s cubic-bezier(.22,1,.36,1) both}

  .tilt3d {
    transition:transform .12s ease,box-shadow .12s ease;
    transform-style:preserve-3d;will-change:transform;
  }
  .pill {
    display:flex;align-items:center;gap:7px;padding:7px 13px;
    cursor:none;transition:all .18s cubic-bezier(.22,1,.36,1);
    user-select:none;border:1px solid ${C.rule};
    font-family:${C.M};font-size:11px;position:relative;overflow:hidden;
  }
  .pill:hover{border-color:${C.dim};transform:scale(1.03) translateY(-1px)}
  .pill:active{transform:scale(.96)}

  .glow-btn {
    font-family:${C.M};font-weight:700;letter-spacing:.14em;cursor:none;
    border:none;position:relative;overflow:hidden;transition:all .22s cubic-bezier(.22,1,.36,1);
  }
  .glow-btn::before{
    content:'';position:absolute;top:0;left:-80%;width:40%;height:100%;
    background:linear-gradient(90deg,transparent,#ffffff55,transparent);
    animation:shimmer 2.2s 1s infinite;
  }
  .glow-btn:hover{filter:brightness(1.18);transform:translateY(-3px) scale(1.02)}
  .glow-btn:active{transform:translateY(0) scale(.98)}

  .row-tr{transition:background .1s,transform .1s}
  .row-tr:hover{background:#14142a !important;transform:translateX(2px)}

  .nav-btn{
    font-family:${C.M};font-size:10px;letter-spacing:.1em;cursor:none;
    background:transparent;transition:all .18s;padding:7px 14px;
  }
  .nav-btn:hover{color:${C.bright} !important;border-color:${C.dim} !important;transform:translateY(-1px)}
`;
document.head.appendChild(_gs);

/* ── FEATURES ────────────────────────────────────────────────────────────────── */
const FEATURES = [
  { key:"customerID",     label:"CUSTOMER ID",    tag:"ID",   color:C.blue,   required:true,
    desc:"Row identifier", hint:"First selected column used.",
    aliases:["customerid","customer_id","id","userid","user_id","cust_id","custid"],
    keywords:["id","customer","user","cust","custid","userid"] },
  { key:"churnRisk",      label:"CHURN RISK",     tag:"RISK", color:C.red,    required:true,
    desc:"Probability or risk tier", hint:"Numeric cols averaged; categorical auto-mapped.",
    aliases:["churnprobability","churn_probability","churn_prob","churnprob","churn_score","churnscore","probability","risktier","risk_tier","risk","risk_level","churn_tier","churnrisk","churn_risk"],
    keywords:["churn","risk","probability","prob","attrition","likelihood"] },
  { key:"tenure",         label:"TENURE",         tag:"TEN",  color:C.amber,  required:false,
    desc:"Time active with you", hint:"Cols averaged per customer.",
    aliases:["tenure","months_active","account_age","customer_age","age","months","days_active","subscription_age","length_of_stay","customer_tenure"],
    keywords:["tenure","seniority","onboard","duration"] },
  { key:"engagement",     label:"ENGAGEMENT",     tag:"ENG",  color:C.lime,   required:false,
    desc:"Activity level", hint:"Cols summed then normalised.",
    aliases:["engagement","engagement_score","activity","activity_score","interaction_score","usage_score","activitylevel","logins","login_count","sessions","watch_time","session_duration","page_views"],
    keywords:["engagement","engage","logins","sessions","clicks","interaction","activity","visits","frequency","playtime","watchtime","pageviews"] },
  { key:"retentionValue", label:"RETENTION VALUE",tag:"VAL",  color:C.purple, required:false,
    desc:"Monetary value if retained", hint:"Cols summed per customer.",
    aliases:["expectedretentiongain","expected_retention_gain","retention_gain","retentiongain","retention_value","ltv","clv","lifetime_value","revenue","mrr","arpu"],
    keywords:["retention","gain","ltv","clv","arpu","mrr","revenue","profit","lifetime","spend"] },
];

/* ── LOGIC HELPERS ───────────────────────────────────────────────────────────── */
function segs(h){return h.replace(/([a-z])([A-Z])/g,"$1_$2").toLowerCase().split(/[_\s]+/).filter(Boolean)}
function relevantCols(headers,feat){return headers.filter(h=>feat.keywords.some(kw=>segs(h).includes(kw)))}
function autoDetect(headers){
  const m={},lower=headers.map(h=>h.toLowerCase().replace(/\s+/g,"_"));
  FEATURES.forEach(f=>{const matched=headers.filter((_,i)=>f.aliases.includes(lower[i]));if(matched.length)m[f.key]=matched;});
  return m;
}
function parseCSV(text){
  const lines=text.trim().split("\n");if(lines.length<2)return{headers:[],rows:[]};
  const headers=lines[0].split(",").map(h=>h.trim().replace(/"/g,""));
  const rows=lines.slice(1).map(line=>{
    const vals=line.split(",").map(v=>v.trim().replace(/"/g,""));
    const obj={};headers.forEach((h,i)=>{obj[h]=vals[i]??"";});return obj;
  }).filter(r=>r[headers[0]]);
  return{headers,rows};
}
function isCat(rows,col){return new Set(rows.slice(0,100).map(r=>r[col])).size<=10}
function riskNum(val){
  const v=(val||"").toLowerCase();
  if(v.includes("high"))return .85;if(v.includes("med"))return .5;
  if(v.includes("low")||v.includes("safe"))return .15;
  const n=parseFloat(val);return isNaN(n)?null:n;
}
function avgC(row,cols){const v=cols.map(c=>parseFloat(row[c])).filter(n=>!isNaN(n));return v.length?v.reduce((a,b)=>a+b)/v.length:null}
function sumC(row,cols){const v=cols.map(c=>parseFloat(row[c])).filter(n=>!isNaN(n));return v.length?v.reduce((a,b)=>a+b):null}

function buildStats(rows,mapping){
  const total=rows.length;
  const rc=mapping.churnRisk||[];
  let high=0,medium=0,safe=0,cv=[];
  if(rc.length){
    const s100=rows.slice(0,100);
    const cat=rc.filter(c=>isCat(s100,c)),num=rc.filter(c=>!isCat(s100,c));
    rows.forEach(r=>{
      const ns=[...num.map(c=>parseFloat(r[c])).filter(n=>!isNaN(n)),...cat.map(c=>riskNum(r[c])).filter(n=>n!==null)];
      const s=ns.length?ns.reduce((a,b)=>a+b)/ns.length:null;
      if(s!==null)cv.push(s);
      const cv2=cat.map(c=>(r[c]||"").toLowerCase());
      if(cv2.some(v=>v.includes("high"))||(s!==null&&s>=.65))high++;
      else if(cv2.some(v=>v.includes("med"))||(s!==null&&s>=.35))medium++;
      else safe++;
    });
  }
  const avgChurn=cv.length?(cv.reduce((a,b)=>a+b)/cv.length*100).toFixed(1):null;
  const tc=mapping.tenure||[];
  let tenureBuckets=null;
  if(tc.length){
    const b={"0–3":0,"3–6":0,"6–12":0,"12–24":0,"24+":0};
    rows.forEach(r=>{const v=avgC(r,tc);if(v===null)return;if(v<=3)b["0–3"]++;else if(v<=6)b["3–6"]++;else if(v<=12)b["6–12"]++;else if(v<=24)b["12–24"]++;else b["24+"]++;});
    tenureBuckets=Object.entries(b).map(([l,v])=>({label:l,value:v}));
  }
  const ec=mapping.engagement||[];
  let engBuckets=null;
  if(ec.length){
    const vs=rows.map(r=>sumC(r,ec)).filter(v=>v!==null);
    if(vs.length){
      const mn=Math.min(...vs),mx=Math.max(...vs),rng=mx-mn||1;
      const b={LOW:0,MED:0,HIGH:0};
      vs.forEach(v=>{const p=(v-mn)/rng;if(p<.33)b.LOW++;else if(p<.66)b.MED++;else b.HIGH++;});
      engBuckets=Object.entries(b).map(([l,v])=>({label:l,value:v}));
    }
  }
  const vc=mapping.retentionValue||[];
  let totalRet=null;
  if(vc.length){const vs=rows.map(r=>sumC(r,vc)).filter(v=>v!==null);if(vs.length)totalRet=vs.reduce((a,b)=>a+b);}
  const retDisplay=totalRet!==null?(totalRet>=1e6?"$"+(totalRet/1e6).toFixed(1)+"M":"$"+(totalRet/1000).toFixed(0)+"K"):null;
  return{total,high,medium,safe,avgChurn,tenureBuckets,engBuckets,retDisplay};
}

/* ── CUSTOM CURSOR ───────────────────────────────────────────────────────────── */
function Cursor(){
  const dot=useRef(),ring=useRef();
  const pos=useRef({x:0,y:0});
  const ring_pos=useRef({x:0,y:0});
  const clicked=useRef(false);
  useEffect(()=>{
    const move=e=>{pos.current={x:e.clientX,y:e.clientY};};
    const down=()=>{clicked.current=true;};
    const up=()=>{clicked.current=false;};
    window.addEventListener("mousemove",move);
    window.addEventListener("mousedown",down);
    window.addEventListener("mouseup",up);
    let raf;
    const animate=()=>{
      ring_pos.current.x+=(pos.current.x-ring_pos.current.x)*.14;
      ring_pos.current.y+=(pos.current.y-ring_pos.current.y)*.14;
      if(dot.current){
        dot.current.style.transform=`translate(${pos.current.x-4}px,${pos.current.y-4}px) scale(${clicked.current?.5:1})`;
      }
      if(ring.current){
        ring.current.style.transform=`translate(${ring_pos.current.x-18}px,${ring_pos.current.y-18}px) scale(${clicked.current?1.6:1})`;
      }
      raf=requestAnimationFrame(animate);
    };
    animate();
    return()=>{
      window.removeEventListener("mousemove",move);
      window.removeEventListener("mousedown",down);
      window.removeEventListener("mouseup",up);
      cancelAnimationFrame(raf);
    };
  },[]);
  return (
    <>
      <div ref={dot} style={{position:"fixed",top:0,left:0,width:8,height:8,borderRadius:"50%",background:C.lime,pointerEvents:"none",zIndex:99999,mixBlendMode:"screen",transition:"transform .06s"}}/>
      <div ref={ring} style={{position:"fixed",top:0,left:0,width:36,height:36,borderRadius:"50%",border:`1.5px solid ${C.lime}88`,pointerEvents:"none",zIndex:99998,transition:"transform .06s"}}/>
    </>
  );
}

/* ── AURORA MESH BACKGROUND ──────────────────────────────────────────────────── */
function AuroraBg({intensity=1}){
  const canvasRef=useRef();
  const mouse=useRef({x:0.5,y:0.5});
  useEffect(()=>{
    const canvas=canvasRef.current;
    const ctx=canvas.getContext("2d");
    let W,H,raf;
    const orbs=[
      {x:.15,y:.3,r:.55,color:"#a8ff3e",vx:.0003,vy:.0002},
      {x:.8,y:.2,r:.45,color:"#ff2d55",vx:-.0002,vy:.0003},
      {x:.5,y:.75,r:.5,color:"#0aaff0",vx:.0002,vy:-.0002},
      {x:.3,y:.6,r:.4,color:"#bf5fff",vx:-.0003,vy:-.0003},
      {x:.75,y:.65,r:.38,color:"#ff9500",vx:.0002,vy:.0002},
    ];
    const resize=()=>{W=canvas.width=canvas.offsetWidth;H=canvas.height=canvas.offsetHeight;};
    resize();window.addEventListener("resize",resize);
    const onMouse=e=>{mouse.current={x:e.clientX/window.innerWidth,y:e.clientY/window.innerHeight};};
    window.addEventListener("mousemove",onMouse);
    let t=0;
    const draw=()=>{
      t+=.004;
      ctx.clearRect(0,0,W,H);
      ctx.fillStyle=C.ink;ctx.fillRect(0,0,W,H);
      orbs.forEach((o,i)=>{
        o.x+=o.vx+Math.sin(t+i)*.0002;
        o.y+=o.vy+Math.cos(t+i*.7)*.0002;
        if(o.x<0||o.x>1)o.vx*=-1;
        if(o.y<0||o.y>1)o.vy*=-1;
        // mouse attraction
        const mx=mouse.current.x,my=mouse.current.y;
        o.x+=(mx-o.x)*.0003;o.y+=(my-o.y)*.0003;

        const grd=ctx.createRadialGradient(o.x*W,o.y*H,0,o.x*W,o.y*H,o.r*Math.min(W,H));
        grd.addColorStop(0,o.color+"22");
        grd.addColorStop(.5,o.color+"0a");
        grd.addColorStop(1,"transparent");
        ctx.beginPath();
        ctx.ellipse(o.x*W,o.y*H,o.r*W*.5,o.r*H*.5,t*0.1+i,0,Math.PI*2);
        ctx.fillStyle=grd;ctx.fill();
      });
      // grid overlay
      ctx.strokeStyle=`${C.rule}66`;ctx.lineWidth=.5;
      const gs=60;
      for(let x=0;x<W;x+=gs){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
      for(let y=0;y<H;y+=gs){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
      raf=requestAnimationFrame(draw);
    };
    draw();
    return()=>{cancelAnimationFrame(raf);window.removeEventListener("resize",resize);window.removeEventListener("mousemove",onMouse);};
  },[]);
  return <canvas ref={canvasRef} style={{position:"fixed",inset:0,width:"100%",height:"100%",pointerEvents:"none",zIndex:0,opacity:intensity}}/>;
}

/* ── PARTICLE NETWORK ────────────────────────────────────────────────────────── */
function ParticleNet(){
  const canvasRef=useRef();
  const mouse=useRef({x:-999,y:-999});
  useEffect(()=>{
    const canvas=canvasRef.current;
    const ctx=canvas.getContext("2d");
    let W,H,raf;
    const pts=[];
    const resize=()=>{W=canvas.width=canvas.offsetWidth;H=canvas.height=canvas.offsetHeight;};
    resize();window.addEventListener("resize",resize);
    const onMouse=e=>{const r=canvas.getBoundingClientRect();mouse.current={x:e.clientX-r.left,y:e.clientY-r.top};};
    window.addEventListener("mousemove",onMouse);
    for(let i=0;i<90;i++)pts.push({
      x:Math.random()*1400,y:Math.random()*900,
      vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,
      r:Math.random()*1.2+.4,alpha:Math.random()*.4+.1,
    });
    const draw=()=>{
      ctx.clearRect(0,0,W,H);
      const {x:mx,y:my}=mouse.current;
      pts.forEach(p=>{
        const dx=p.x-mx,dy=p.y-my,d=Math.sqrt(dx*dx+dy*dy);
        if(d<130){const f=(130-d)/130*.6;p.vx+=dx/d*f*.1;p.vy+=dy/d*f*.1;}
        p.vx*=.97;p.vy*=.97;
        p.x+=p.vx;p.y+=p.vy;
        if(p.x<0)p.x=W;if(p.x>W)p.x=0;
        if(p.y<0)p.y=H;if(p.y>H)p.y=0;
      });
      for(let i=0;i<pts.length;i++){
        for(let j=i+1;j<pts.length;j++){
          const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);
          if(d<110){
            ctx.strokeStyle=`rgba(168,255,62,${(1-d/110)*.15})`;
            ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.stroke();
          }
        }
        const d=Math.sqrt((pts[i].x-mx)**2+(pts[i].y-my)**2);
        const a=d<160?(pts[i].alpha+(1-d/160)*.7):pts[i].alpha;
        ctx.beginPath();ctx.arc(pts[i].x,pts[i].y,pts[i].r,0,Math.PI*2);
        ctx.fillStyle=`rgba(168,255,62,${a})`;ctx.fill();
      }
      raf=requestAnimationFrame(draw);
    };
    draw();
    return()=>{cancelAnimationFrame(raf);window.removeEventListener("resize",resize);window.removeEventListener("mousemove",onMouse);};
  },[]);
  return <canvas ref={canvasRef} style={{position:"absolute",inset:0,width:"100%",height:"100%",pointerEvents:"none",zIndex:1}}/>;
}

/* ── RIPPLE EFFECT ───────────────────────────────────────────────────────────── */
function useRipple(){
  const [ripples,setRipples]=useState([]);
  const addRipple=useCallback((e)=>{
    const r=e.currentTarget.getBoundingClientRect();
    const id=Date.now();
    setRipples(p=>[...p,{id,x:e.clientX-r.left,y:e.clientY-r.top}]);
    setTimeout(()=>setRipples(p=>p.filter(r=>r.id!==id)),700);
  },[]);
  const rippleEls=ripples.map(r=>(
    <div key={r.id} style={{position:"absolute",left:r.x-20,top:r.y-20,width:40,height:40,borderRadius:"50%",background:C.lime,animation:"ripple .7s ease-out forwards",pointerEvents:"none",zIndex:10}}/>
  ));
  return{addRipple,rippleEls};
}

/* ── 3D TILT CARD ────────────────────────────────────────────────────────────── */
function TiltCard({children,style={},color=C.lime,className=""}){
  const ref=useRef();
  const onMove=useCallback(e=>{
    if(!ref.current)return;
    const r=ref.current.getBoundingClientRect();
    const x=(e.clientX-r.left)/r.width-.5;
    const y=(e.clientY-r.top)/r.height-.5;
    ref.current.style.transform=`perspective(600px) rotateX(${-y*10}deg) rotateY(${x*10}deg) translateY(-4px) scale(1.02)`;
    ref.current.style.boxShadow=`${-x*20}px ${-y*20}px 40px #00000066, 0 0 30px ${color}18`;
    ref.current.style.setProperty("--gx",`${(x+.5)*100}%`);
    ref.current.style.setProperty("--gy",`${(y+.5)*100}%`);
    ref.current.style.setProperty("--ga","1");
  },[color]);
  const onLeave=useCallback(()=>{
    if(!ref.current)return;
    ref.current.style.transform="perspective(600px) rotateX(0) rotateY(0) translateY(0) scale(1)";
    ref.current.style.boxShadow="none";
    ref.current.style.setProperty("--ga","0");
  },[]);
  return (
    <div ref={ref} onMouseMove={onMove} onMouseLeave={onLeave} className={`tilt3d ${className}`}
      style={{position:"relative",overflow:"hidden","--gx":"50%","--gy":"50%","--ga":"0",transition:"transform .15s ease, box-shadow .15s ease",...style}}>
      <div style={{position:"absolute",inset:0,pointerEvents:"none",zIndex:1,
        background:`radial-gradient(220px circle at var(--gx) var(--gy), ${color}12 0%, transparent 70%)`,
        opacity:"var(--ga)",transition:"opacity .3s"}}/>
      <div style={{position:"relative",zIndex:2}}>{children}</div>
    </div>
  );
}

/* ── TYPEWRITER ──────────────────────────────────────────────────────────────── */
function Typewriter({words,speed=80}){
  const [idx,setIdx]=useState(0);
  const [chars,setChars]=useState(0);
  const [deleting,setDeleting]=useState(false);
  useEffect(()=>{
    const word=words[idx%words.length];
    const delay=deleting?40:speed;
    const t=setTimeout(()=>{
      if(!deleting&&chars<word.length)setChars(c=>c+1);
      else if(!deleting&&chars===word.length)setTimeout(()=>setDeleting(true),1400);
      else if(deleting&&chars>0)setChars(c=>c-1);
      else{setDeleting(false);setIdx(i=>(i+1)%words.length);}
    },delay);
    return()=>clearTimeout(t);
  },[chars,deleting,idx,words,speed]);
  const word=words[idx%words.length];
  return (
    <span style={{color:C.lime}}>
      {word.slice(0,chars)}
      <span style={{animation:"cursorBlink 1s infinite",color:C.lime}}>|</span>
    </span>
  );
}

/* ── ANIMATED NUMBER ─────────────────────────────────────────────────────────── */
function Ticker({to,prefix="",suffix="",decimals=0}){
  const num=parseFloat(String(to).replace(/[^0-9.]/g,""))||0;
  const [val,setVal]=useState(0);
  useEffect(()=>{
    setVal(0);let i=0;const steps=72,dur=1400;
    const t=setInterval(()=>{
      i++;const p=1-Math.pow(1-i/steps,4);setVal(num*p);
      if(i>=steps){setVal(num);clearInterval(t);}
    },dur/steps);
    return()=>clearInterval(t);
  },[num]);
  return <>{prefix}{decimals>0?val.toFixed(decimals):Math.round(val).toLocaleString()}{suffix}</>;
}

/* ── MARQUEE ─────────────────────────────────────────────────────────────────── */
function Marquee({stats,fileName}){
  const items=[
    `FILE: ${fileName||"—"}`,`CUSTOMERS: ${stats?.total?.toLocaleString()||"—"}`,
    `HIGH RISK: ${stats?.high?.toLocaleString()||"—"}`,`MEDIUM: ${stats?.medium?.toLocaleString()||"—"}`,
    `SAFE: ${stats?.safe?.toLocaleString()||"—"}`,`AVG CHURN: ${stats?.avgChurn||"—"}%`,
    stats?.retDisplay?`RETENTION: ${stats.retDisplay}`:null,
  ].filter(Boolean).join("  ◆  ");
  return (
    <div style={{background:C.lime,color:C.ink,overflow:"hidden",height:26,display:"flex",alignItems:"center",fontFamily:C.M,fontSize:10,letterSpacing:".08em",fontWeight:700,whiteSpace:"nowrap"}}>
      <div style={{display:"flex",animation:"ticker 20s linear infinite"}}>
        {[items,items].map((t,i)=><span key={i} style={{paddingRight:80}}>{t}</span>)}
      </div>
    </div>
  );
}

/* ── DONUT ───────────────────────────────────────────────────────────────────── */
function Donut({high,medium,safe,total}){
  const [on,setOn]=useState(false);
  const [hov,setHov]=useState(null);
  useEffect(()=>{setTimeout(()=>setOn(true),400);},[]);
  const sz=200,cx=100,cy=100,r=72,sw=20,circ=2*Math.PI*r;
  const segs=[
    {pct:total?high/total:0,off:0,color:C.red,label:"HIGH"},
    {pct:total?medium/total:0,off:total?high/total:0,color:C.amber,label:"MEDIUM"},
    {pct:total?safe/total:0,off:total?(high+medium)/total:0,color:C.lime,label:"SAFE"},
  ];
  return (
    <svg width={sz} height={sz} style={{overflow:"visible",filter:hov!==null?`drop-shadow(0 0 12px ${segs[hov].color}66)`:"none",transition:"filter .3s"}}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.rule} strokeWidth={sw}/>
      {segs.map((s,i)=>(
        <circle key={i} cx={cx} cy={cy} r={hov===i?r+4:r} fill="none"
          stroke={s.color} strokeWidth={hov===i?sw+5:sw}
          strokeDasharray={`${(on?s.pct:0)*circ} ${circ}`}
          strokeDashoffset={-s.off*circ}
          style={{transform:"rotate(-90deg)",transformOrigin:`${cx}px ${cy}px`,
            transition:"stroke-dasharray 1.4s cubic-bezier(.22,1,.36,1), r .25s, stroke-width .25s",cursor:"none"}}
          onMouseEnter={()=>setHov(i)} onMouseLeave={()=>setHov(null)}/>
      ))}
      <circle cx={cx} cy={cy} r={r-sw/2-4} fill={C.panel}/>
      <text x={cx} y={cy-8} textAnchor="middle" fill={C.muted} fontSize="9" fontFamily={C.M} letterSpacing=".15em">TOTAL</text>
      <text x={cx} y={cy+16} textAnchor="middle" fill={hov!==null?segs[hov].color:C.white} fontSize="28" fontFamily={C.D} style={{transition:"fill .2s"}}>
        {hov!==null?(total?((segs[hov].pct)*100).toFixed(1)+"%":"0%"):(total>=1000?(total/1000).toFixed(0)+"K":total)}
      </text>
      {hov!==null&&<text x={cx} y={cy+32} textAnchor="middle" fill={segs[hov].color} fontSize="9" fontFamily={C.M} letterSpacing=".1em">{segs[hov].label}</text>}
    </svg>
  );
}

/* ── BAR CHART ───────────────────────────────────────────────────────────────── */
function Bar({items,color}){
  const [on,setOn]=useState(false);
  const [hov,setHov]=useState(null);
  useEffect(()=>{setTimeout(()=>setOn(true),300);},[]);
  const max=Math.max(...items.map(i=>i.value),1);
  return (
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      {items.map((it,i)=>(
        <div key={i} className="fu" style={{animationDelay:`${i*80}ms`}}
          onMouseEnter={()=>setHov(i)} onMouseLeave={()=>setHov(null)}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
            <span style={{fontFamily:C.M,fontSize:10,color:hov===i?C.bright:C.muted,letterSpacing:".06em",transition:"color .15s"}}>{it.label}</span>
            <span style={{fontFamily:C.M,fontSize:10,color:hov===i?color:C.muted,fontWeight:700,transition:"color .15s",textShadow:hov===i?`0 0 10px ${color}`:"none"}}>{it.value.toLocaleString()}</span>
          </div>
          <div style={{height:hov===i?10:6,background:C.rule,position:"relative",overflow:"hidden",borderRadius:2,transition:"height .2s cubic-bezier(.22,1,.36,1)"}}>
            <div style={{
              position:"absolute",top:0,left:0,height:"100%",
              background:hov===i?`linear-gradient(90deg,${color}cc,${color})`:`${color}77`,
              width:on?`${(it.value/max)*100}%`:"0%",
              transition:`width 1.1s cubic-bezier(.22,1,.36,1) ${i*80}ms, background .2s`,
              boxShadow:hov===i?`0 0 16px ${color}88`:"none",
            }}/>
            {hov===i&&<div style={{position:"absolute",top:0,left:"-40%",width:"30%",height:"100%",background:"linear-gradient(90deg,transparent,#ffffff44,transparent)",animation:"shimmer 1.2s infinite"}}/>}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── STAT CARD ───────────────────────────────────────────────────────────────── */
function StatCard({label,value,sub,color,index=0}){
  const isNum=!isNaN(parseFloat(String(value).replace(/[^0-9.]/g,"")));
  const {addRipple,rippleEls}=useRipple();
  return (
    <TiltCard color={color} className="fu" style={{
      animationDelay:`${index*70}ms`,
      background:C.card,
      border:`1px solid ${C.rule}`,
      borderTop:`3px solid ${color}`,
      padding:"24px 20px",
      cursor:"none",
    }}
    onClick={addRipple}>
      {rippleEls}
      <div style={{position:"absolute",top:-30,right:-30,width:100,height:100,borderRadius:"50%",background:color,opacity:.05,filter:"blur(30px)",pointerEvents:"none"}}/>
      <div style={{position:"absolute",bottom:0,left:0,right:0,height:1,background:`linear-gradient(90deg,transparent,${color}33,transparent)`}}/>
      <div style={{fontFamily:C.M,fontSize:9,letterSpacing:".2em",color:C.muted,marginBottom:14,fontWeight:700}}>{label}</div>
      <div className="cu" style={{animationDelay:`${index*70+130}ms`,fontFamily:C.D,fontSize:46,color,lineHeight:1,letterSpacing:".02em",textShadow:`0 0 20px ${color}44`}}>
        {isNum
          ? <Ticker to={value}
              prefix={String(value).startsWith("$")?"$":""}
              suffix={String(value).endsWith("%")?"%":String(value).endsWith("K")?"K":String(value).endsWith("M")?"M":""}
              decimals={String(value).includes(".")?1:0}/>
          : value}
      </div>
      {sub&&<div style={{fontFamily:C.M,fontSize:10,color:C.muted,marginTop:8,letterSpacing:".04em"}}>{sub} of total</div>}
    </TiltCard>
  );
}

/* ── DATA RAIN (upload BG) ───────────────────────────────────────────────────── */
function DataRain(){
  const chars="01アイウエオ></>{}[]◆◈◎◬";
  const cols=useMemo(()=>Array.from({length:30},(_,i)=>({
    id:i,left:`${(i/30)*100+Math.random()*2}%`,
    delay:`${Math.random()*5}s`,dur:`${2.5+Math.random()*3}s`,
    chars:Array.from({length:20},()=>chars[Math.floor(Math.random()*chars.length)]),
    opacity:.05+Math.random()*.1,
    color:[C.lime,C.blue,C.purple,C.red][Math.floor(Math.random()*4)],
    fontSize:10+Math.random()*4,
  })),[]);
  return (
    <div style={{position:"absolute",inset:0,overflow:"hidden",pointerEvents:"none",zIndex:1}}>
      {cols.map(col=>(
        <div key={col.id} style={{
          position:"absolute",top:0,left:col.left,
          fontFamily:C.M,fontSize:col.fontSize,color:col.color,opacity:col.opacity,
          display:"flex",flexDirection:"column",gap:3,
          animation:`dataRain ${col.dur} ${col.delay} infinite linear`,
          letterSpacing:".05em",
        }}>
          {col.chars.map((ch,i)=><span key={i}>{ch}</span>)}
        </div>
      ))}
    </div>
  );
}

/* ── UPLOAD ──────────────────────────────────────────────────────────────────── */
function Upload({onFile}){
  const ref=useRef();
  const [drag,setDrag]=useState(false);
  const [scanning,setScanning]=useState(false);
  const {addRipple,rippleEls}=useRipple();
  const handle=useCallback(file=>{
    if(!file||!file.name.endsWith(".csv"))return;
    setScanning(true);
    const r=new FileReader();
    r.onload=e=>{setTimeout(()=>{setScanning(false);onFile(file.name,e.target.result);},1000);};
    r.readAsText(file);
  },[onFile]);

  return (
    <div style={{minHeight:"calc(100vh - 54px)",display:"flex",position:"relative",overflow:"hidden"}}>
      <AuroraBg intensity={.6}/>
      <ParticleNet/>
      {scanning&&(
        <div style={{position:"fixed",inset:0,zIndex:200,background:`${C.ink}cc`,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",backdropFilter:"blur(8px)"}}>
          <div style={{animation:"float 1.6s ease-in-out infinite"}}>
            <div style={{fontFamily:C.D,fontSize:72,color:C.lime,letterSpacing:".04em",textShadow:`0 0 40px ${C.lime}88`,marginBottom:8}}>SCANNING</div>
            <div style={{fontFamily:C.M,fontSize:12,color:C.lime,letterSpacing:".25em",textAlign:"center"}}>
              PROCESSING<span style={{animation:"blink 1s infinite"}}>_</span>
            </div>
            <div style={{marginTop:28,display:"flex",justifyContent:"center",gap:8}}>
              {[0,1,2,3,4,5,6].map(i=>(
                <div key={i} style={{width:5,height:5,background:C.lime,borderRadius:"50%",animation:`pulse .9s ${i*.12}s infinite`}}/>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* left hero */}
      <div style={{flex:"0 0 50%",display:"flex",flexDirection:"column",justifyContent:"center",padding:"60px 52px",borderRight:`1px solid ${C.rule}44`,position:"relative",zIndex:10}}>
        <div className="fu" style={{fontFamily:C.M,fontSize:10,letterSpacing:".3em",color:C.lime,marginBottom:16,textShadow:`0 0 20px ${C.lime}66`}}>
          CHURN RISK ANALYTICS
        </div>
        <h1 className="fu" style={{fontFamily:C.D,fontSize:"clamp(64px,8.5vw,112px)",color:C.white,lineHeight:.88,letterSpacing:".02em",marginBottom:22,animationDelay:"80ms"}}>
          KNOW<br/>
          WHO'S<br/>
          <span style={{display:"inline-block",animation:"glitch 6s infinite",color:C.lime,textShadow:`0 0 30px ${C.lime}66`}}>LEAVING</span>
        </h1>
        <div className="fu" style={{fontFamily:C.D,fontSize:28,color:C.body,marginBottom:28,animationDelay:"160ms",letterSpacing:".04em"}}>
          <Typewriter words={["PREDICT CHURN","SAVE CUSTOMERS","DRIVE RETENTION","MAXIMISE LTV"]} speed={90}/>
        </div>
        <p className="fu" style={{fontFamily:C.S,fontSize:14,color:C.muted,maxWidth:340,lineHeight:1.8,marginBottom:36,animationDelay:"220ms"}}>
          Upload any customer CSV · map columns · get instant risk intelligence.
        </p>
        <div className="fu" style={{display:"flex",gap:8,flexWrap:"wrap",animationDelay:"300ms"}}>
          {["CSV Upload","Column Mapping","Risk Scoring","Visual Analytics"].map((f,i)=>(
            <div key={i} style={{fontFamily:C.M,fontSize:9,color:C.dim,border:`1px solid ${C.rule}`,padding:"5px 10px",letterSpacing:".1em",animation:"borderGlow 3s infinite",animationDelay:`${i*500}ms`}}>
              {f}
            </div>
          ))}
        </div>
      </div>

      {/* right drop zone */}
      <div style={{flex:1,display:"flex",flexDirection:"column",justifyContent:"center",padding:"60px 44px",position:"relative",zIndex:10}}>
        <div
          onDragOver={e=>{e.preventDefault();setDrag(true);}}
          onDragLeave={()=>setDrag(false)}
          onDrop={e=>{e.preventDefault();setDrag(false);handle(e.dataTransfer.files[0]);}}
          onClick={e=>{if(!scanning){addRipple(e);ref.current.click();}}
          }
          style={{
            border:`2px ${drag?"solid":"dashed"} ${drag?C.lime:C.dim}`,
            padding:"56px 36px",textAlign:"center",cursor:"none",
            background:drag?`${C.lime}0d`:C.paper,
            transition:"all .3s cubic-bezier(.22,1,.36,1)",
            boxShadow:drag?`0 0 80px ${C.lime}22,inset 0 0 60px ${C.lime}09`:
              "0 0 0 1px transparent",
            transform:drag?"scale(1.025)":"scale(1)",
            position:"relative",overflow:"hidden",
          }}>
          <input ref={ref} type="file" accept=".csv" style={{display:"none"}} onChange={e=>handle(e.target.files[0])}/>
          {rippleEls}
          <div style={{fontFamily:C.D,fontSize:64,color:drag?C.lime:C.white,marginBottom:10,letterSpacing:".02em",
            transition:"color .2s",animation:drag?"pulse .8s infinite":"float 4s ease-in-out infinite",
            textShadow:drag?`0 0 40px ${C.lime}88`:"none"}}>
            {drag?"DROP IT!":"UPLOAD"}
          </div>
          <div style={{fontFamily:C.M,fontSize:11,color:C.muted,marginBottom:28,letterSpacing:".06em"}}>
            {drag?"RELEASE TO SCAN":"DRAG CSV HERE OR CLICK TO BROWSE"}
          </div>
          <button className="glow-btn" style={{fontSize:13,background:C.lime,color:C.ink,padding:"14px 40px",letterSpacing:".14em",boxShadow:`0 4px 30px ${C.lime}44`}}
            onClick={e=>{e.stopPropagation();ref.current.click();}}>
            SELECT FILE
          </button>
        </div>

        <div className="fu" style={{marginTop:20,padding:"14px 18px",border:`1px solid ${C.rule}`,background:`${C.paper}88`,backdropFilter:"blur(8px)",animationDelay:"350ms"}}>
          <div style={{fontFamily:C.M,fontSize:9,letterSpacing:".18em",color:C.muted,marginBottom:8,fontWeight:700}}>RECOGNISED COLUMNS</div>
          <div style={{display:"flex",flexWrap:"wrap",gap:5}}>
            {["CustomerID","ChurnProbability","RiskTier","Tenure","Engagement","RetentionValue","LTV","CLV","MRR"].map((col,i)=>(
              <span key={col} style={{fontFamily:C.M,fontSize:9,color:C.dim,border:`1px solid ${C.rule}`,padding:"3px 7px",letterSpacing:".04em",
                transition:"all .2s",animationDelay:`${i*60}ms`}}
                onMouseEnter={e=>{e.target.style.borderColor=C.lime;e.target.style.color=C.lime;}}
                onMouseLeave={e=>{e.target.style.borderColor=C.rule;e.target.style.color=C.dim;}}>
                {col}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── MAP STEP ────────────────────────────────────────────────────────────────── */
function MapStep({headers,initialMapping,fileName,onConfirm}){
  const [mapping,setMapping]=useState(()=>{const m={};FEATURES.forEach(f=>{m[f.key]=new Set(initialMapping[f.key]||[]);});return m;});
  const [nothing,setNothing]=useState(()=>new Set());
  const toggle=(fk,col)=>{
    setMapping(p=>{const n={...p};const s=new Set(n[fk]);s.has(col)?s.delete(col):s.add(col);n[fk]=s;return n;});
    setNothing(p=>{const n=new Set(p);n.delete(fk);return n;});
  };
  const toggleNone=fk=>{
    setNothing(p=>{
      const n=new Set(p);
      if(n.has(fk))n.delete(fk);
      else{n.add(fk);setMapping(pm=>({...pm,[fk]:new Set()}));}
      return n;
    });
  };
  const ok=FEATURES.filter(f=>f.required).every(f=>mapping[f.key].size>0||nothing.has(f.key));
  const total=Object.values(mapping).reduce((a,s)=>a+s.size,0);

  return (
    <div style={{maxWidth:920,margin:"0 auto",padding:"44px 24px 60px",position:"relative"}}>
      <AuroraBg intensity={.3}/>
      <div style={{position:"relative",zIndex:1}}>
        <div className="fu" style={{display:"flex",alignItems:"flex-end",justifyContent:"space-between",marginBottom:36,paddingBottom:20,borderBottom:`1px solid ${C.rule}`}}>
          <div>
            <div style={{fontFamily:C.M,fontSize:9,letterSpacing:".22em",color:C.lime,marginBottom:8}}>STEP 02 / 03</div>
            <h2 style={{fontFamily:C.D,fontSize:56,color:C.white,lineHeight:1}}>MAP COLUMNS</h2>
            <div style={{fontFamily:C.M,fontSize:11,color:C.muted,marginTop:8}}>
              <span style={{color:C.blue}}>{fileName}</span> — {headers.length} cols detected
            </div>
          </div>
          <div style={{textAlign:"right"}}>
            <div style={{fontFamily:C.D,fontSize:40,color:C.lime,lineHeight:1,textShadow:total>0?`0 0 20px ${C.lime}66`:"none",transition:"text-shadow .4s",animation:total>0?"pulse 2.5s infinite":"none"}}>{total}</div>
            <div style={{fontFamily:C.M,fontSize:9,color:C.muted,letterSpacing:".1em"}}>COLS MAPPED</div>
          </div>
        </div>

        <div style={{display:"flex",flexDirection:"column",gap:0}}>
          {FEATURES.map((feat,fi)=>{
            const checked=mapping[feat.key],isNone=nothing.has(feat.key),cols=relevantCols(headers,feat),done=checked.size>0||isNone;
            return (
              <div key={feat.key} className="fu" style={{
                animationDelay:`${fi*90}ms`,
                borderTop:`1px solid ${C.rule}`,padding:"22px 0",
                borderLeft:done?`3px solid ${feat.color}`:`3px solid transparent`,
                paddingLeft:done?20:0,
                transition:"border-color .4s cubic-bezier(.22,1,.36,1), padding-left .4s cubic-bezier(.22,1,.36,1), background .4s",
                background:done?`${feat.color}05`:"transparent",
              }}>
                <div style={{display:"grid",gridTemplateColumns:"200px 1fr",gap:24,alignItems:"flex-start"}}>
                  <div>
                    <div style={{display:"flex",alignItems:"center",gap:7,marginBottom:5}}>
                      <span style={{fontFamily:C.M,fontSize:9,letterSpacing:".1em",background:feat.color,color:C.ink,padding:"2px 7px",fontWeight:700}}>{feat.tag}</span>
                      {feat.required&&<span style={{fontFamily:C.M,fontSize:9,color:C.red}}>REQ</span>}
                      {done&&<span style={{fontFamily:C.M,fontSize:9,color:feat.color,animation:"pulse 2s infinite",textShadow:`0 0 10px ${feat.color}`}}>✓ DONE</span>}
                    </div>
                    <div style={{fontFamily:C.D,fontSize:22,color:done?feat.color:C.bright,letterSpacing:".04em",marginBottom:3,transition:"color .3s, text-shadow .3s",textShadow:done?`0 0 15px ${feat.color}44`:"none"}}>{feat.label}</div>
                    <div style={{fontFamily:C.M,fontSize:10,color:C.muted,lineHeight:1.5}}>{feat.desc}</div>
                    {checked.size>1&&<div style={{fontFamily:C.M,fontSize:9,color:feat.color,marginTop:5}}>{feat.hint}</div>}
                  </div>
                  <div style={{display:"flex",flexWrap:"wrap",gap:6,paddingTop:4}}>
                    <div className="pill" onClick={()=>toggleNone(feat.key)} style={{border:`1px solid ${isNone?"#666":C.rule}`,background:isNone?"#66666618":C.paper,opacity:checked.size>0?.35:1}}>
                      <div style={{width:10,height:10,border:`1.5px solid ${isNone?"#999":C.dim}`,background:isNone?"#999":"transparent",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,transition:"all .15s"}}>
                        {isNone&&<span style={{color:C.ink,fontSize:7,fontWeight:900}}>✓</span>}
                      </div>
                      <span style={{color:isNone?C.bright:C.muted,fontStyle:"italic"}}>nothing</span>
                    </div>
                    {cols.length===0&&!isNone&&<span style={{fontFamily:C.M,fontSize:10,color:C.muted,fontStyle:"italic",alignSelf:"center",paddingLeft:4}}>no matching columns found</span>}
                    {cols.map(col=>{
                      const ck=checked.has(col);
                      return (
                        <div key={col} className="pill" onClick={()=>!isNone&&toggle(feat.key,col)} style={{
                          border:`1px solid ${ck?feat.color:C.rule}`,
                          background:ck?`${feat.color}18`:C.paper,
                          opacity:isNone?.2:1,cursor:isNone?"not-allowed":"none",
                          boxShadow:ck?`0 0 14px ${feat.color}33`:"none",
                        }}>
                          <div style={{width:10,height:10,border:`1.5px solid ${ck?feat.color:C.dim}`,background:ck?feat.color:"transparent",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,transition:"all .15s"}}>
                            {ck&&<span style={{color:C.ink,fontSize:7,fontWeight:900}}>✓</span>}
                          </div>
                          <span style={{color:ck?feat.color:C.muted,transition:"color .15s"}}>{col}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div style={{marginTop:36,paddingTop:24,borderTop:`1px solid ${C.rule}`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <div style={{fontFamily:C.M,fontSize:11,color:ok?C.lime:C.red,textShadow:ok?`0 0 12px ${C.lime}66`:"none"}}>
            {ok?"✓ READY TO ANALYSE":`MISSING: ${FEATURES.filter(f=>f.required&&mapping[f.key].size===0&&!nothing.has(f.key)).map(f=>f.label).join(", ")}`}
          </div>
          <button className="glow-btn" onClick={()=>ok&&onConfirm(mapping,nothing)}
            style={{fontSize:13,background:ok?C.lime:C.rule,color:ok?C.ink:C.muted,padding:"14px 44px",letterSpacing:".14em",cursor:ok?"none":"not-allowed",boxShadow:ok?`0 4px 30px ${C.lime}44`:"none"}}>
            RUN ANALYSIS →
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── TABLE ───────────────────────────────────────────────────────────────────── */
function Table({rows,mapping}){
  const [search,setSearch]=useState("");
  const [filter,setFilter]=useState("ALL");
  const [page,setPage]=useState(0);
  const PER=10;
  const g=k=>mapping[k];
  const rc=g("churnRisk")||[];
  const samp=useMemo(()=>rows.slice(0,50),[rows]);
  const getScore=r=>{
    if(!rc.length)return null;
    const n=rc.filter(c=>!isCat(samp,c)),ca=rc.filter(c=>isCat(samp,c));
    const ns=[...n.map(c=>parseFloat(r[c])).filter(x=>!isNaN(x)),...ca.map(c=>riskNum(r[c])).filter(x=>x!==null)];
    return ns.length?ns.reduce((a,b)=>a+b)/ns.length:null;
  };
  const getTier=r=>{
    const ca=rc.filter(c=>isCat(samp,c));
    const cv=ca.length?(r[ca[0]]||"").toLowerCase():"";
    const s=getScore(r);
    if(cv.includes("high")||(!cv&&s!==null&&s>=.65))return"HIGH";
    if(cv.includes("med")||(!cv&&s!==null&&s>=.35))return"MEDIUM";
    return"SAFE";
  };
  const filtered=rows.filter(r=>{
    const ic=g("customerID");
    return(!search||(ic&&(r[ic]||"").toLowerCase().includes(search.toLowerCase())))&&(filter==="ALL"||getTier(r)===filter);
  });
  const pages=Math.ceil(filtered.length/PER);
  const visible=filtered.slice(page*PER,(page+1)*PER);
  const cols=[
    g("customerID")&&{label:"CUSTOMER ID",render:r=><span style={{fontFamily:C.M,fontSize:11,color:C.lime,textShadow:`0 0 10px ${C.lime}44`}}>{r[g("customerID")]||"—"}</span>},
    rc.length&&{label:"RISK",render:r=>{const t=getTier(r);const col=t==="HIGH"?C.red:t==="MEDIUM"?C.amber:C.lime;return <span style={{fontFamily:C.M,fontSize:10,color:col,background:`${col}12`,border:`1px solid ${col}33`,padding:"2px 8px",letterSpacing:".06em",boxShadow:`0 0 8px ${col}22`}}>{t}</span>;}},
    rc.length&&{label:"SCORE",render:r=>{const s=getScore(r);const col=s===null?C.muted:s>=.65?C.red:s>=.35?C.amber:C.lime;return <span style={{fontFamily:C.M,fontSize:11,color:col}}>{s===null?"—":(s*100).toFixed(1)+"%"}</span>;}},
    g("tenure")&&{label:"TENURE",render:r=>{const v=avgC(r,[g("tenure")]);return <span style={{fontFamily:C.M,fontSize:11}}>{v===null?"—":v.toFixed(1)}</span>;}},
    g("engagement")&&{label:"ENGAGEMENT",render:r=>{const v=sumC(r,[g("engagement")]);return <span style={{fontFamily:C.M,fontSize:11}}>{v===null?"—":v.toFixed(1)}</span>;}},
    g("retentionValue")&&{label:"RET. VALUE",render:r=>{const v=sumC(r,[g("retentionValue")]);return <span style={{fontFamily:C.M,fontSize:11,color:C.purple,textShadow:`0 0 8px ${C.purple}44`}}>{v===null?"—":"$"+v.toFixed(2)}</span>;}},
  ].filter(Boolean);
  const TH={fontFamily:C.M,fontSize:9,letterSpacing:".14em",color:C.muted,padding:"10px 16px",textAlign:"left",borderBottom:`1px solid ${C.rule}`,background:C.panel,fontWeight:700};
  const TD={padding:"12px 16px",borderBottom:`1px solid ${C.rule}15`,fontSize:13,color:C.body};
  return (
    <TiltCard color={C.lime} className="fu" style={{border:`1px solid ${C.rule}`,background:C.paper}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"16px 20px",borderBottom:`1px solid ${C.rule}`}}>
        <div style={{fontFamily:C.D,fontSize:22,color:C.white,letterSpacing:".06em"}}>CUSTOMER RECORDS</div>
        <div style={{display:"flex",gap:8}}>
          {g("customerID")&&(
            <input value={search} onChange={e=>{setSearch(e.target.value);setPage(0);}} placeholder="SEARCH ID..."
              style={{fontFamily:C.M,fontSize:11,background:C.card,border:`1px solid ${C.rule}`,color:C.bright,padding:"8px 14px",outline:"none",letterSpacing:".04em",width:180,transition:"border-color .2s, box-shadow .2s"}}
              onFocus={e=>{e.target.style.borderColor=C.lime;e.target.style.boxShadow=`0 0 14px ${C.lime}33`;}} onBlur={e=>{e.target.style.borderColor=C.rule;e.target.style.boxShadow="none";}}/>
          )}
          {rc.length&&(
            <select value={filter} onChange={e=>{setFilter(e.target.value);setPage(0);}}
              style={{fontFamily:C.M,fontSize:11,background:C.card,border:`1px solid ${C.rule}`,color:C.bright,padding:"8px 14px",outline:"none",cursor:"none",letterSpacing:".04em"}}>
              {["ALL","HIGH","MEDIUM","SAFE"].map(o=><option key={o}>{o}</option>)}
            </select>
          )}
        </div>
      </div>
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%",borderCollapse:"collapse"}}>
          <thead><tr>{cols.map((c,i)=><th key={i} style={TH}>{c.label}</th>)}</tr></thead>
          <tbody>
            {visible.length===0
              ?<tr><td colSpan={cols.length} style={{...TD,textAlign:"center",padding:44,color:C.muted,fontFamily:C.M,fontSize:11,letterSpacing:".1em"}}>NO RECORDS FOUND</td></tr>
              :visible.map((r,i)=>(
                <tr key={i} className="row-tr" style={{background:i%2===0?"transparent":C.panel+"55"}}>
                  {cols.map((c,j)=><td key={j} style={TD}>{c.render(r)}</td>)}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      {pages>1&&(
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"12px 20px",borderTop:`1px solid ${C.rule}`}}>
          <span style={{fontFamily:C.M,fontSize:10,color:C.muted,letterSpacing:".06em"}}>{filtered.length.toLocaleString()} RECORDS / PAGE {page+1} OF {pages}</span>
          <div style={{display:"flex",gap:4}}>
            <button onClick={()=>setPage(p=>Math.max(0,p-1))} disabled={page===0}
              style={{fontFamily:C.M,fontSize:11,padding:"5px 10px",background:"transparent",border:`1px solid ${C.rule}`,color:page===0?C.dim:C.body,cursor:page===0?"not-allowed":"none",transition:"all .15s"}}>‹</button>
            {[...Array(Math.min(pages,7))].map((_,i)=>{
              const t=Math.max(0,Math.min(pages-7,page-3))+i;
              return <button key={t} onClick={()=>setPage(t)} style={{fontFamily:C.M,fontSize:10,padding:"5px 10px",background:t===page?C.lime:"transparent",border:`1px solid ${t===page?C.lime:C.rule}`,color:t===page?C.ink:C.body,cursor:"none",transition:"all .15s",boxShadow:t===page?`0 0 12px ${C.lime}44`:"none"}}>{t+1}</button>;
            })}
            <button onClick={()=>setPage(p=>Math.min(pages-1,p+1))} disabled={page===pages-1}
              style={{fontFamily:C.M,fontSize:11,padding:"5px 10px",background:"transparent",border:`1px solid ${C.rule}`,color:page===pages-1?C.dim:C.body,cursor:page===pages-1?"not-allowed":"none",transition:"all .15s"}}>›</button>
          </div>
        </div>
      )}
    </TiltCard>
  );
}

/* ── ANALYTICS ───────────────────────────────────────────────────────────────── */
function Analytics({rows,mapping}){
  const stats=useMemo(()=>buildStats(rows,mapping),[rows]);
  const hasRisk=!!(mapping.churnRisk||[]).length;
  const hasTen=!!stats.tenureBuckets,hasEng=!!stats.engBuckets;
  const charts=[hasRisk,hasTen,hasEng].filter(Boolean).length;
  const cards=[
    {label:"TOTAL CUSTOMERS",value:stats.total,color:C.blue},
    hasRisk&&{label:"HIGH RISK",value:stats.high,color:C.red,sub:`${stats.total?((stats.high/stats.total)*100).toFixed(1):0}%`},
    hasRisk&&{label:"MEDIUM RISK",value:stats.medium,color:C.amber,sub:`${stats.total?((stats.medium/stats.total)*100).toFixed(1):0}%`},
    hasRisk&&{label:"SAFE",value:stats.safe,color:C.lime,sub:`${stats.total?((stats.safe/stats.total)*100).toFixed(1):0}%`},
    stats.avgChurn&&{label:"AVG CHURN",value:stats.avgChurn+"%",color:C.red},
    stats.retDisplay&&{label:"TOTAL RETENTION",value:stats.retDisplay,color:C.purple},
  ].filter(Boolean);
  return (
    <div style={{padding:"0 0 60px",position:"relative"}}>
      <div style={{display:"grid",gridTemplateColumns:`repeat(${Math.min(cards.length,3)},1fr)`,gap:1,marginBottom:1}}>
        {cards.map((c,i)=><StatCard key={i} {...c} index={i}/>)}
      </div>
      {charts>0&&(
        <div style={{display:"grid",gridTemplateColumns:charts>=3?"1fr 1fr 1fr":charts===2?"1fr 1fr":"1fr",gap:1,marginBottom:1}}>
          {hasRisk&&(
            <TiltCard color={C.red} className="fu" style={{background:C.card,padding:26,borderTop:`3px solid ${C.red}`}}>
              <div style={{fontFamily:C.M,fontSize:9,letterSpacing:".18em",color:C.muted,marginBottom:18}}>RISK DISTRIBUTION</div>
              <div style={{display:"flex",alignItems:"center",gap:20}}>
                <Donut high={stats.high} medium={stats.medium} safe={stats.safe} total={stats.total}/>
                <div style={{flex:1}}>
                  {[{l:"HIGH",c:C.red,n:stats.high},{l:"MEDIUM",c:C.amber,n:stats.medium},{l:"SAFE",c:C.lime,n:stats.safe}].map((x,i)=>(
                    <div key={i} style={{marginBottom:14}}>
                      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                        <span style={{fontFamily:C.M,fontSize:10,color:x.c,letterSpacing:".1em"}}>{x.l}</span>
                        <span style={{fontFamily:C.M,fontSize:10,color:x.c,textShadow:`0 0 8px ${x.c}`}}>{stats.total?((x.n/stats.total)*100).toFixed(1):0}%</span>
                      </div>
                      <div style={{height:4,background:C.rule,overflow:"hidden"}}>
                        <div style={{height:"100%",width:`${stats.total?(x.n/stats.total)*100:0}%`,background:x.c,transition:"width 1.3s cubic-bezier(.22,1,.36,1)",boxShadow:`0 0 10px ${x.c}77`}}/>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </TiltCard>
          )}
          {hasTen&&(
            <TiltCard color={C.amber} className="fu" style={{background:C.card,padding:26,borderTop:`3px solid ${C.amber}`,animationDelay:"80ms"}}>
              <div style={{fontFamily:C.M,fontSize:9,letterSpacing:".18em",color:C.muted,marginBottom:18}}>TENURE DISTRIBUTION</div>
              <Bar items={stats.tenureBuckets} color={C.amber}/>
            </TiltCard>
          )}
          {hasEng&&(
            <TiltCard color={C.lime} className="fu" style={{background:C.card,padding:26,borderTop:`3px solid ${C.lime}`,animationDelay:"160ms"}}>
              <div style={{fontFamily:C.M,fontSize:9,letterSpacing:".18em",color:C.muted,marginBottom:18}}>ENGAGEMENT DISTRIBUTION</div>
              <Bar items={stats.engBuckets} color={C.lime}/>
            </TiltCard>
          )}
        </div>
      )}
      <div style={{marginTop:1}}><Table rows={rows} mapping={mapping}/></div>
    </div>
  );
}

/* ── APP ─────────────────────────────────────────────────────────────────────── */
export default function App(){
  const [step,setStep]=useState("upload");
  const [fileName,setFileName]=useState(null);
  const [headers,setHeaders]=useState([]);
  const [rows,setRows]=useState([]);
  const [mapping,setMapping]=useState({});
  const [nothing,setNothing]=useState(new Set());
  const handleFile=(name,text)=>{
    const{headers:h,rows:r}=parseCSV(text);
    setFileName(name);setHeaders(h);setRows(r);setMapping(autoDetect(h));setNothing(new Set());setStep("map");
  };
  const clear=()=>{setStep("upload");setFileName(null);setHeaders([]);setRows([]);setMapping({});setNothing(new Set());};
  const arrMap=useMemo(()=>Object.fromEntries(
    Object.entries(mapping).filter(([k])=>!nothing.has(k)).map(([k,v])=>[k,v instanceof Set?[...v]:v])
  ),[mapping,nothing]);
  const stats=useMemo(()=>step==="analytics"?buildStats(rows,arrMap):null,[step,rows,arrMap]);

  return (
    <div style={{minHeight:"100vh",background:C.ink,maxWidth:1280,margin:"0 auto"}}>
      <Cursor/>
      {step==="analytics"&&<Marquee stats={stats} fileName={fileName}/>}
      <nav style={{
        display:"flex",justifyContent:"space-between",alignItems:"center",
        padding:"0 32px",height:54,borderBottom:`1px solid ${C.rule}`,
        background:`${C.ink}dd`,backdropFilter:"blur(16px)",
        position:"sticky",top:step==="analytics"?26:0,zIndex:100,
      }}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{fontFamily:C.D,fontSize:22,color:C.white,letterSpacing:".08em",lineHeight:1}}>
            COMMU<span style={{color:C.lime,textShadow:`0 0 20px ${C.lime}88`}}>NO</span>
          </div>
          <div style={{width:1,height:18,background:C.rule}}/>
          <div style={{fontFamily:C.M,fontSize:9,color:C.muted,letterSpacing:".15em"}}>RISK ANALYTICS</div>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:0}}>
          {["upload","map","analytics"].map((s,i)=>{
            const done=["upload","map","analytics"].indexOf(step)>i,active=step===s;
            return (
              <div key={s} style={{display:"flex",alignItems:"center"}}>
                <div style={{display:"flex",alignItems:"center",gap:7,padding:"6px 14px",background:active?C.lime:"transparent",border:`1px solid ${active?C.lime:done?C.lime+"44":C.rule}`,borderRight:"none",transition:"all .35s",boxShadow:active?`0 0 20px ${C.lime}44`:"none"}}>
                  <span style={{fontFamily:C.M,fontSize:9,fontWeight:700,color:active?C.ink:done?C.lime:C.muted,letterSpacing:".08em"}}>{done?"✓":String(i+1).padStart(2,"0")}</span>
                  <span style={{fontFamily:C.M,fontSize:10,color:active?C.ink:done?C.body:C.muted,letterSpacing:".06em"}}>{s==="upload"?"UPLOAD":s==="map"?"MAP":"ANALYSE"}</span>
                </div>
                {i<2&&<div style={{width:10,height:1,background:C.rule}}/>}
              </div>
            );
          })}
          <div style={{border:`1px solid ${C.rule}`,width:1,height:30}}/>
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          {step==="analytics"&&<>
            <button className="nav-btn" onClick={()=>setStep("map")} style={{border:`1px solid ${C.rule}`,color:C.muted}}>REMAP</button>
            <button className="nav-btn" onClick={clear} style={{border:`1px solid ${C.rule}`,color:C.muted}}
              onMouseOver={e=>{e.currentTarget.style.color=C.red;e.currentTarget.style.borderColor=C.red+"55";}}
              onMouseOut={e=>{e.currentTarget.style.color=C.muted;e.currentTarget.style.borderColor=C.rule;}}>CLEAR</button>
            <label className="nav-btn" style={{border:`1px solid ${C.rule}`,color:C.muted,cursor:"none"}}>
              NEW CSV<input type="file" accept=".csv" style={{display:"none"}} onChange={e=>{const f=e.target.files[0];if(f){const r=new FileReader();r.onload=ev=>handleFile(f.name,ev.target.result);r.readAsText(f);}}}/>
            </label>
            <button className="glow-btn" onClick={()=>{const h=Object.keys(rows[0]).join(",");const b=rows.map(r=>Object.values(r).join(",")).join("\n");const blob=new Blob([h+"\n"+b],{type:"text/csv"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="churn_export.csv";a.click();}}
              style={{fontSize:10,background:C.lime,color:C.ink,padding:"7px 18px",letterSpacing:".12em",boxShadow:`0 2px 16px ${C.lime}44`}}>
              EXPORT ↓
            </button>
          </>}
        </div>
      </nav>
      <div style={{padding:step==="analytics"?"0 32px":"0"}}>
        {step==="upload"    && <Upload onFile={handleFile}/>}
        {step==="map"       && <MapStep headers={headers} initialMapping={mapping} fileName={fileName} onConfirm={(m,n)=>{setMapping(m);setNothing(n);setStep("analytics");}}/>}
        {step==="analytics" && <Analytics rows={rows} mapping={arrMap}/>}
      </div>
    </div>
  );
}