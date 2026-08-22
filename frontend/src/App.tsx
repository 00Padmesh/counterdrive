import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  Check,
  Cpu,
  FileImage,
  Gauge,
  Code2,
  Play,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Upload,
} from "lucide-react";

type ScenarioName = "hard_brake" | "maintain" | "accelerate" | "turn_left" | "turn_right";
type Prediction = {
  trajectory: number[][];
  collision_probability: number;
  risk_label: "low" | "elevated";
  risk_threshold: number;
};

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const labels: Record<ScenarioName, string> = {
  hard_brake: "Hard brake",
  maintain: "Maintain",
  accelerate: "Accelerate",
  turn_left: "Turn left",
  turn_right: "Turn right",
};
const colors: Record<ScenarioName, string> = {
  hard_brake: "#77b48f",
  maintain: "#e8a23a",
  accelerate: "#ef684f",
  turn_left: "#9386d8",
  turn_right: "#4f9fb0",
};

const demo: Record<ScenarioName, Prediction> = {
  hard_brake: { trajectory: [[-.01,.15],[-.01,.3],[0,.45],[0,.59],[0,.74],[0,.89]], collision_probability: .000008, risk_label: "low", risk_threshold: .0227 },
  maintain: { trajectory: [[-.02,.72],[-.01,1.39],[-.01,2.1],[0,2.83],[.01,3.53],[.01,4.28]], collision_probability: .9992, risk_label: "elevated", risk_threshold: .0227 },
  accelerate: { trajectory: [[-.01,1.3],[0,2.54],[0,3.79],[-.01,5.01],[.01,6.3],[.01,7.59]], collision_probability: .99998, risk_label: "elevated", risk_threshold: .0227 },
  turn_left: { trajectory: [[-.12,.63],[-.49,1.2],[-1.08,1.81],[-1.9,2.42],[-2.98,3.04],[-4.26,3.66]], collision_probability: .00019, risk_label: "low", risk_threshold: .0227 },
  turn_right: { trajectory: [[.09,.63],[.47,1.19],[1.08,1.8],[1.89,2.41],[2.96,3.03],[4.26,3.63]], collision_probability: .00018, risk_label: "low", risk_threshold: .0227 },
};

function TrajectoryPlot({ results, active }: { results: Record<ScenarioName, Prediction>; active: ScenarioName }) {
  const scale = 37;
  const originX = 240;
  const originY = 330;
  return (
    <svg className="trajectory-plot" viewBox="0 0 480 360" role="img" aria-label="Predicted counterfactual trajectories">
      <defs>
        <pattern id="grid" width="37" height="37" patternUnits="userSpaceOnUse">
          <path d="M 37 0 L 0 0 0 37" fill="none" stroke="#d5d9d2" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="480" height="360" fill="url(#grid)" />
      <path d={`M ${originX - 64} 360 L ${originX - 64} 0 M ${originX + 64} 360 L ${originX + 64} 0`} stroke="#c5cac1" strokeDasharray="8 8" />
      <circle cx={originX} cy={originY} r="7" fill="#111814" />
      {(Object.entries(results) as [ScenarioName, Prediction][]).map(([name, result]) => {
        const points = result.trajectory.map(([x,y]) => `${originX + x * scale},${originY - y * scale}`).join(" ");
        return <polyline key={name} points={points} fill="none" stroke={colors[name]} strokeWidth={name === active ? 6 : 2.5} strokeLinecap="round" strokeLinejoin="round" opacity={name === active ? 1 : .42} />;
      })}
      <text x="16" y="26" className="plot-label">FUTURE EGO PATH · METRES</text>
    </svg>
  );
}

function App() {
  const [active, setActive] = useState<ScenarioName>("maintain");
  const [results, setResults] = useState<Record<ScenarioName, Prediction>>(demo);
  const [frames, setFrames] = useState<string[]>([]);
  const [exportedPast, setExportedPast] = useState<number[][] | null>(null);
  const [speed, setSpeed] = useState(4);
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState("Showing the validated synthetic counterfactual run.");

  useEffect(() => {
    fetch(`${API_URL}/health`).then((response) => {
      if (!response.ok) throw new Error();
      setStatus("online");
    }).catch(() => setStatus("offline"));
  }, []);

  const generatedPast = useMemo(() => [-3, -2, -1, 0].map((step) => [0, step * speed * .5]), [speed]);
  const pastTrajectory = exportedPast ?? generatedPast;
  const selected = results[active];

  const onFrames = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []).slice(0, 4);
    const encoded = await Promise.all(files.map((file) => new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    })));
    setFrames(encoded);
    setExportedPast(null);
    setMessage(encoded.length === 4 ? "Four frames ready for a live rollout." : `Add ${4 - encoded.length} more frame${4 - encoded.length === 1 ? "" : "s"}.`);
  };

  const loadSyntheticExample = () => {
    const generated = Array.from({ length: 4 }, (_, step) => {
      const canvas = document.createElement("canvas");
      canvas.width = 160;
      canvas.height = 160;
      const context = canvas.getContext("2d")!;
      context.fillStyle = "#232923";
      context.fillRect(0, 0, 160, 160);
      context.fillStyle = "#3d443e";
      context.fillRect(0, 48, 160, 112);
      context.strokeStyle = "#ece9d9";
      context.lineWidth = 3;
      context.beginPath();
      context.moveTo(52, 160);
      context.lineTo(70, 48);
      context.moveTo(108, 160);
      context.lineTo(90, 48);
      context.stroke();
      context.fillStyle = "#ef684f";
      context.beginPath();
      context.arc(80, 86 + step * 6, 6 + step, 0, Math.PI * 2);
      context.fill();
      return canvas.toDataURL("image/jpeg", .9);
    });
    setFrames(generated);
    setExportedPast(null);
    setMessage("Bundled synthetic scene loaded. Start the API to run it live.");
  };

  const loadExportedScene = async () => {
    setMessage("Looking for the latest exported nuScenes scene…");
    try {
      const response = await fetch("/examples/nuscenes_scene/scene.json", { cache: "no-store" });
      if (!response.ok) throw new Error("No exported scene found. Run counterdrive-export-sample first.");
      const manifest = await response.json();
      const base = "/examples/nuscenes_scene/";
      const encoded = await Promise.all(manifest.frames.map(async (name: string) => {
        const imageResponse = await fetch(base + name, { cache: "no-store" });
        if (!imageResponse.ok) throw new Error(`Missing exported frame: ${name}`);
        const blob = await imageResponse.blob();
        return await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = () => reject(reader.error);
          reader.readAsDataURL(blob);
        });
      }));
      setFrames(encoded);
      setExportedPast(manifest.past_trajectory);
      setMessage(`${manifest.name} loaded with its measured ego history.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load the exported scene.");
    }
  };

  const runLive = async () => {
    if (status !== "online" || frames.length !== 4) return;
    setRunning(true);
    setMessage("Rolling five action futures through the model…");
    try {
      const response = await fetch(`${API_URL}/counterfactual`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames, past_trajectory: pastTrajectory }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "Inference failed");
      const payload = await response.json();
      setResults(payload.scenarios);
      setMessage("Live model result. Select an action to inspect its future.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Inference failed.");
    } finally {
      setRunning(false);
    }
  };

  const resetDemo = () => {
    setResults(demo);
    setFrames([]);
    setExportedPast(null);
    setMessage("Showing the validated synthetic counterfactual run.");
  };

  return (
    <main>
      <nav>
        <a className="wordmark" href="#top"><span>CD</span> CounterDrive</a>
        <div className="nav-links"><a href="#lab">Lab</a><a href="#evidence">Evidence</a><a href="#architecture">System</a></div>
        <div className={`api-state ${status}`}><span />API {status}</div>
      </nav>

      <header id="top" className="hero">
        <div className="eyebrow"><Sparkles size={14} /> Action-conditioned world model · v0.2</div>
        <h1>Ask the road<br /><em>what happens next.</em></h1>
        <p>CounterDrive predicts future ego motion and proximity risk under alternative steering, throttle, and braking actions.</p>
        <div className="hero-actions"><a className="primary" href="#lab">Open the lab <ArrowRight size={18} /></a><a className="secondary" href="https://github.com/00Padmesh/counterdrive" target="_blank" rel="noreferrer"><Code2 size={18} /> View source</a></div>
        <div className="hero-proof"><div><strong>314</strong><span>nuScenes windows</span></div><div><strong>5×</strong><span>counterfactual actions</span></div><div><strong>0.76</strong><span>risk AUROC</span></div></div>
      </header>

      <section id="lab" className="lab-section">
        <div className="section-heading"><div><span className="kicker">01 · Interactive lab</span><h2>One scene. Five futures.</h2></div><p>Use the validated demo or connect the local inference API and upload four consecutive front-camera frames.</p></div>
        <div className="lab-shell">
          <div className="control-rail">
            <div className="control-title"><Activity size={18} /><span>Action set</span></div>
            {(Object.keys(labels) as ScenarioName[]).map((name, index) => (
              <button key={name} className={active === name ? "scenario active" : "scenario"} onClick={() => setActive(name)}>
                <span className="scenario-index">0{index + 1}</span><span>{labels[name]}</span><i style={{ background: colors[name] }} />
              </button>
            ))}
            <div className="upload-block">
              <div className="example-actions">
                <button onClick={loadSyntheticExample}>Load synthetic</button>
                <button onClick={loadExportedScene}>Load exported</button>
              </div>
              <label className="upload-button"><Upload size={17} /><span>{frames.length ? `${frames.length}/4 frames` : "Choose 4 frames"}</span><input type="file" accept="image/*" multiple onChange={onFrames} /></label>
              {frames.length > 0 && <div className="frame-strip">{frames.map((frame, index) => <img key={index} src={frame} alt={`Input frame ${index + 1}`} />)}</div>}
              <label className="speed-control"><span>Observed speed <b>{speed} m/s</b></span><input type="range" min="1" max="15" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} /></label>
              <button className="run-button" onClick={runLive} disabled={status !== "online" || frames.length !== 4 || running}><Play size={16} fill="currentColor" />{running ? "Running…" : "Run live model"}</button>
              <button className="reset-button" onClick={resetDemo}><RotateCcw size={15} /> Reset demo</button>
            </div>
          </div>
          <div className="plot-panel"><TrajectoryPlot results={results} active={active} /><div className="plot-caption"><span className="status-dot" style={{ background: colors[active] }} /><span>{labels[active]} selected</span><small>{message}</small></div></div>
          <aside className="risk-panel">
            <div className="risk-top"><span>Predicted proximity risk</span><ShieldAlert size={20} /></div>
            <div className={`risk-orb ${selected.risk_label}`}><strong>{(selected.collision_probability * 100).toFixed(selected.collision_probability < .01 ? 3 : 1)}%</strong><span>{selected.risk_label}</span></div>
            <div className="threshold"><span>Decision threshold</span><strong>{(selected.risk_threshold * 100).toFixed(2)}%</strong></div>
            <div className="endpoint"><span>Final position</span><strong>{selected.trajectory.at(-1)?.[0].toFixed(2)}m lateral</strong><strong>{selected.trajectory.at(-1)?.[1].toFixed(2)}m forward</strong></div>
            <p><FileImage size={15} /> Research output—not driving advice.</p>
          </aside>
        </div>
      </section>

      <section id="evidence" className="evidence-section">
        <div className="section-heading light"><div><span className="kicker">02 · Evidence</span><h2>Measured, not hand-waved.</h2></div><p>The model is compared against meaningful baselines, with scene-disjoint nuScenes validation.</p></div>
        <div className="evidence-grid">
          <article className="big-result"><span>Synthetic counterfactual</span><strong>44×</strong><h3>lower final displacement error</h3><p>0.063m conditioned FDE versus 3.065m when the same network cannot see the proposed action.</p><div className="bar"><i style={{ width: "2%" }} /><em /></div></article>
          <article><Gauge /><span>Real trajectory</span><strong>3.039m</strong><p>FDE on nuScenes-mini, improving on the 3.123m constant-velocity baseline.</p></article>
          <article><ShieldAlert /><span>Risk ranking</span><strong>0.758</strong><p>AUROC with a separate risk-selected checkpoint; average precision reaches 0.674.</p></article>
          <article><Check /><span>Data integrity</span><strong>10 scenes</strong><p>314 windows split by scene, not adjacent frames. Proximity-risk rate: 19.4%.</p></article>
        </div>
      </section>

      <section id="architecture" className="architecture-section">
        <div className="section-heading"><div><span className="kicker">03 · System</span><h2>Small enough to train.<br />Serious enough to inspect.</h2></div></div>
        <div className="pipeline">
          <div><span>01</span><Cpu /><strong>See</strong><p>Frozen ResNet-18 turns four RGB frames into compact visual tokens.</p></div><ArrowRight />
          <div><span>02</span><Activity /><strong>Remember</strong><p>A temporal Transformer forms the current latent world state.</p></div><ArrowRight />
          <div><span>03</span><Gauge /><strong>Imagine</strong><p>Action embeddings condition six future latent rollouts.</p></div><ArrowRight />
          <div><span>04</span><ShieldAlert /><strong>Forecast</strong><p>Separate heads predict ego trajectory and calibrated risk.</p></div>
        </div>
      </section>

      <footer><div className="wordmark"><span>CD</span> CounterDrive</div><p>Student-scale driving world model · Python, PyTorch, FastAPI, React</p><a href="#top">Back to top ↑</a></footer>
    </main>
  );
}

export default App;
