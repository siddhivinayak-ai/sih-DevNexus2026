"""Generate the narrated SmartScan walkthrough video.

    1. start the app:      streamlit run app.py            (http://localhost:8501)
    2. run:                python tools/demo_video/make_demo_video.py
    -> demo/SmartScan_Demo.mp4

Pipeline: Windows text-to-speech (System.Speech) renders one WAV per segment ->
Playwright drives Chromium through the real app and records the screen, holding each
segment for the length of its narration -> ffmpeg places every clip at the time its
segment started and muxes audio + video into an H.264 MP4. Subtitles and a visible
cursor are overlaid in the page for first-time viewers.

Only session-level actions are performed (no config saves, no training, no model writes).
Requires (dev only): pip install playwright imageio-ffmpeg && python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "demo" / "_build"
OUT = ROOT / "demo" / "SmartScan_Demo.mp4"
W, H = 1600, 900

# --------------------------------------------------------------------------- narration
SEGMENTS: list[tuple[str, str]] = [
    ("intro",
     "Welcome to SmartScan, an adaptive RF spectrum scanning research platform built for Smart India Hackathon "
     "problem statement two six zero five five. In this video we walk through every screen, so that a first time "
     "user knows exactly what each part does and how to use it."),
    ("overview",
     "This is the Overview page. The problem is simple to state. A receiver has to watch a wide spectrum, but it can "
     "only look at one frequency band at a time. A fixed sweep, F1, F2, F3 and so on, ignores what it has already "
     "seen, so a short transmission on F6 can be over before the sweep gets there. SmartScan treats the choice of "
     "the next band as a sequential decision problem."),
    ("overview_arch",
     "The block diagram shows the closed loop. The RF environment holds the ground truth. The receiver observes one "
     "band and reports hit or no hit. That observation updates the state, the scheduler picks the next band, and the "
     "loop repeats. The dashed lines are important: ground truth only reaches the reward and evaluation engines. "
     "The scheduler never sees it."),
    ("workspace",
     "On the left is the navigation, grouped like an engineering tool, and below it the Workspace panel. The "
     "workspace always shows the active configuration: the scenario, number of bands, receiver probability of "
     "detection and false alarm, the seed splits, whether a PPO model is trained, and whether the Turing dataset "
     "is available."),
    ("scenario",
     "Scenario Builder is where you define the electromagnetic environment. Choose a preset and press Load Preset. "
     "Here we load the S I H example: a periodic emitter on F3, a short burst threat on F6, a frequency agile emitter "
     "hopping F2, F6, F4, F7, and a static emitter on F1."),
    ("scenario_tabs",
     "The tabs hold every parameter. Emitters is an editable table; any blank cell is drawn at random for each "
     "scenario seed. Receiver sets the detection model, probability of detection, false alarm rate, instantaneous "
     "bandwidth, dwell time and retune latency. Reward and seeds holds the reward weights and the separate "
     "training, validation and test seed ranges."),
    ("scenario_preview",
     "At the bottom, the ground truth preview shows the environment for any seed. Grey cells mean an emitter is "
     "transmitting in that band at that time. This view is for us, the evaluators. The scheduler never gets it. "
     "For the rest of the demo we reload the default mixed scenario, which is the one PPO was trained on."),
    ("live_intro",
     "Live Simulation is the heart of the demo. Choose a scheduler and a scenario seed. The buttons work like a "
     "simulation debugger. Start runs continuously, Pause stops, Step executes exactly one decision, Run 10 and "
     "Run 100 jump ahead, and Reset starts the episode again. We begin with the open loop sweep."),
    ("live_openloop",
     "Each Step performs one full decision cycle. The strip at the top shows the time, the band just scanned, the "
     "next action, the receiver observation, the outcome, the reward and how many transmissions have been "
     "intercepted. On the spectrum map, grey is the hidden ground truth and each marker is a receiver dwell. A green "
     "circle is a detection, a red cross is a false alarm, and an orange diamond is a missed detection."),
    ("live_openloop_run",
     "Now Run 100. The open loop scanner produces a perfect staircase, F1 to F8 and back, no matter what it "
     "observes. Below, the scan trajectory, the belief map, the event timeline and the reward curve update with "
     "every decision."),
    ("live_pomdp",
     "Next the P O M D P scheduler, the original belief state method of this project. Open the Action tab of the "
     "decision cycle panel. The bars are the belief that each band is active, and the lines are the score "
     "components. The scheduler picks the band with the highest score, so it adapts to what it has observed."),
    ("live_pomdp_run",
     "After a hundred decisions you can see it concentrating on bands where it has already found activity. It "
     "adapts, but it also tends to stay on confirmed emitters. The comparison page will measure that effect."),
    ("live_ppo",
     "Now the reinforcement learning scheduler, PPO. The Action tab now shows the policy, pi of a given s: the "
     "probability the trained network assigns to every band. The Observe and reward tab breaks the reward into its "
     "terms: new intercepts, detections, false alarms, latency and empty revisits. The last tab shows the next "
     "decision before it is executed."),
    ("live_ppo_run",
     "Press Start to watch it run continuously, and Pause at any moment. At the bottom, the command window logs every "
     "decision, similar to a MATLAB console: time, action, observation, outcome and reward."),
    ("training",
     "The R L Training page trains PPO with Stable Baselines 3. All hyper parameters are editable: timesteps, "
     "learning rate, gamma, G A E lambda, batch size, entropy and network size. Start Training runs in the "
     "background; Stop ends cleanly and still saves. Training uses only the training seeds. The best model is chosen "
     "on validation seeds, and test seeds are never touched."),
    ("training_monitor",
     "The monitor shows only real logged values: episode reward, the validation curve on unseen seeds, and PPO's "
     "own update statistics such as value loss, entropy and K L divergence. The models panel lets you load, "
     "snapshot, or quickly evaluate a trained model."),
    ("evaluation",
     "Policy Evaluation runs one scheduler over many seeds. Choose the scheduler, the split, test by default, and "
     "the number of episodes, then press Evaluate. You get probability of detection, false alarm rate, interception "
     "rate, intercept times, reward and coverage, all as mean plus or minus standard deviation over episodes."),
    ("evaluation_trace",
     "Below, the raw event trace shows every single decision of a chosen episode together with its spectrum map, "
     "so any number in the table can be traced back to what actually happened."),
    ("comparison",
     "Algorithm Comparison is the fair experiment. Open loop, random, P O M D P and PPO all run on exactly the same "
     "scenarios, the same receiver and even the same detector noise. Only the scan decisions differ. Press Run "
     "Comparison."),
    ("comparison_results",
     "The table reports every metric as mean plus or minus standard deviation with the number of episodes. There is "
     "no winner badge, only measured results. The paired differences compare each scheduler with open loop on the "
     "same seeds, with ninety five percent confidence intervals. Below are per metric panels, distributions and "
     "band revisit frequency."),
    ("analytics",
     "Analytics reopens saved experiments, such as the hundred seed test run stored with the project, keeps the run "
     "history of this session, and has a generalisation check that compares training seeds against unseen test seeds."),
    ("dataset",
     "The Dataset Explorer integrates the Alan Turing Institute synthetic radar dataset. When files are present it "
     "shows pulse descriptor words: time of arrival, centre frequency, pulse width, angle of arrival, amplitude and "
     "emitter labels, reading only bounded samples. Without the dataset, as here, the app says so clearly and keeps "
     "working on the synthetic environment."),
    ("methodology",
     "Methodology contains the full documentation: problem statement, architecture, R L formulation, the exact "
     "reward equation, evaluation method, user guide, limitations and the demo script."),
    ("outro",
     "That is SmartScan. The receiver can see only part of the spectrum, so every scan decision changes what it "
     "learns next, and SmartScan learns where to look from its own observations. This is a research simulation "
     "prototype, and every number you saw was produced live by the simulation. Thank you."),
]


# --------------------------------------------------------------------------- TTS
def synthesize(voice: str | None, rate: int) -> dict[str, float]:
    WORK.mkdir(parents=True, exist_ok=True)
    jobs = [{"text": t, "path": str(WORK / f"{k}.wav")} for k, t in SEGMENTS]
    (WORK / "tts.json").write_text(json.dumps(jobs), encoding="utf-8")
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
{"$s.SelectVoice('" + voice + "')" if voice else ""}
$s.Rate = {rate}
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$jobs = Get-Content -Raw -Encoding UTF8 '{WORK / "tts.json"}' | ConvertFrom-Json
foreach ($j in $jobs) {{ $s.SetOutputToWaveFile($j.path, $fmt); $s.Speak($j.text) }}
$s.SetOutputToNull()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    durations = {}
    for k, _ in SEGMENTS:
        with wave.open(str(WORK / f"{k}.wav")) as w:
            durations[k] = w.getnframes() / w.getframerate()
    return durations


# --------------------------------------------------------------------------- browser helpers
OVERLAY_JS = """
(() => {
  if (window.__ssOverlay) return; window.__ssOverlay = true;
  const add = () => {
    if (!document.body) return setTimeout(add, 50);
    const cur = document.createElement('div'); cur.id = 'ss-cursor';
    cur.style.cssText = 'position:fixed;z-index:2147483647;width:18px;height:18px;border-radius:50%;' +
      'background:rgba(217,48,37,.55);border:2px solid #fff;box-shadow:0 0 0 1px #d93025;pointer-events:none;' +
      'transform:translate(-50%,-50%);left:-40px;top:-40px;transition:width .12s,height .12s';
    const sub = document.createElement('div'); sub.id = 'ss-sub';
    sub.style.cssText = 'position:fixed;z-index:2147483646;left:50%;bottom:18px;transform:translateX(-50%);' +
      'max-width:1150px;background:rgba(20,28,40,.86);color:#fff;font:15px/1.4 "Segoe UI",Arial;' +
      'padding:7px 14px;border-radius:3px;pointer-events:none;text-align:center;display:none';
    const title = document.createElement('div'); title.id = 'ss-title';
    title.style.cssText = 'position:fixed;inset:0;z-index:2147483645;background:#fff;display:none;' +
      'flex-direction:column;align-items:center;justify-content:center;font-family:"Segoe UI",Arial;color:#1f3a5f';
    title.innerHTML = '<div style="border-top:4px solid #1f3a5f;padding-top:18px;text-align:center">' +
      '<div style="font-size:54px;font-weight:700;letter-spacing:.08em">SMARTSCAN</div>' +
      '<div style="font-size:20px;color:#4d4d4d;margin-top:8px">Adaptive RF Spectrum Scanning &amp; Interception</div>' +
      '<div style="font-size:16px;color:#6b6b6b;margin-top:26px">Solution walkthrough · SIH 2026 · Problem Statement 26055</div>' +
      '<div style="font-size:13px;color:#8a8a8a;margin-top:6px">Research simulation prototype</div></div>';
    document.body.append(title, sub, cur);
    document.addEventListener('mousemove', e => { cur.style.left = e.clientX + 'px'; cur.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', () => { cur.style.width = cur.style.height = '30px'; }, true);
    document.addEventListener('mouseup', () => { cur.style.width = cur.style.height = '18px'; }, true);
  };
  add();
})();
"""


class Demo:
    def __init__(self, page, url: str):
        self.page = page
        self.url = url

    # overlays -------------------------------------------------------------
    def subtitle(self, text: str | None) -> None:
        self.page.evaluate(
            "t => { const s = document.getElementById('ss-sub'); if (!s) return;"
            " s.style.display = t ? 'block' : 'none'; s.textContent = t || ''; }", text)

    def title(self, show: bool) -> None:
        self.page.evaluate("v => { const t = document.getElementById('ss-title'); if (t) t.style.display = v ? 'flex' : 'none'; }", show)

    # streamlit helpers ----------------------------------------------------
    def settle(self, extra: float = 0.4) -> None:
        self.page.wait_for_timeout(500)
        try:
            self.page.wait_for_function(
                "() => !document.querySelector('[data-testid=\"stStatusWidget\"] [data-testid=\"stStatusWidgetRunningIcon\"]')"
                " && !document.querySelector('[data-testid=\"stSkeleton\"]')", timeout=30000)
        except Exception:
            pass
        self.page.wait_for_timeout(int(extra * 1000))

    def move_to(self, locator) -> None:
        box = locator.bounding_box()
        if box:
            self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=18)

    def click(self, locator, settle: float = 0.6) -> None:
        locator.scroll_into_view_if_needed()
        self.move_to(locator)
        self.page.wait_for_timeout(250)
        locator.click()
        self.settle(settle)

    def nav(self, name: str) -> None:
        link = self.page.locator('[data-testid="stSidebarNav"] a').filter(has_text=name).first
        self.click(link, 1.0)
        self.page.mouse.move(W * 0.6, H * 0.4, steps=10)

    def button(self, name: str, settle: float = 0.6) -> None:
        self.click(self.page.get_by_role("button", name=name, exact=True).first, settle)

    def tab(self, name: str) -> None:
        self.click(self.page.get_by_role("tab", name=name).first, 0.3)

    def select(self, label: str, option: str) -> None:
        box = self.page.locator('[data-testid="stSelectbox"]').filter(has_text=label).first
        self.click(box, 0.2)
        self.click(self.page.get_by_role("option", name=option, exact=True).first, 0.8)

    def number(self, label: str, value: int) -> None:
        inp = self.page.get_by_label(label, exact=True).first
        inp.scroll_into_view_if_needed()
        self.move_to(inp)
        inp.fill(str(value))
        inp.press("Enter")
        self.settle(0.3)

    def scroll(self, dy: int, steps: int = 8, pause: float = 0.12) -> None:
        self.page.mouse.move(W * 0.62, H * 0.5)
        for _ in range(steps):
            self.page.mouse.wheel(0, dy / steps)
            self.page.wait_for_timeout(int(pause * 1000))

    def top(self) -> None:
        self.scroll(-8000, 4, 0.05)

    def wait(self, s: float) -> None:
        self.page.wait_for_timeout(int(s * 1000))


# --------------------------------------------------------------------------- the walkthrough
def actions(d: Demo) -> dict:
    return {
        "intro": lambda: (d.title(True), d.wait(8), d.title(False)),
        "overview": lambda: (d.wait(3), d.scroll(250), d.wait(6), d.scroll(-250)),
        "overview_arch": lambda: (d.move_to(d.page.locator('[data-testid="stGraphVizChart"]').first), d.wait(2),
                                  d.scroll(300), d.wait(6), d.top()),
        "workspace": lambda: (d.move_to(d.page.locator('[data-testid="stSidebarNav"]').first), d.wait(4),
                              d.move_to(d.page.locator(".ss-ws").first), d.wait(3)),
        "scenario": lambda: (d.nav("Scenario Builder"), d.select("Preset", "sih_example.yaml"),
                             d.button("LOAD PRESET", 1.2), d.wait(1)),
        "scenario_tabs": lambda: (d.tab("Emitters"), d.wait(5), d.tab("Receiver"), d.wait(4),
                                  d.tab("Reward & seeds"), d.wait(4), d.tab("Scenario")),
        "scenario_preview": lambda: (d.scroll(700, 10), d.wait(7), d.top(), d.select("Preset", "default.yaml"),
                                     d.button("LOAD PRESET", 1.2)),
        "live_intro": lambda: (d.nav("Live Simulation"), d.select("Scheduler", "Open Loop"), d.wait(1),
                               *[d.move_to(d.page.get_by_role("button", name=b, exact=True).first) or d.wait(0.7)
                                 for b in ("START", "PAUSE", "STEP", "RUN 10", "RUN 100", "RESET")]),
        "live_openloop": lambda: ([d.button("STEP", 0.5) for _ in range(4)], d.wait(2),
                                  d.move_to(d.page.locator(".ss-kpis").first), d.wait(3),
                                  d.move_to(d.page.locator('[data-testid="stPlotlyChart"]').first), d.wait(3)),
        "live_openloop_run": lambda: (d.button("RUN 100", 1.0), d.wait(3), d.scroll(700, 10), d.wait(5),
                                      d.scroll(500, 8), d.wait(3), d.top()),
        "live_pomdp": lambda: (d.select("Scheduler", "POMDP"), [d.button("STEP", 0.5) for _ in range(6)],
                               d.tab("2 Action"), d.wait(4)),
        "live_pomdp_run": lambda: (d.button("RUN 100", 1.0), d.tab("2 Action"), d.wait(5)),
        "live_ppo": lambda: (d.select("Scheduler", "PPO"), [d.button("STEP", 0.5) for _ in range(3)],
                             d.tab("2 Action"), d.wait(4), d.tab("3–5 Observe & reward"), d.wait(5),
                             d.tab("6–7 Next decision"), d.wait(3)),
        "live_ppo_run": lambda: (d.button("START", 0.5), d.wait(8), d.button("PAUSE", 0.8),
                                 d.scroll(1600, 12), d.wait(4), d.top()),
        "training": lambda: (d.nav("RL Training"), d.wait(4),
                             d.move_to(d.page.get_by_role("button", name="START TRAINING").first), d.wait(4)),
        "training_monitor": lambda: (d.scroll(600, 10), d.wait(6), d.scroll(500, 8), d.wait(4), d.top()),
        "evaluation": lambda: (d.nav("Policy Evaluation"), d.number("Episodes", 10), d.button("EVALUATE", 1.5),
                               d.scroll(350, 8), d.wait(5)),
        "evaluation_trace": lambda: (d.scroll(700, 10), d.wait(6), d.top()),
        "comparison": lambda: (d.nav("Algorithm Comparison"), d.number("Episodes", 20),
                               d.button("RUN COMPARISON", 2.0)),
        "comparison_results": lambda: (d.scroll(350, 8), d.wait(6), d.scroll(700, 10), d.wait(5),
                                       d.scroll(900, 10), d.wait(4), d.scroll(700, 10), d.wait(3), d.top()),
        "analytics": lambda: (d.nav("Analytics"), d.wait(3), d.scroll(600, 10), d.wait(4), d.top(),
                              d.tab("Training vs unseen seeds"), d.wait(2)),
        "dataset": lambda: (d.nav("Dataset Explorer"), d.wait(6)),
        "methodology": lambda: (d.nav("Methodology"), d.wait(2),
                                d.click(d.page.get_by_text("05 rl formulation").first, 0.8), d.wait(3), d.scroll(500, 8)),
        "outro": lambda: (d.nav("Overview"), d.wait(4), d.title(True), d.wait(3)),
    }


def record(url: str, durations: dict[str, float], headless: bool = True) -> tuple[Path, list[tuple[str, float]]]:
    from playwright.sync_api import sync_playwright

    vid_dir = WORK / "video"
    vid_dir.mkdir(parents=True, exist_ok=True)
    for f in vid_dir.glob("*.webm"):
        f.unlink()
    starts: list[tuple[str, float]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(vid_dir),
                                  record_video_size={"width": W, "height": H})
        ctx.add_init_script(OVERLAY_JS)
        page = ctx.new_page()
        t0 = time.monotonic()
        page.goto(url, wait_until="networkidle")
        d = Demo(page, url)
        d.settle(2.0)
        acts = actions(d)
        for key, text in SEGMENTS:
            start = time.monotonic() - t0
            starts.append((key, start))
            d.subtitle(text if len(text) < 260 else None)
            if len(text) >= 260:  # long narration: show it sentence by sentence
                _captioned(d, text, durations[key], acts[key])
            else:
                acts[key]()
            remaining = durations[key] + 0.7 - (time.monotonic() - t0 - start)
            if remaining > 0:
                d.wait(remaining)
            print(f"  {key:22s} start {start:6.1f}s  narration {durations[key]:5.1f}s", flush=True)
        d.subtitle(None)
        d.wait(1.5)
        video_path = Path(page.video.path())
        ctx.close()
        browser.close()
    return video_path, starts


def _captioned(d: Demo, text: str, duration: float, action) -> None:
    """Run the action while cycling subtitle sentences in a timer inside the page."""
    import re

    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    total = sum(len(s) for s in parts)
    schedule, t = [], 0.0
    for s in parts:
        schedule.append([round(t * 1000), s])
        t += duration * len(s) / total
    d.page.evaluate(
        """sched => { const s = document.getElementById('ss-sub'); if (!s) return;
             s.style.display = 'block';
             (window.__ssTimers || []).forEach(clearTimeout);
             window.__ssTimers = sched.map(([ms, txt]) => setTimeout(() => { s.textContent = txt; }, ms)); }""",
        schedule)
    action()


def mux(video: Path, starts: list[tuple[str, float]]) -> None:
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    args = [ff, "-y", "-i", str(video)]
    filters, labels = [], []
    for i, (key, start) in enumerate(starts, start=1):
        args += ["-i", str(WORK / f"{key}.wav")]
        ms = int(start * 1000)
        filters.append(f"[{i}:a]adelay={ms}|{ms},aresample=44100[a{i}]")
        labels.append(f"[a{i}]")
    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:dropout_transition=0,volume=1.6[aout]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    args += ["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-r", "25",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(OUT)]
    subprocess.run(args, check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--voice", default=None, help="Windows voice name, e.g. 'Microsoft Zira Desktop'")
    ap.add_argument("--rate", type=int, default=0, help="speech rate -10..10")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    print("1/3 synthesising narration ...")
    durations = synthesize(args.voice, args.rate)
    print(f"    {len(durations)} clips, {sum(durations.values()) / 60:.1f} min of speech")
    print("2/3 recording the app ...")
    video, starts = record(args.url, durations, headless=not args.headed)
    (WORK / "timeline.json").write_text(json.dumps(starts, indent=1), encoding="utf-8")
    print("3/3 muxing audio + video ...")
    mux(video, starts)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
