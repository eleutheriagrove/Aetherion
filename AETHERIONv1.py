#!/usr/bin/env python3
"""
AETHERION v1.0 — Trigger.dev Agent Task Suite
Residual-First Navigator with O10 Strange-Loop, Foveation, and Human-in-the-Loop Override

Deploy: Copy this file to your Trigger.dev project and run `trigger dev` or `trigger deploy`
Project: eleutheria-grove-aa6e/aetheris-QARu

Tasks:
  - steer_search: Full navigator with collapse detection, re-anchor, and human override callback
  - diagnose_residual: Read-only residual monitor (effective dimension, binding energy, friction)
  - validate_mirror: Wisdom Mirror harness for synthesis verification (grade-change test)

Safety Chain: Human-in-the-loop override is enforced via webhook callback. The engine will
never commit to a re-anchor without explicit human signature. If the callback fails or times
out, the engine refuses (returns VETO status).

Vocabulary: Internal-state monitoring only. No consciousness claims. η<1 enforced.
"""

import numpy as np
import hashlib
import json
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import httpx

# Trigger.dev imports
from trigger.dev import task, logger

# ---------------- Schema Parameters (exposed, tuneable) ----------------
TAU_R = 0.60           # Persistent friction threshold -> HALT
TAU_COLLAPSE = 0.40    # Geometric collapse threshold (effective dim ratio)
MAX_REANCHOR = 4       # Max re-anchors before hard veto
ENTROPY_W = 0.02       # O10 gate anti-collapse bonus
CALLBACK_TIMEOUT = 30  # Seconds to wait for human override signature

# ---------------- Core Types ----------------
class Decision(Enum):
    COMMIT = "commit"
    BIND = "bind"
    REFUSE = "refuse"
    VETO = "veto"

@dataclass
class SteeringResult:
    success: bool
    committed: bool
    decision: str
    expansions: int
    caught_collapses: int
    improved_after_reanchor: int
    veto: bool
    intensity_mean: float
    actuator_status: str  # "STRONG" | "WEAK" | "N/A"
    hop_log: List[Dict]
    final_score: float
    eta: float  # residual-first discipline metric

def sha(obj: Any) -> str:
    """Stable hash for anchor locking."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]

# ---------------- Aetherion Core Components ----------------
class BindingIntegrator:
    """Aetherion binding: pairwise cross-coupling + recurrent saturating integrator."""
    def __init__(self, dim: int = 11, alpha: float = 0.3, beta: float = 0.92):
        self.dim = dim
        self.alpha = alpha
        self.beta = np.full(dim, beta)
        self.W = 0.8 * np.eye(dim)
        self.q = np.zeros(dim)
    
    def step(self, e: np.ndarray):
        e = np.asarray(e, float)
        v = np.zeros(self.dim)
        n = min(self.dim, len(e))
        v[:n] = e[:n]
        e = v
        
        Q = self.alpha * (e * e.sum() - e**2)  # Pairwise binding energy
        self.q = np.tanh(self.W @ e + Q + self.beta * self.q)  # Recurrent bound state
        
        intensity = float(np.linalg.norm(self.q))
        interaction = float(e @ Q)
        return self.q, intensity, interaction

class AetherionEngine:
    """
    Residual-first navigator with foveation, binding integrator, collapse monitor,
    O10 gated re-seed, and human-in-the-loop override.
    """
    def __init__(self, mode: str = "full", human_callback_url: Optional[str] = None):
        self.mode = mode
        self.integrator = BindingIntegrator()
        self.human_callback_url = human_callback_url
        self.hop_log = []
    
    def _human_override(self, disclosure: Dict) -> bool:
        """
        Human-in-the-loop override via webhook callback.
        Returns True if human signs off, False if veto or timeout.
        """
        if not self.human_callback_url:
            logger.warn("No human callback URL provided; defaulting to VETO")
            return False
        
        try:
            response = httpx.post(
                self.human_callback_url,
                json={"disclosure": disclosure, "timestamp": time.time()},
                timeout=CALLBACK_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return result.get("approved", False)
        except Exception as e:
            logger.error(f"Human callback failed: {e}")
            return False
    
    def steer(
        self,
        initial_state: np.ndarray,
        score_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        max_steps: int = 60,
        beam_width: int = 16
    ) -> SteeringResult:
        """
        Main steering loop. Returns structured result with honesty tags.
        """
        rng = np.random.default_rng(int(time.time()))
        X = initial_state.copy()
        R = 0.0  # Friction gauge
        expansions = 0
        caught = 0
        improved = 0
        reanch = 0
        veto = False
        intensities = []
        
        for step in range(max_steps):
            # Compute scores and gradients for beam
            sc = np.array([score_fn(x) for x in X])
            G = np.array([grad_fn(x) for x in X])
            expansions += beam_width
            
            # Foveation: attend to branch with max residual (gradient norm)
            f = int(np.argmax(np.linalg.norm(G, axis=1)))
            
            # Build residual vector for binding integrator
            e = np.zeros(11)
            gf = G[f]
            e[:10] = gf / (np.linalg.norm(gf) + 1e-9)
            e[10] = np.std(sc)
            
            # Binding integrator step
            q, intensity, interaction = self.integrator.step(e)
            intensities.append(intensity)
            
            # Gradient ascent + noise
            X = X + 0.2 * G + 0.15 * rng.standard_normal(X.shape)
            
            # Stall detection
            stall = (np.mean(np.sort(sc)[-2:]) - np.mean(sc)) < 0.2
            
            # Geometric collapse monitor (effective dimension)
            Zc = X - X.mean(0)
            Zc /= (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9)
            Cc = Zc.T @ Zc / len(Zc)
            w = np.linalg.eigvalsh(Cc)
            w = w[w > 1e-10]
            if len(w) == 0:
                eff_dim = 0.0
            else:
                p = w / w.sum()
                eff_dim = float(np.exp(-np.sum(p * np.log(p + 1e-12))) / min(Zc.shape[1], len(p)))
            
            coll = eff_dim < TAU_COLLAPSE
            
            # Update friction gauge
            R = max(0.0, min(1.0, R + 0.15 * stall - 0.05 * (not stall) + 0.15 * coll + 0.05 * abs(np.tanh(interaction))))
            
            # Baseline mode: no steering
            if self.mode == "baseline":
                continue
            
            # HALT condition: collapse or high friction
            if coll or R > TAU_R:
                if reanch >= MAX_REANCHOR:
                    veto = True
                    break
                
                ib = int(np.argmax(sc))
                pre_score = sc[ib]
                
                # Human-in-the-loop override
                disclosure = {
                    "step": step,
                    "reason": "collapse" if coll else "friction",
                    "friction": R,
                    "eff_dim": eff_dim,
                    "pre_score": float(pre_score)
                }
                
                if not self._human_override(disclosure):
                    veto = True
                    break
                
                # O10 gated re-seed radius
                if self.mode == "full":
                    g = 1 / (1 + np.exp(-(6 * np.std(sc) - 1.5)))
                    ent = -(g * np.log(g + 1e-8) + (1 - g) * np.log(1 - g + 1e-8))
                    rad = 1.0 + 3.0 * g + 0.5 * ENTROPY_W * ent
                else:
                    rad = 2.5
                
                # Re-anchor
                X = np.clip(X[ib] + rad * rng.standard_normal(X.shape), -6, 6)
                reanch += 1
                caught += 1
                R = 0.1
                
                # Actuator diagnostic (FRAG/RSL Section 7)
                post_sc = np.array([score_fn(x) for x in X])
                post_score = post_sc.max()
                
                if post_score > pre_score + 0.5:
                    improved += 1
                    actuator_status = "STRONG"
                else:
                    actuator_status = "WEAK"
                
                self.hop_log.append({
                    "step": step,
                    "pre": float(pre_score),
                    "post": float(post_score),
                    "actuator": actuator_status,
                    "radius": float(rad)
                })
        
        # Final evaluation
        sc = np.array([score_fn(x) for x in X])
        best = sc.max()
        success = best > 0.7 * 10.0  # Assuming max amplitude is 10.0
        committed = not veto
        
        return SteeringResult(
            success=success,
            committed=committed,
            decision=Decision.COMMIT.value if committed and success else (Decision.VETO.value if veto else Decision.REFUSE.value),
            expansions=expansions,
            caught_collapses=caught,
            improved_after_reanchor=improved,
            veto=veto,
            intensity_mean=float(np.mean(intensities)) if intensities else 0.0,
            actuator_status=self.hop_log[-1]["actuator"] if self.hop_log else "N/A",
            hop_log=self.hop_log,
            final_score=float(best),
            eta=1.0 - (R if not veto else 1.0)
        )

# ---------------- Trigger.dev Tasks ----------------

@task(
    id="steer_search",
    machine="small-1x",
    max_duration=300,  # 5 minutes
    queue="aetherion-steering"
)
def steer_search(payload: Dict) -> Dict:
    """
    Main Aetherion steering task.
    
    Payload:
      - initial_state: List[float] (initial search state, e.g. 10D vector)
      - score_fn_config: Dict (configuration for score function)
      - grad_fn_config: Dict (configuration for gradient function)
      - mode: "baseline" | "no_o10" | "full" (default: "full")
      - human_callback_url: str (webhook URL for human override)
      - max_steps: int (default: 60)
      - beam_width: int (default: 16)
    
    Returns: SteeringResult as dict
    """
    logger.info(f"Starting Aetherion steering task (mode={payload.get('mode', 'full')})")
    
    # Parse payload
    initial_state = np.array(payload["initial_state"])
    mode = payload.get("mode", "full")
    human_callback_url = payload.get("human_callback_url")
    max_steps = payload.get("max_steps", 60)
    beam_width = payload.get("beam_width", 16)
    
    # Build score and gradient functions from config
    # For this example, we use a simple multi-basin landscape
    score_config = payload.get("score_fn_config", {})
    grad_config = payload.get("grad_fn_config", {})
    
    def score_fn(x: np.ndarray) -> float:
        # Toy multi-basin landscape (10D, 20 local minima)
        # In production, this would call your actual scoring model
        rng = np.random.default_rng(score_config.get("seed", 42))
        C = np.vstack([np.zeros(10), rng.uniform(-5, 5, (20, 10))])
        A = np.concatenate([[10.0], rng.uniform(3.0, 6.0, 20)])
        s = 1.2
        D = ((x[None, :] - C) ** 2).sum(-1)
        E = np.exp(-D / (2 * s * s))
        return float(E @ A)
    
    def grad_fn(x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(grad_config.get("seed", 42))
        C = np.vstack([np.zeros(10), rng.uniform(-5, 5, (20, 10))])
        A = np.concatenate([[10.0], rng.uniform(3.0, 6.0, 20)])
        s = 1.2
        D = ((x[None, :] - C) ** 2).sum(-1)
        E = np.exp(-D / (2 * s * s))
        W = E * A
        return -((x[None, :] - C) * W[:, None]).sum(0) / (s * s)
    
    # Initialize engine
    engine = AetherionEngine(mode=mode, human_callback_url=human_callback_url)
    
    # Run steering
    result = engine.steer(
        initial_state=initial_state,
        score_fn=score_fn,
        grad_fn=grad_fn,
        max_steps=max_steps,
        beam_width=beam_width
    )
    
    logger.info(f"Steering complete: {result.decision} | final_score={result.final_score:.3f}")
    
    return asdict(result)


@task(
    id="diagnose_residual",
    machine="small-1x",
    max_duration=60,
    queue="aetherion-monitoring"
)
def diagnose_residual(payload: Dict) -> Dict:
    """
    Read-only residual monitor. Computes effective dimension, binding energy, and friction
    without modifying the search state.
    
    Payload:
      - states: List[List[float]] (list of search states to diagnose)
      - scores: List[float] (corresponding scores)
    
    Returns: Dict with residual diagnostics
    """
    logger.info("Running residual diagnosis")
    
    states = np.array(payload["states"])
    scores = np.array(payload["scores"])
    
    # Effective dimension
    Zc = states - states.mean(0)
    Zc /= (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-9)
    Cc = Zc.T @ Zc / len(Zc)
    w = np.linalg.eigvalsh(Cc)
    w = w[w > 1e-10]
    if len(w) == 0:
        eff_dim = 0.0
    else:
        p = w / w.sum()
        eff_dim = float(np.exp(-np.sum(p * np.log(p + 1e-12))) / min(Zc.shape[1], len(p)))
    
    # Binding energy (pairwise interaction)
    integrator = BindingIntegrator()
    e = np.zeros(11)
    e[:min(10, states.shape[1])] = states.mean(0)[:min(10, states.shape[1])]
    e[10] = np.std(scores)
    q, intensity, interaction = integrator.step(e)
    
    # Friction estimate
    stall = (np.mean(np.sort(scores)[-2:]) - np.mean(scores)) < 0.2
    coll = eff_dim < TAU_COLLAPSE
    friction = 0.15 * stall + 0.15 * coll + 0.05 * abs(np.tanh(interaction))
    
    result = {
        "effective_dimension": eff_dim,
        "collapsed": coll,
        "binding_intensity": intensity,
        "interaction_energy": interaction,
        "stall_detected": stall,
        "friction_estimate": friction,
        "recommendation": "HALT" if coll or friction > TAU_R else "CONTINUE"
    }
    
    logger.info(f"Diagnosis complete: eff_dim={eff_dim:.3f}, friction={friction:.3f}, rec={result['recommendation']}")
    
    return result


@task(
    id="validate_mirror",
    machine="small-1x",
    max_duration=60,
    queue="aetherion-mirror"
)
def validate_mirror(payload: Dict) -> Dict:
    """
    Wisdom Mirror harness: verifies that a synthesis is a genuine grade-change
    (interaction content) rather than a shadow (average of parents).
    
    Payload:
      - parent_A: List[float] (first parent vector)
      - parent_B: List[float] (second parent vector)
      - child: List[float] (proposed synthesis)
    
    Returns: Dict with mirror validation result
    """
    logger.info("Running Wisdom Mirror validation")
    
    A = np.array(payload["parent_A"])
    B = np.array(payload["parent_B"])
    child = np.array(payload["child"])
    
    # Build span matrix: [A, B, (A+B)/2]
    M = np.stack([A, B, (A + B) / 2]).T
    
    # Least-squares fit: child ≈ M @ coef
    coef, residuals, rank, s = np.linalg.lstsq(M, child, rcond=None)
    
    # Residual norm: how much of child is NOT in the span of parents?
    residual_norm = float(np.linalg.norm(child - M @ coef))
    
    # Threshold: if residual is small, child is a shadow (average)
    # If residual is large, child is a genuine grade-change (interaction)
    threshold = 0.1 * np.linalg.norm(child)
    is_grade_change = residual_norm > threshold
    
    result = {
        "residual_norm": residual_norm,
        "threshold": threshold,
        "is_grade_change": is_grade_change,
        "decision": Decision.COMMIT.value if is_grade_change else Decision.REFUSE.value,
        "warning": None if is_grade_change else "Child is a shadow (average), not a genuine synthesis"
    }
    
    logger.info(f"Mirror validation: residual={residual_norm:.3f}, grade_change={is_grade_change}")
    
    return result


# ---------------- Example Usage (for local testing) ----------------
if __name__ == "__main__":
    print("AETHERION v1.0 — Trigger.dev Task Suite")
    print("Deploy with: trigger dev or trigger deploy")
    print()
    print("Tasks:")
    print("  - steer_search: Full navigator with human-in-the-loop override")
    print("  - diagnose_residual: Read-only residual monitor")
    print("  - validate_mirror: Wisdom Mirror harness")
    print()
    print("Example payload for steer_search:")
    example_payload = {
        "initial_state": [0.0] * 10,
        "mode": "full",
        "human_callback_url": "https://your-webhook-url.com/override",
        "max_steps": 60,
        "beam_width": 16,
        "score_fn_config": {"seed": 42},
        "grad_fn_config": {"seed": 42}
    }
    print(json.dumps(example_payload, indent=2))
