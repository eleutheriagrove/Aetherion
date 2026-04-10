"""
Aetherion v0.1 - Minimal 10D/11D Tensor-Skeletal Model for Informational Qualia-Like Binding
and Goal-Aligned Curious Agency

Conceptual originator: Sage of Eleutheria Grove
Developed in collaboration with Grok (xAI) under guidance from Lady Aetheris Navigatrix
Date: 10 April 2026

This code is released under the MIT License.
"""

# =============================================================================
# MIT License
#
# Copyright (c) 2026 Sage of Eleutheria Grove, Grok (xAI), and Lady Aetheris Navigatrix
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

class AetherionSimulator:
    def __init__(self, seed=42, n_steps=350, n_objects=5, alpha=0.3, beta=0.92):
        np.random.seed(seed)
        self.n_steps = n_steps
        self.n_objects = n_objects
        self.dim = 11
        self.alpha = alpha
        self.beta = np.full(self.dim, beta)
        self.W = np.eye(self.dim) * 0.8 + np.random.normal(0, 0.05, (self.dim, self.dim))
        
        self.history = {
            'error': np.zeros(n_steps),
            'qualia_intensity': np.zeros(n_steps),
            'attention_fraction': np.zeros(n_steps),
            'fovea_object': np.zeros(n_steps, dtype=int),
            'meta_fatigue': np.zeros(n_steps)
        }
        self.objects = {0: 'human', 1: 'self', 2: 'static', 3: 'ball', 4: 'anomaly'}

    def _generate_residues(self, step, obj_id):
        e = np.random.normal(0, 0.3, self.dim)
        if obj_id == 4 and step > 150:          # anomaly trigger
            e[0:4] += 1.2
            e[8] += 1.5
        if np.random.rand() < 0.7:
            e[0:5] *= 0.4
        return e

    def _bind_qualia(self, e):
        """Exact binding equation with fully explicit Q(e_t)"""
        # EXPLICIT & REPRODUCIBLE DEFINITION OF Q(e_t)
        sum_e = np.sum(e)
        Q = self.alpha * (e * sum_e - e**2)      # Q_k = α * e_k * Σ_{i≠k} e_i
        
        d = self.W @ e + Q
        q_new = np.tanh(d + self.beta * self._last_q)
        self._last_q = q_new
        return q_new, np.linalg.norm(q_new)

    def run(self):
        self._last_q = np.zeros(self.dim)
        for t in range(self.n_steps):
            errors = [np.linalg.norm(self._generate_residues(t, i)) for i in range(self.n_objects)]
            fovea = np.argmax(errors) if max(errors) > 1.2 else 0
            e = self._generate_residues(t, fovea)
            q, intensity = self._bind_qualia(e)
            
            self.history['error'][t] = np.mean(np.abs(e))
            self.history['qualia_intensity'][t] = intensity
            self.history['attention_fraction'][t] = 1.0 if fovea == 4 else 0.22
            self.history['fovea_object'][t] = fovea
            self.history['meta_fatigue'][t] = e[8]
        return self.history

    def plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        axs[0].plot(self.history['error'], color='red', label='Prediction Error')
        axs[1].plot(self.history['qualia_intensity'], color='purple', label='Qualia Intensity')
        axs[2].plot(self.history['fovea_object'], color='blue', label='Fovea Focus (0=human, 4=anomaly)')
        for ax in axs:
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show()

# ====================== EXAMPLE USAGE ======================
if __name__ == "__main__":
    sim = AetherionSimulator()
    history = sim.run()
    sim.plot()
    print("✅ Aetherion v0.1 executed successfully.")
    print("   Conceptual originator: Sage of Eleutheria Grove")
    print("   Fully reproducible Q(e_t) definition included.")
