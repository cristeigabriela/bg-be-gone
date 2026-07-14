/**
 * The animation state machine — the mirror of engine/anim.py.
 *
 * The timings below are the single source of truth on the Python side too; if
 * one of these numbers drifts, every eased curve in the display list drifts with
 * it, and the goldens say so immediately.
 *
 * Timestamps are absolute microseconds, never deltas, so "the scene at time T"
 * is always the same answer.
 */
import { easeOut, easeOutBack } from "./geometry";

export const REVEAL_MS = 300.0;
export const POP_MS = 300.0;
export const DWELL_MS = 1600.0;
export const MORPH_MS = 240.0;
export const MORPH_N = 64;

export const PRESS_WAVE_MS = 720.0;
export const PRESS_SPAWN_MS = 450.0;
export const PRESS_DECAY_MS = 460.0;
export const PRESS_SPRING_MS = 360.0;

export const PULSE_HZ = 1.1;
export const ANT_SPEED = 24.0;
export const SCAN_PERIOD_S = 1.2;

export interface Tick {
  animating: boolean;
  dwellFired: boolean;
  pressRetired: boolean;
  morphRetired: boolean;
  changed: boolean;
}

export interface Press {
  held: boolean;
  fade: number;
  scale: number;
  glow: number;
  tint: number;
  ageMs: number;
}

export class AnimState {
  t0: number | null = null;
  pulse = 0.0;
  ant = 0.0;
  scanPhase = 0.0;
  reveal = 1.0;

  scanning = false;
  revealT0: number | null = null;
  pop = new Map<number, number>();

  dwellMs: number;
  dwellT0: number | null = null;
  drilled = false;

  pressObj = 0;
  pressPt: [number, number] | null = null;
  pressT0: number | null = null;
  releaseT0: number | null = null;

  morphT0: number | null = null;

  constructor(dwellMs: number = DWELL_MS) {
    this.dwellMs = dwellMs;
  }

  beginReveal(now: number) {
    this.reveal = 0.0;
    this.revealT0 = now;
  }

  beginPop(oid: number, now: number) {
    this.pop.set(oid, now);
  }

  beginDwell(now: number) {
    this.dwellT0 = now;
    this.drilled = false;
  }

  cancelDwell() {
    this.dwellT0 = null;
    this.drilled = false;
  }

  beginPress(oid: number, pt: [number, number], now: number) {
    this.pressObj = oid;
    this.pressPt = pt;
    this.pressT0 = now;
    this.releaseT0 = null;
  }

  releasePress(now: number): boolean {
    if (this.pressObj && this.releaseT0 === null) {
      this.releaseT0 = now;
      return true;
    }
    return false;
  }

  clearPress() {
    this.pressObj = 0;
    this.pressPt = null;
    this.pressT0 = null;
    this.releaseT0 = null;
  }

  beginMorph(now: number) {
    this.morphT0 = now;
  }

  clearMorph() {
    this.morphT0 = null;
  }

  get pressing(): boolean {
    return !!this.pressObj;
  }

  get held(): boolean {
    return !!this.pressObj && this.releaseT0 === null;
  }

  needsTick(active = false): boolean {
    return !!(
      active ||
      this.pressObj ||
      this.scanning ||
      this.morphT0 !== null ||
      this.revealT0 !== null ||
      this.pop.size
    );
  }

  advance(now: number, active = false): Tick {
    if (this.t0 === null) this.t0 = now;
    const el = (now - this.t0) / 1_000_000.0;

    this.pulse = 0.5 * (1.0 + Math.sin(el * 2.0 * Math.PI * PULSE_HZ));
    this.ant = el * ANT_SPEED;
    if (this.scanning) {
      this.scanPhase = (el / SCAN_PERIOD_S) % 1.0;
    }

    if (this.revealT0 !== null) {
      const p = (now - this.revealT0) / 1000.0 / REVEAL_MS;
      if (p >= 1.0) {
        this.reveal = 1.0;
        this.revealT0 = null;
      } else {
        this.reveal = easeOut(p);
      }
    }

    for (const [oid, t] of [...this.pop]) {
      if ((now - t) / 1000.0 > POP_MS) this.pop.delete(oid);
    }

    let dwellFired = false;
    if (
      this.dwellT0 !== null &&
      !this.drilled &&
      (now - this.dwellT0) / 1000.0 > this.dwellMs
    ) {
      this.drilled = true;
      dwellFired = true;
    }

    let pressRetired = false;
    if (
      this.pressObj &&
      this.releaseT0 !== null &&
      (now - this.releaseT0) / 1000.0 > PRESS_WAVE_MS
    ) {
      this.clearPress();
      pressRetired = true;
    }

    let morphRetired = false;
    if (this.morphT0 !== null && (now - this.morphT0) / 1000.0 >= MORPH_MS) {
      this.morphT0 = null;
      morphRetired = true;
    }

    return {
      animating: this.needsTick(active),
      dwellFired,
      pressRetired,
      morphRetired,
      changed: dwellFired || pressRetired || morphRetired,
    };
  }

  popScale(oid: number, now: number): number {
    const t = this.pop.get(oid);
    if (t === undefined) return 1.0;
    const p = (now - t) / 1000.0 / POP_MS;
    if (p <= 0.0 || p >= 1.0) return 1.0;
    return 1.0 + 0.09 * Math.sin(p * Math.PI) * (1.0 - p);
  }

  press(now: number): Press | null {
    if (!this.pressObj || this.pressT0 === null) return null;
    const held = this.releaseT0 === null;
    const relMs = held ? 0.0 : (now - this.releaseT0!) / 1000.0;
    const fade = held ? 1.0 : Math.max(0.0, 1.0 - relMs / PRESS_DECAY_MS);
    const scale = held
      ? 0.97
      : 0.97 + 0.03 * easeOutBack(Math.min(1.0, relMs / PRESS_SPRING_MS));
    const glow = held ? 0.34 + 0.24 * this.pulse : 0.4 * fade;
    const tint = 0.3 * (held ? 1.0 : fade);
    return {
      held,
      fade,
      scale,
      glow,
      tint,
      ageMs: (now - this.pressT0) / 1000.0,
    };
  }

  /** The ripple rings alive at `now`: [phase, alpha][]. */
  waves(now: number): [number, number][] {
    const p = this.press(now);
    if (p === null) return [];
    const ageTotal = p.ageMs;
    const lastSpawn = p.held
      ? ageTotal
      : (this.releaseT0! - this.pressT0!) / 1000.0;
    const n = Math.trunc(lastSpawn / PRESS_SPAWN_MS) + 1;
    const first = Math.max(
      0,
      Math.trunc((ageTotal - PRESS_WAVE_MS) / PRESS_SPAWN_MS),
    );
    const out: [number, number][] = [];
    for (let i = first; i < n; i++) {
      const age = ageTotal - i * PRESS_SPAWN_MS;
      if (age < 0.0 || age >= PRESS_WAVE_MS) continue;
      const ph = age / PRESS_WAVE_MS;
      out.push([ph, (1.0 - ph) * 0.5 * (p.held ? 1.0 : p.fade)]);
    }
    return out;
  }

  /** 0..1 eased progress of the outline tween, or null when idle. */
  morphProgress(now: number): number | null {
    if (this.morphT0 === null) return null;
    const raw = (now - this.morphT0) / 1000.0 / MORPH_MS;
    if (raw >= 1.0) return null;
    return easeOut(raw);
  }
}
