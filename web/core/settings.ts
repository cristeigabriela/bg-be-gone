/** Settings as data — the mirror of engine/settings.py. */

export const CHOICE = "choice";
export const BOOL = "bool";
export const INT = "int";
export const COLOR = "color";
export const EXPANDER = "expander";

export const SIDEBAR = "sidebar";
export const PAGE = "page";

export type Value = string | number | boolean | null;

export class Choice {
  constructor(
    public value: Value,
    public label: string,
  ) {}
}

type Op = "==" | "!=" | "truthy";

export class Cond {
  constructor(
    public key: string,
    public op: Op,
    public value: Value = null,
  ) {}

  test(env: Record<string, Value>): boolean {
    const v = env[this.key];
    if (this.op === "==") return v === this.value;
    if (this.op === "!=") return v !== this.value;
    return !!v;
  }
}

export const eq = (k: string, v: Value) => new Cond(k, "==", v);
export const ne = (k: string, v: Value) => new Cond(k, "!=", v);
export const truthy = (k: string) => new Cond(k, "truthy");

export interface SettingOpts {
  subtitle?: string;
  default?: Value;
  options?: Choice[];
  minimum?: number;
  maximum?: number;
  step?: number;
  pageStep?: number;
  children?: Setting[];
  visibleWhen?: Cond[];
  sensitiveWhen?: Cond[];
  surface?: string;
}

export class Setting {
  subtitle: string | null;
  default: Value;
  options: Choice[];
  minimum: number | null;
  maximum: number | null;
  step: number;
  pageStep: number;
  children: Setting[];
  visibleWhen: Cond[];
  sensitiveWhen: Cond[];
  surface: string;

  constructor(
    public id: string,
    public kind: string,
    public label: string,
    o: SettingOpts = {},
  ) {
    this.subtitle = o.subtitle ?? null;
    this.default = o.default ?? null;
    this.options = o.options ?? [];
    this.minimum = o.minimum ?? null;
    this.maximum = o.maximum ?? null;
    this.step = o.step ?? 1;
    this.pageStep = o.pageStep ?? 5;
    this.children = o.children ?? [];
    this.visibleWhen = o.visibleWhen ?? [];
    this.sensitiveWhen = o.sensitiveWhen ?? [];
    this.surface = o.surface ?? SIDEBAR;
  }

  *walk(): Generator<Setting> {
    yield this;
    for (const c of this.children) yield* c.walk();
  }
}

export class Group {
  constructor(
    public id: string,
    public title: string,
    public settings: Setting[] = [],
    public visibleWhen: Cond[] = [],
  ) {}
}

export class Schema {
  constructor(public groups: Group[] = []) {}

  *walk(): Generator<Setting> {
    for (const g of this.groups) for (const s of g.settings) yield* s.walk();
  }

  get(sid: string): Setting {
    for (const s of this.walk()) if (s.id === sid) return s;
    throw new Error(`unknown setting ${sid}`);
  }
}

export class Settings {
  values = new Map<string, Value>();
  private opts = new Map<string, Choice[]>();
  context: Record<string, Value> = {};

  constructor(
    public schema: Schema,
    context: Record<string, Value> = {},
  ) {
    for (const s of schema.walk()) {
      if (s.kind !== EXPANDER) this.values.set(s.id, s.default);
      this.opts.set(s.id, [...s.options]);
    }
    this.context = { ...context };
  }

  get(sid: string): Value {
    return this.values.get(sid) ?? null;
  }

  set(sid: string, value: Value): boolean {
    if (this.values.get(sid) === value) return false;
    this.values.set(sid, value);
    return true;
  }

  env(): Record<string, Value> {
    return { ...Object.fromEntries(this.values), ...this.context };
  }

  setContext(kv: Record<string, Value>) {
    Object.assign(this.context, kv);
  }

  options(sid: string): Choice[] {
    return this.opts.get(sid) ?? [];
  }

  setOptions(sid: string, choices: Choice[]) {
    this.opts.set(sid, [...choices]);
    if (!choices.some((c) => c.value === this.values.get(sid))) {
      this.values.set(sid, choices.length ? choices[0].value : null);
    }
  }

  index(sid: string): number {
    const o = this.options(sid);
    for (let i = 0; i < o.length; i++) {
      if (o[i].value === this.values.get(sid)) return i;
    }
    return 0;
  }

  setIndex(sid: string, i: number): boolean {
    const o = this.options(sid);
    if (i >= 0 && i < o.length) return this.set(sid, o[i].value);
    return false;
  }

  label(sid: string): string {
    for (const c of this.options(sid)) {
      if (c.value === this.values.get(sid)) return c.label;
    }
    return "";
  }

  visible(sid: string): boolean {
    const env = this.env();
    return this.schema.get(sid).visibleWhen.every((c) => c.test(env));
  }

  sensitive(sid: string): boolean {
    const env = this.env();
    return this.schema.get(sid).sensitiveWhen.every((c) => c.test(env));
  }

  groupVisible(gid: string): boolean {
    const env = this.env();
    for (const g of this.schema.groups) {
      if (g.id === gid) return g.visibleWhen.every((c) => c.test(env));
    }
    throw new Error(`unknown group ${gid}`);
  }

  /** Everything a UI needs to build itself. The whole contract. */
  describe(surface: string | null = null): any {
    return {
      groups: this.schema.groups.map((g) => ({
        id: g.id,
        title: g.title,
        visible: this.groupVisible(g.id),
        rows: g.settings
          .filter((s) => surface === null || s.surface === surface)
          .map((s) => this.row(s)),
      })),
    };
  }

  private row(s: Setting): any {
    const d: any = {
      id: s.id,
      kind: s.kind,
      label: s.label,
      visible: this.visible(s.id),
      sensitive: this.sensitive(s.id),
    };
    if (s.subtitle) d.subtitle = s.subtitle;
    if (s.kind === EXPANDER) {
      d.rows = s.children.map((c) => this.row(c));
      return d;
    }
    d.value = this.values.get(s.id) ?? null;
    if (s.kind === CHOICE) {
      d.options = this.options(s.id).map((c) => ({
        value: c.value,
        label: c.label,
      }));
      d.selected = this.index(s.id);
    } else if (s.kind === INT) {
      d.min = s.minimum;
      d.max = s.maximum;
      d.step = s.step;
    }
    return d;
  }
}
