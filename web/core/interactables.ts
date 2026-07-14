/**
 * The catalogue — the mirror of engine/interactables.py.
 *
 * Every setting the user can touch, declared once. The web sidebar is a walk
 * over Settings.describe(), exactly as the GTK one is, and both are checked
 * against spec/goldens/ui_schema.json.
 */
import {
  Choice,
  Setting,
  Group,
  Schema,
  Settings,
  CHOICE,
  BOOL,
  INT,
  COLOR,
  EXPANDER,
  PAGE,
  eq,
  ne,
  truthy,
} from "./settings";

export const SUBJECTS = [
  new Choice("birefnet-general", "General (objects, scenes)"),
  new Choice("birefnet-portrait", "Person / portrait"),
  new Choice("isnet-anime", "Anime / illustration"),
  new Choice("u2net", "Fast (lower quality)"),
  new Choice("custom", "Custom model…"),
];

export const MODELS = [
  new Choice("birefnet-general", "BiRefNet — General"),
  new Choice("birefnet-general-lite", "BiRefNet — General Lite"),
  new Choice("birefnet-massive", "BiRefNet — Massive"),
  new Choice("birefnet-portrait", "BiRefNet — Portrait"),
  new Choice("birefnet-hrsod", "BiRefNet — HRSOD"),
  new Choice("birefnet-dis", "BiRefNet — DIS"),
  new Choice("isnet-general-use", "ISNet — General"),
  new Choice("isnet-anime", "ISNet — Anime"),
  new Choice("u2net", "U2Net"),
  new Choice("u2net_human_seg", "U2Net — Human Seg"),
  new Choice("silueta", "Silueta"),
];

export const BACKGROUNDS = [
  new Choice("transparent", "Transparent"),
  new Choice("blur", "Blur background"),
  new Choice("#ffffff", "White"),
  new Choice("#000000", "Black"),
  new Choice("#00b140", "Green screen"),
  new Choice("custom", "Custom…"),
];

export const SEG_MODES = [
  new Choice("everything", "Everything"),
  new Choice("point", "Click to select"),
];

export const SEG_DETAIL = [
  new Choice(0, "Auto"),
  new Choice(16, "Fast"),
  new Choice(24, "Balanced"),
  new Choice(32, "Fine"),
  new Choice(44, "Maximum"),
];

export const SEG_FOCUS = [
  new Choice(800, "Quick"),
  new Choice(1600, "Normal"),
  new Choice(2600, "Relaxed"),
];

export const AUTO_MODEL = new Choice("auto", "Auto");

export function schema(): Schema {
  return new Schema([
    new Group(
      "model",
      "Model",
      [
        new Setting("subject", CHOICE, "Subject", {
          subtitle: "What the image mainly contains",
          options: SUBJECTS,
          default: "birefnet-general",
        }),
        new Setting("model", CHOICE, "Model", {
          subtitle: "Exact rembg model",
          options: MODELS,
          default: "birefnet-general",
          visibleWhen: [eq("subject", "custom")],
        }),
      ],
      [ne("page", "segment")],
    ),
    new Group(
      "seg",
      "Segmentation",
      [
        new Setting("seg_mode", CHOICE, "Mode", {
          options: SEG_MODES,
          default: "everything",
          surface: PAGE,
        }),
        new Setting("seg_focus", CHOICE, "Focus speed", {
          subtitle: "Hover this long to focus a sub-object",
          options: SEG_FOCUS,
          default: 1600,
        }),
        new Setting("seg_advanced", EXPANDER, "Advanced", {
          subtitle: "Model and detail",
          children: [
            new Setting("seg_model", CHOICE, "Model", {
              subtitle: "Auto — by GPU / VRAM",
              options: [AUTO_MODEL],
              default: "auto",
            }),
            new Setting("seg_detail", CHOICE, "Detail", {
              subtitle: "Finer finds more, but is slower",
              options: SEG_DETAIL,
              default: 0,
            }),
          ],
        }),
      ],
      [eq("page", "segment"), truthy("seg_available")],
    ),
    new Group("output", "Output", [
      new Setting("bg", CHOICE, "Background", {
        options: BACKGROUNDS,
        default: "transparent",
      }),
      new Setting("bg_color", COLOR, "Custom colour", {
        default: "#00b140",
        sensitiveWhen: [eq("bg", "custom")],
      }),
      new Setting("blur", INT, "Blur strength", {
        default: 20,
        minimum: 2,
        maximum: 80,
        step: 1,
        visibleWhen: [eq("bg", "blur")],
      }),
      new Setting("alpha", BOOL, "Alpha matting", {
        subtitle: "Cleaner edges, a little slower",
        default: false,
        visibleWhen: [ne("page", "segment")],
      }),
    ]),
  ]);
}

export function newSettings(): Settings {
  return new Settings(schema(), { page: "single", seg_available: false });
}

// ---------------------------------------------------------- the resolvers ---
export const modelId = (st: Settings) =>
  st.get("subject") === "custom" ? st.get("model") : st.get("subject");

export const background = (st: Settings) =>
  st.get("bg") === "custom" ? st.get("bg_color") : st.get("bg");

export const blurStrength = (st: Settings) => Number(st.get("blur"));
export const alphaMatting = (st: Settings) => Boolean(st.get("alpha"));

/** Points per side, or null to let the worker choose. Auto is 0 -> NULL. */
export const segDetail = (st: Settings) => st.get("seg_detail") || null;

export const segModel = (st: Settings) => st.get("seg_model") || "auto";
export const dwellMs = (st: Settings) => Number(st.get("seg_focus"));
export const segMode = (st: Settings) => String(st.get("seg_mode"));

// -------------------------------------------------- the runtime SAM ladder ---
export function vramLabel(mb: number): string {
  if (!mb) return "CPU · low VRAM";
  return `~${(mb / 1000).toFixed(1)} GB`.replace(".0", "");
}

export function segModelChoices(
  models: { rung: string; label: string; vram?: number }[],
): Choice[] {
  const out = [new Choice("auto", "Auto — best for your GPU")];
  for (const m of models) {
    out.push(new Choice(m.rung, `${m.label} · ${vramLabel(m.vram ?? 0)}`));
  }
  return out;
}
