/**
 * Hit-testing against the per-pixel lookup maps — the mirror of engine/hittest.py.
 *
 * The engine does not decode PNGs. The shell hands over a raw pixel buffer and
 * its geometry (GdkPixbuf on the desktop, ImageData/ImageBitmap here) and this
 * just indexes it — which is why the same code works over a Uint8Array.
 *
 * Object ids are packed as R + G*256. `depth` is a single-channel count, NOT an
 * id: decoding it as one gives 257 for a depth of one.
 */
export class PixelMap {
  constructor(
    public data: Uint8Array | number[],
    public stride: number,
    public channels: number,
    public width: number,
    public height: number,
  ) {}

  contains(x: number, y: number): boolean {
    return x >= 0 && x < this.width && y >= 0 && y < this.height;
  }

  private offset(x: number, y: number): number {
    return y * this.stride + x * this.channels;
  }

  /** First channel (R, or the grey level of an L image). */
  valueAt(x: number, y: number): number {
    if (!this.contains(x, y)) return 0;
    return this.data[this.offset(x, y)];
  }

  /** Unpack an object id: R + G*256. 0 means background. */
  idAt(x: number, y: number): number {
    if (!this.contains(x, y)) return 0;
    const i = this.offset(x, y);
    if (this.channels >= 2) return this.data[i] + (this.data[i + 1] << 8);
    return this.data[i];
  }
}

export interface Hit {
  specific: number;
  general: number;
  depth: number;
}

export class HitMaps {
  constructor(
    public label: PixelMap | null = null,
    public general: PixelMap | null = null,
    public depth: PixelMap | null = null,
  ) {}

  clear() {
    this.label = this.general = this.depth = null;
  }

  get loaded(): boolean {
    return this.label !== null;
  }

  specificAt(x: number, y: number): number {
    return this.label ? this.label.idAt(x, y) : 0;
  }

  generalAt(x: number, y: number): number {
    return this.general ? this.general.idAt(x, y) : 0;
  }

  depthAt(x: number, y: number): number {
    return this.depth ? this.depth.valueAt(x, y) : 0;
  }

  hit(x: number, y: number): Hit {
    return {
      specific: this.specificAt(x, y),
      general: this.generalAt(x, y),
      depth: this.depthAt(x, y),
    };
  }
}
