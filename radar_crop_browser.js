/* RADKLIM RW: Monatsarchive im Browser laden, auf den Ausschnitt Prüm/Enz/Nims zuschneiden und nur den Ausschnitt speichern.
   Gleiches Ergebnis wie radar_crop.py (je Stunde 10 Zeichen Zeitstempel + rohe 2-Byte-Werte des Ausschnitts), aber je Jahr
   eine Datei crop_JJJJ.bin.gz; die Metadaten stehen in window.__radar.state.done. Läuft auf einer Seite von opendata.dwd.de (gleiche Herkunft).
   Aufruf: window.__radar.run([2021, 2020, ...])  -> Fortschritt in window.__radar.state */
(() => {
  const BASE = 'https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/radolan/reproc/2017_002/bin/';
  const BOX = {'1100x900': [416, 490, 144, 201], '900x900': [316, 390, 224, 281]};
  const dec = new TextDecoder('latin1');
  const state = {running: false, done: [], errors: [], current: null, bytes: 0, log: []};

  class Buf {                                   // Byte-Puffer über einem Stream
    constructor(stream) { this.r = stream.getReader(); this.chunks = []; this.len = 0; this.eof = false; }
    async fill(n) {
      while (this.len < n && !this.eof) {
        const {done, value} = await this.r.read();
        if (done) { this.eof = true; break; }
        this.chunks.push(value); this.len += value.length;
      }
      return this.len >= n;
    }
    async read(n) {
      if (!(await this.fill(n))) return null;
      const out = new Uint8Array(n);
      let o = 0;
      while (o < n) {
        const c = this.chunks[0], take = Math.min(c.length, n - o);
        out.set(c.subarray(0, take), o); o += take;
        if (take === c.length) this.chunks.shift(); else this.chunks[0] = c.subarray(take);
      }
      this.len -= n;
      return out;
    }
  }

  async function* tarMembers(stream) {          // [Name, Bytes] aller Dateien, auch aus verschachtelten Archiven
    const b = new Buf(stream);
    for (;;) {
      const h = await b.read(512);
      if (!h || h.every(x => x === 0)) return;
      let name = dec.decode(h.subarray(0, 100)).replace(/\0.*$/, '');
      const prefix = dec.decode(h.subarray(345, 500)).replace(/\0.*$/, '');
      if (prefix && dec.decode(h.subarray(257, 262)) === 'ustar') name = prefix + '/' + name;
      const size = parseInt(dec.decode(h.subarray(124, 136)).replace(/[\0 ]/g, ''), 8) || 0;
      const type = String.fromCharCode(h[156] || 48);
      const data = await b.read(size);
      if (data === null) throw new Error('Archiv bricht ab bei ' + name);
      if (size % 512) await b.read(512 - size % 512);
      if (type !== '0' && type !== '\0') continue;
      const base = name.split('/').pop();
      if (/\.(tar\.gz|tgz)$/.test(base)) yield* tarMembers(new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip')));
      else if (/\.tar$/.test(base)) yield* tarMembers(new Blob([data]).stream());
      else if (/\.gz$/.test(base)) {
        const raw = new Uint8Array(await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
        yield [base.slice(0, -3), raw];
      } else yield [base, data];
    }
  }

  function cropFile(name, raw, meta) {
    const s = /-(\d{10})-/.exec(name);
    if (!s) { meta.problems.push(name); return null; }
    const i = raw.indexOf(3);
    if (i < 0 || i > 2000) { meta.problems.push(name + ': kein Kopfende'); return null; }
    const head = dec.decode(raw.subarray(0, i));
    const g = /GP\s*(\d+)\s*x\s*(\d+)/.exec(head);
    if (!g) { meta.problems.push(name + ': kein GP'); return null; }
    const nr = +g[1], nc = +g[2], key = nr + 'x' + nc, box = BOX[key];
    const data = raw.subarray(i + 1);
    if (!box || data.length < nr * nc * 2) { meta.problems.push(name + ': Raster ' + key + ', Länge ' + data.length); return null; }
    if (!meta.grid) { meta.grid = [nr, nc]; meta.box = box; meta.header_first = head; }
    else if (meta.grid[0] !== nr || meta.grid[1] !== nc) { meta.problems.push(name + ': Raster ' + key); return null; }
    const pr = (/PR\s*(E[-+]?\d+)/.exec(head) || [])[1] || '?';
    meta.pr[pr] = (meta.pr[pr] || 0) + 1;
    const [r0, r1, c0, c1] = box, w = (c1 - c0) * 2;
    const out = new Uint8Array(10 + (r1 - r0) * w);
    for (let k = 0; k < 10; k++) out[k] = s[1].charCodeAt(k);
    for (let r = r0; r < r1; r++) out.set(data.subarray((r * nc + c0) * 2, (r * nc + c0) * 2 + w), 10 + (r - r0) * w);
    return out;
  }

  async function month(year, m, meta, recs) {
    const mm = String(m).padStart(2, '0'), url = `${BASE}${year}/RW2017.002_${year}${mm}.tar.gz`;
    const resp = await fetch(url);
    if (!resp.ok) { meta.months[mm] = {status: resp.status, hours: 0}; return; }
    const len = +resp.headers.get('content-length') || 0;
    let n = 0;
    for await (const [name, raw] of tarMembers(resp.body.pipeThrough(new DecompressionStream('gzip')))) {
      const rec = cropFile(name, raw, meta);
      if (rec) { recs.push(rec); n++; }
    }
    meta.months[mm] = {status: 200, hours: n, bytes: len};
    state.bytes += len;
  }

  function save(blob, name) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove();
  }

  async function year(y, months, tag) {
    const meta = {year: y, grid: null, box: null, header_first: null, pr: {}, months: {}, problems: []}, recs = [];
    const todo = months.slice();
    const worker = async () => { while (todo.length) { const m = todo.shift(); state.current = y + '-' + m; await month(y, m, meta, recs); } };
    await Promise.all([worker(), worker(), worker()]);
    meta.hours = recs.length; meta.n_problems = meta.problems.length; meta.problems = meta.problems.slice(0, 50);
    const gz = await new Response(new Blob(recs).stream().pipeThrough(new CompressionStream('gzip'))).blob();
    meta.crop_bytes = gz.size;
    save(gz, `crop_${tag || y}.bin.gz`);
    return meta;                                 // Metadaten gehen über state.done zurück (keine zweite Datei)
  }

  async function run(years, months, tag) {
    if (state.running) return 'läuft schon';
    state.running = true; state.t0 = Date.now();
    try {
      for (const y of years) {
        try { const m = await year(y, months || [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], tag); state.done.push(m); }
        catch (e) { state.errors.push({year: y, error: String(e)}); }
      }
    } finally { state.running = false; state.seconds = (Date.now() - state.t0) / 1000; }
    return 'fertig';
  }
  window.__radar = {run, state};
})();
