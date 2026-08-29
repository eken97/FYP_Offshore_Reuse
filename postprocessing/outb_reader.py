"""
Step 1 -- Targeted .outb reader.

Reads only the OpenFAST binary output channels the fatigue pipeline
actually needs (per member end: axial force + the two bending moments,
~672 columns out of ~2767) instead of loading the whole file through
openfast_toolbox's FASTOutputFile().toDataFrame(), which parses every
column and builds a full pandas DataFrame.

Scope (confirmed 05.08.2026, see Step 0): the pipeline reads OutAll
(J-style, "M{id}J{1|2}...") channels only. Member ID -> channel name is
direct (no lookup table needed) -- proved bit-exact against the file's
legacy NMOutputs channels in step0_channel_probe.py, which is now retired.

Steps:
    1. Read only the header of a .outb file (FileID=3 only -- see the
       assert in read_outb_header) -- no bulk data touched yet.
    2. Build a {channel_name: column_index} map from the header's own
       name list. Rebuilt PER FILE, never cached across files -- channel
       count varies between operating/parked runs (2767 vs 2765) and
       between old test data and the real campaign (e.g. LC_999 has
       2745), so a stale index map would silently read the wrong column.
    3. Memmap the data block and fancy-index only the wanted columns.
       Fancy-indexing a memmap copies just those columns out of the file;
       nothing else is read from disk.
"""
import struct
from pathlib import Path

import numpy as np

# region --- channel component sets ---
# The three components stress.py (Step 4) actually uses: axial force and
# the two bending moments. FKxe/FKye (transverse shear) and MKze (torsion)
# are deliberately excluded -- DNV nominal-stress fatigue for a straight
# member doesn't use them (see docs/decisions.md).
FATIGUE_COMPONENTS = ("FKze", "MKxe", "MKye")

# All 6 elastic ("K") components -- useful for QA/diagnostics beyond the
# fatigue calc itself.
ALL_K_COMPONENTS = ("FKxe", "FKye", "FKze", "MKxe", "MKye", "MKze")

# The inertial ("M") family. Read once in Stage 0 QA to confirm the
# negligibility finding from LC_999 (worst case ~42 N*m vs 3.3e4-3.2e5 N*m
# of elastic bending) holds per-run, then not used again.
ALL_M_COMPONENTS = ("FMxe", "FMye", "FMze", "MMxe", "MMye", "MMze")
# endregion


# region --- header + targeted read ---
def _read_outb_header_raw(path):
    """
    Parse a .outb header with NO integrity assert -- used only by callers
    that need to inspect a possibly-corrupt/truncated file without crashing
    (Stage 0 QA). Everything else should call read_outb_header instead,
    which wraps this and fails loud on a bad file.

    Also asserts FileID=3 (the only format handled) since that's a "this
    code doesn't support this file" condition, not a "this file is corrupt"
    condition -- different failure mode, still worth failing loud on.

    Returns a dict: fid, n_chan, n_t, t_start, t_incr, descr, names, units,
    data_offset, filesize.
    """
    path = Path(path)
    filesize = path.stat().st_size
    with open(path, "rb") as f:
        fid = struct.unpack("<h", f.read(2))[0]
        assert fid == 3, (
            f"{path}: FileID={fid}, only FileID=3 (NoCompressWithoutTime) is "
            f"supported -- extend read_outb_header before using this file."
        )
        n_chan = struct.unpack("<i", f.read(4))[0]
        n_t = struct.unpack("<i", f.read(4))[0]
        t_start, t_incr = struct.unpack("<dd", f.read(16))
        len_descr = struct.unpack("<i", f.read(4))[0]
        descr = f.read(len_descr).decode("latin-1")

        # NumChans+1 name/unit fields (slot 0 is "Time"), 10 bytes each,
        # space-padded.
        names = [f.read(10).decode("latin-1").strip() for _ in range(n_chan + 1)]
        units = [f.read(10).decode("latin-1").strip() for _ in range(n_chan + 1)]
        data_offset = f.tell()

    return dict(
        fid=fid, n_chan=n_chan, n_t=n_t, t_start=t_start, t_incr=t_incr,
        descr=descr, names=names, units=units,
        data_offset=data_offset, filesize=filesize,
    )


def read_outb_header(path):
    """
    Read only the header of an OpenFAST binary output file (.outb), and
    assert it is structurally sound (header/data sizes agree with the file
    on disk). Use this everywhere EXCEPT Stage 0 QA, which needs to inspect
    bad files too -- see _read_outb_header_raw for that case.

    Supports FileID=3 (FileFmtID_NoCompressWithoutTime) only -- every
    .outb checked in this project so far uses FileID=3 (verified directly,
    05.08.2026).
    """
    header = _read_outb_header_raw(path)
    assert header["data_offset"] + header["n_t"] * header["n_chan"] * 8 == header["filesize"], (
        f"{path}: header/data size mismatch -- "
        f"{header['data_offset']} + {header['n_t']}*{header['n_chan']}*8 != {header['filesize']}. "
        f"Corrupt or truncated file (e.g. a crashed run); do not read further."
    )
    return header


def channel_map(header):
    """{channel_name: column index into the (n_t, n_chan) data block}.

    Must be rebuilt per file -- see the module docstring. Never reuse a
    map built from one file's header against another file's data.
    """
    # names[0] is "Time" (not a data column); names[1:] line up with the
    # n_chan stored columns in order.
    data_names = header["names"][1:]
    idx = {}
    for k, name in enumerate(data_names):
        assert name not in idx, f"duplicate channel name {name!r} in header"
        idx[name] = k
    return idx


def read_channels(path, header, wanted_names):
    """
    Memmap + fancy-index just the requested columns.

    Returns (t, arr) where t is (n_t,) reconstructed from t_start/t_incr
    (Time is not stored as a data column in FileID=3 files) and arr is
    (n_t, len(wanted_names)) float64, in the same order as wanted_names.
    """
    cmap = channel_map(header)
    missing = [n for n in wanted_names if n not in cmap]
    assert not missing, (
        f"{path}: {len(missing)} requested channel(s) not in this file's "
        f"header, e.g. {missing[:5]} -- channel set varies by run "
        f"(operating vs parked, old test data vs campaign), never assume "
        f"a channel list built for one file applies to another."
    )
    cols = [cmap[n] for n in wanted_names]
    mm = np.memmap(
        path, dtype="<f8", mode="r",
        offset=header["data_offset"],
        shape=(header["n_t"], header["n_chan"]),
    )
    arr = np.asarray(mm[:, cols])  # fancy-index copies out of the memmap
    t = header["t_start"] + np.arange(header["n_t"]) * header["t_incr"]
    return t, arr
# endregion


# region --- channel-name builder ---
def member_end_channels(member_ids, components=FATIGUE_COMPONENTS):
    """
    Build the OutAll (J-style) channel names for a set of members.

    Returns a flat list ordered member, then end (J1, J2), then component
    -- e.g. for member_ids=[4], components=FATIGUE_COMPONENTS:
    ['M4J1FKze', 'M4J1MKxe', 'M4J1MKye', 'M4J2FKze', 'M4J2MKxe', 'M4J2MKye']

    {id} in "M{id}J..." is the MemberID directly (see Step 0) -- no lookup
    table needed.
    """
    names = []
    for mid in member_ids:
        for end in (1, 2):
            for c in components:
                names.append(f"M{mid}J{end}{c}")
    return names
# endregion


# region --- self-check: cross-validate against openfast_toolbox ---
def _self_check():
    """
    Cross-check read_channels against FASTOutputFile(...).toDataFrame() on
    a real run, and time both. Run as `python outb_reader.py`.
    """
    import time
    import random

    project = Path(__file__).resolve().parent.parent.parent   # .../OpenFast
    outb_path = (project / "TestScenario" / "LC_V20_H3p5_T8" / "S100001" /
                 "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb")
    assert outb_path.exists(), f"missing fixture: {outb_path}"

    header = read_outb_header(outb_path)
    print(f"{outb_path.name}")
    print(f"  n_chan={header['n_chan']}  n_t={header['n_t']}  "
          f"dt={header['t_incr']}  data_offset={header['data_offset']}")

    # The actual target read: all 112 members, both ends, the 3 fatigue
    # components -- 672 columns.
    wanted = member_end_channels(range(1, 113), FATIGUE_COMPONENTS)
    assert len(wanted) == 112 * 2 * 3 == 672

    t0 = time.time()
    t, arr = read_channels(outb_path, header, wanted)
    dt_targeted = time.time() - t0
    print(f"\ntargeted read: {len(wanted)} cols x {header['n_t']} rows "
          f"in {dt_targeted:.3f} s  ({arr.nbytes / 1e6:.1f} MB)")
    print(f"  t[0]={t[0]}  t[-1]={t[-1]}  (expect 0.0 .. "
          f"{(header['n_t']-1)*header['t_incr']:.1f})")
    assert t[0] == 0.0
    assert abs(t[-1] - (header["n_t"] - 1) * header["t_incr"]) < 1e-9

    # Cross-check against openfast_toolbox on a random sample of channels.
    from openfast_toolbox.io import FASTOutputFile

    t0 = time.time()
    df = FASTOutputFile(str(outb_path)).toDataFrame()
    dt_toolbox = time.time() - t0
    print(f"\nopenfast_toolbox full read: {dt_toolbox:.3f} s  "
          f"({df.memory_usage(deep=True).sum() / 1e6:.1f} MB)  "
          f"speedup {dt_toolbox / dt_targeted:.1f}x")

    # openfast_toolbox appends "_[unit]" to every column name.
    toolbox_bare = {c.split("_[")[0]: c for c in df.columns}

    random.seed(0)
    sample = random.sample(wanted, 20)
    print(f"\ncross-check on {len(sample)} random channels vs FASTOutputFile:")
    max_diff = 0.0
    for name in sample:
        mine = arr[:, wanted.index(name)]
        theirs = df[toolbox_bare[name]].to_numpy()
        d = float(np.max(np.abs(mine - theirs)))
        max_diff = max(max_diff, d)
    print(f"  max|diff| over {len(sample)} channels x {header['n_t']} rows "
          f"= {max_diff:.6e}")
    assert max_diff == 0.0, "targeted reader disagrees with FASTOutputFile -- STOP"
    print("  EXACT MATCH")

    # Same check for the inertial family, read once here (per the module
    # docstring) rather than as part of the routine fatigue read.
    m_wanted = member_end_channels(range(1, 113), ALL_M_COMPONENTS)
    _, m_arr = read_channels(outb_path, header, m_wanted)
    m_sample = random.sample(m_wanted, 10)
    m_max_diff = 0.0
    for name in m_sample:
        mine = m_arr[:, m_wanted.index(name)]
        theirs = df[toolbox_bare[name]].to_numpy()
        m_max_diff = max(m_max_diff, float(np.max(np.abs(mine - theirs))))
    print(f"\ninertial (FM/MM) family cross-check, {len(m_sample)} channels: "
          f"max|diff|={m_max_diff:.6e}")
    assert m_max_diff == 0.0


if __name__ == "__main__":
    _self_check()
