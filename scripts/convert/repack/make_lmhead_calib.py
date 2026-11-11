import numpy as np, tarfile, io, os
out = os.path.expanduser("~/work/breeze-tts/repack/calib/input.tar.gz")
rng = np.random.default_rng(20261111)
members = []
for i in range(64):
    scale = 0.5 + 1.5 * rng.random()
    a = (rng.standard_normal((1, 1, 1024)) * scale).astype(np.float32)
    buf = io.BytesIO(); np.save(buf, a); buf.seek(0)
    members.append(("input_%03d.npy" % i, buf.getvalue()))
with tarfile.open(out, "w:gz") as tar:
    for name, data in members:
        info = tarfile.TarInfo(name); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
print("wrote", out, len(members), "samples")
