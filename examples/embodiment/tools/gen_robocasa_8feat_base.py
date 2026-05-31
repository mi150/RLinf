import pickle, glob, numpy as np

SRC = "/workspace/RLinf/checkpoints/probe_robocasa/rollout_collect_turnsinkspout_eval"
OUT = "/workspace/RLinf/checkpoints/probe_robocasa/robocasa_base_8feat_turnsinkspout.pkl"
EXEC_H = 20
D = 7  # n_action_dims

eps = []
for f in sorted(glob.glob(SRC + "/*.pkl")):
    eps.extend(pickle.load(open(f, "rb")))
print("loaded", len(eps), "episodes")

base = []
for ep in eps:
    ac = np.asarray(ep["action_chunk"], dtype=np.float32)  # (M, 50, 7)
    eefp = np.asarray(ep["eef_pos"], dtype=np.float32)      # (M, 3)
    M, H, _ = ac.shape
    exec_h = min(EXEC_H, H)
    overlap = max(H - exec_h, 0)
    feat = np.zeros((M, 9), dtype=np.float32)  # [a_norm, c_mse, x,y,z, vx,vy,vz, step]
    prev_chunk = None
    prev_eef = None
    for t in range(M):
        cur = ac[t]
        a_norm = np.sqrt(np.mean(cur[:exec_h, :D] ** 2))
        c_mse = 0.0
        if prev_chunk is not None and overlap > 0:
            a = prev_chunk[exec_h:exec_h + overlap, :D]
            b = cur[:overlap, :D]
            m = min(a.shape[0], b.shape[0])
            if m > 0:
                c_mse = float(np.mean((a[:m] - b[:m]) ** 2))
        eef = eefp[t, :3]
        vel = (eef - prev_eef) if prev_eef is not None else np.zeros(3, dtype=np.float32)
        feat[t] = [a_norm, c_mse, eef[0], eef[1], eef[2], vel[0], vel[1], vel[2], t]
        prev_chunk = cur
        prev_eef = eef
    base.append({"feat": feat, "length": int(M), "success": bool(ep["success"]),
                 "task_id": ep.get("task_id", "TurnSinkSpout")})

with open(OUT, "wb") as f:
    pickle.dump(base, f)

n_s = sum(1 for e in base if e["success"])
print("saved", len(base), "base episodes ->", OUT)
print("feat shape per ep[0]:", base[0]["feat"].shape, "(should be (M, 9))")
print("SR in base: %d/%d = %.0f%%" % (n_s, len(base), 100.0 * n_s / len(base)))
print("feat[0] sample row:", np.round(base[0]["feat"][1], 4))
