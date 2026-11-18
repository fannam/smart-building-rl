import sys, time, json, torch
from pathlib import Path
sys.path.insert(0, "src")
torch.set_num_threads(4)
from dynamics.train import fold_data, make_windows, loader, multitask_loss, inner_split
from dynamics.model import GRUDynamics, Norm, rollout
cfg = json.load(open("configs/dynamics_gru.json")); fold = json.load(open("configs/leave_one_building_out.json"))["folds"][5]
cache = Path("scratch/phase1_5/cache")
t=time.time(); root, tr, te, stats = fold_data(cfg, fold, cache); print("fit norm", round(time.time()-t,1), "s", {k:(round(v['mean'],4),round(v['std'],4)) for k,v in stats.stats.items()})
norm = Norm.from_stats(stats)
f,_ = inner_split(24)
t=time.time(); ds = make_windows(root, tr[:6], 12, 24, cache, stride=4, t0_filter=f); print("windows", len(ds), round(time.time()-t,1),"s")
t=time.time(); b = next(iter(loader(ds, 32, True, 0, 0))); print("collate", round(time.time()-t,2), {k: tuple(v.shape) for k,v in b["inputs"].items()}, b["info"]["V"][:3])
m = GRUDynamics(norm); opt = torch.optim.AdamW(m.parameters(), 2e-3)
print("params", sum(p.numel() for p in m.parameters()))
for it in range(30):
    t=time.time(); p = rollout(m, b); loss, parts = multitask_loss(p, b, norm, cfg["loss_weights"]); opt.zero_grad(); loss.backward(); opt.step()
    if it % 5 == 0: print(it, {k: round(v,4) for k,v in parts.items()}, f"{time.time()-t:.2f}s/step")
