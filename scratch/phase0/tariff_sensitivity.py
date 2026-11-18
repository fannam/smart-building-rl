"""Q3: per-watt normalized reward penalty of electricity vs gas, by hour/daytype/month.

Penalty per step for draw (E watts elec, G watts gas) in a K-AHU building is
  0.2*(pe*E + pg*G)/(K*(pe*80k + pg*205k)) + 0.2*(ke*E + kg*G)/(K*(ke*80k + kg*205k))
(the 1/(sum of weights)=1 factor omitted). Prices from the public starter-kit classes.
Prints the per-W penalty (x1e6, K=1) and the error from using the annual-mean gas price.
"""
import numpy as np
import pandas as pd
from smart_buildings.smart_control.reward.electricity_energy_cost import ElectricityEnergyCost
from smart_buildings.smart_control.reward.natural_gas_energy_cost import NaturalGasEnergyCost
from smart_buildings.smart_control.reward.energy_scale import C_ELECTRICITY_PER_AHU as CE, C_NATURAL_GAS_PER_AHU as CG

e, g = ElectricityEnergyCost(), NaturalGasEnergyCost()
dt = 300.0
def pen(ts, gas_price_override=None):
    s = pd.Timestamp(ts); t = s + pd.Timedelta(seconds=dt)
    pe, ke = e.cost(s, t, 1.0), e.carbon(s, t, 1.0)
    pg = g.cost(s, t, 1.0) if gas_price_override is None else gas_price_override
    kg = g.carbon(s, t, 1.0)
    de, dk = pe * CE + pg * CG, ke * CE + kg * CG
    return 0.2 * pe / de + 0.2 * ke / dk, 0.2 * pg / de + 0.2 * kg / dk, pg
mean_pg = np.mean([pen(f"2023-{m:02d}-10 03:00")[2] for m in range(1, 13)])
rows = []
for m in range(1, 13):
    for label, day, h in [("wkday03", 10, 3), ("wkday10", 10, 10), ("wkday15", 10, 15), ("wkend15", 14, 15)]:
        ts = pd.Timestamp(f"2023-{m:02d}-{day:02d} {h:02d}:00")
        # move to a weekday/weekend as labelled
        while (ts.weekday() < 5) != label.startswith("wkday"):
            ts += pd.Timedelta(days=1)
        pe_, pg_, _ = pen(ts)
        pe2, pg2, _ = pen(ts, mean_pg)
        rows.append((m, label, pe_ * 1e6, pg_ * 1e6, (pe2 - pe_) / pe_, (pg2 - pg_) / pg_))
df = pd.DataFrame(rows, columns=["month", "slot", "elec_pen_e6", "gas_pen_e6", "elec_err_meanpg", "gas_err_meanpg"])
pd.set_option("display.width", 160)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nmax |relative error| using annual-mean gas price: elec %.3f, gas %.3f" %
      (df.elec_err_meanpg.abs().max(), df.gas_err_meanpg.abs().max()))
print("gas/elec penalty ratio range: %.3f .. %.3f" % ((df.gas_pen_e6 / df.elec_pen_e6).min(), (df.gas_pen_e6 / df.elec_pen_e6).max()))
