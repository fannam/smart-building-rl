import pandas as pd
from smart_buildings.smart_control.reward.electricity_energy_cost import ElectricityEnergyCost
from smart_buildings.smart_control.reward.natural_gas_energy_cost import NaturalGasEnergyCost
e, g = ElectricityEnergyCost(), NaturalGasEnergyCost()
Emax, Gmax = 80000.0, 205000.0  # per AHU; ratio is building-size independent
for label, ts in [("wkday 03h", "2023-01-10 03:00"), ("wkday 10h", "2023-01-10 10:00"),
                  ("wkday 15h", "2023-01-10 15:00"), ("wkend 15h", "2023-01-14 15:00"),
                  ("jul wkday 15h", "2023-07-11 15:00")]:
    s = pd.Timestamp(ts); t = s + pd.Timedelta(minutes=5)
    ce, cg = e.cost(s, t, 1.0), g.cost(s, t, 1.0)
    ke, kg = e.carbon(s, t, 1.0), g.carbon(s, t, 1.0)
    cmax = e.cost(s, t, Emax) + g.cost(s, t, Gmax)
    kmax = e.carbon(s, t, Emax) + g.carbon(s, t, Gmax)
    # reward penalty per AHU-normalised watt (x1e6 for readability), weights 0.2/0.2, /1.0 total
    pe = 0.2 * ce / cmax + 0.2 * ke / kmax
    pg = 0.2 * cg / cmax + 0.2 * kg / kmax
    print(f"{label:14s} elec {pe*1e6:.3f}  gas {pg*1e6:.3f}  gas/elec {pg/pe:.2f}")
