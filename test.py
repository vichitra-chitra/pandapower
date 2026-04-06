import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pandapower as pp
import pandapower.plotting as plot

# ============================================================
# PRESENTATION-GRADE LOCAL GRID MODEL WITH PROTECTION DEVICES
# ============================================================
# Features
# - 20 kV / 0.4 kV transformer
# - 20 homes with rooftop PV + EV charging
# - separate 50 kWp PV plant
# - separate 100 kW wind plant
# - 5 industrial customers (50–100 kW each)
# - annotated bus voltages / currents / powers
# - annotated line distances / currents / loading
# - switches (CB / LBS / DS)
# - optional fuse objects (if pandapower protection module is available)
# - clean manual coordinates for presentation in VS Code / Jupyter
# ============================================================

# -----------------------------
# deterministic data for repeatability
# -----------------------------
np.random.seed(42)

# -----------------------------
# helper functions
# -----------------------------
def pf_to_q_mvar(p_mw, pf=0.97):
    """Return reactive power (MVAr) for inductive load at given power factor."""
    phi = math.acos(pf)
    return p_mw * math.tan(phi)

def bus_result_current_ka(vn_kv, p_mw, q_mvar):
    """Approximate 3-phase bus current from bus P and Q results."""
    s_mva = math.sqrt(p_mw**2 + q_mvar**2)
    if vn_kv <= 0:
        return 0.0
    return s_mva / (math.sqrt(3) * vn_kv)

def create_lv_line(net, from_bus, to_bus, length_km, name, max_i_ka=0.315):
    """
    Create a representative 0.4 kV underground cable using explicit parameters.
    Values are generic but realistic for a local feeder study.
    """
    return pp.create_line_from_parameters(
        net,
        from_bus=from_bus,
        to_bus=to_bus,
        length_km=length_km,
        r_ohm_per_km=0.206,   # representative LV cable resistance
        x_ohm_per_km=0.080,   # representative LV cable reactance
        c_nf_per_km=210,
        max_i_ka=max_i_ka,
        name=name
    )

# -----------------------------
# create network
# -----------------------------
net = pp.create_empty_network(name="Presentation_Grade_Local_Grid")

# -----------------------------
# buses
# -----------------------------
# 20 kV source bus
b_hv = pp.create_bus(net, vn_kv=20.0, name="HV_SOURCE")

# LV main busbar
b_lv_main = pp.create_bus(net, vn_kv=0.4, name="LV_MAIN")

# Residential feeder A
b_res_nw = pp.create_bus(net, vn_kv=0.4, name="RES_NW")
b_res_w  = pp.create_bus(net, vn_kv=0.4, name="RES_W")
b_res_sw = pp.create_bus(net, vn_kv=0.4, name="RES_SW")

# Residential feeder B
b_res_ne = pp.create_bus(net, vn_kv=0.4, name="RES_NE")
b_res_e  = pp.create_bus(net, vn_kv=0.4, name="RES_E")
b_res_se = pp.create_bus(net, vn_kv=0.4, name="RES_SE")

# Industrial park and DER park
b_ind = pp.create_bus(net, vn_kv=0.4, name="IND_PARK")
b_der = pp.create_bus(net, vn_kv=0.4, name="DER_PARK")

# -----------------------------
# manual coordinates (clean layout)
# -----------------------------
net.bus_geodata = pd.DataFrame(index=net.bus.index, columns=["x", "y"], dtype=float)

coords = {
    b_hv:      (0.0,  6.0),
    b_lv_main: (0.0,  4.8),

    b_res_nw:  (-3.2, 4.1),
    b_res_w:   (-4.5, 3.1),
    b_res_sw:  (-5.5, 2.1),

    b_res_ne:  ( 3.2, 4.1),
    b_res_e:   ( 4.5, 3.1),
    b_res_se:  ( 5.5, 2.1),

    b_ind:     ( 0.0, 2.8),
    b_der:     ( 0.0, 1.3),
}

for bus, (x, y) in coords.items():
    net.bus_geodata.loc[bus, "x"] = x
    net.bus_geodata.loc[bus, "y"] = y

# -----------------------------
# source + transformer
# -----------------------------
pp.create_ext_grid(net, bus=b_hv, vm_pu=1.02, name="GRID_20kV")

trafo = pp.create_transformer_from_parameters(
    net,
    hv_bus=b_hv,
    lv_bus=b_lv_main,
    sn_mva=0.63,
    vn_hv_kv=20.0,
    vn_lv_kv=0.4,
    vk_percent=6.0,
    vkr_percent=1.2,
    pfe_kw=0.94,
    i0_percent=0.25,
    shift_degree=30,
    name="TR_630kVA_20_0.4kV_Dyn11"
)

# -----------------------------
# transformer protection / switching
# -----------------------------
sw_hv_cb = pp.create_switch(net, bus=b_hv, element=trafo, et="t", type="CB", closed=True, name="CB_HV_TRAFO")
sw_lv_cb = pp.create_switch(net, bus=b_lv_main, element=trafo, et="t", type="CB", closed=True, name="CB_LV_TRAFO")
sw_hv_ds = pp.create_switch(net, bus=b_hv, element=trafo, et="t", type="DS", closed=True, name="DS_HV_ISOLATOR")

# -----------------------------
# lines
# -----------------------------
# residential feeder A
l_main_nw = create_lv_line(net, b_lv_main, b_res_nw, 0.18, "L_MAIN_RES_NW", max_i_ka=0.315)
l_nw_w    = create_lv_line(net, b_res_nw,  b_res_w,  0.16, "L_RES_NW_RES_W", max_i_ka=0.250)
l_w_sw    = create_lv_line(net, b_res_w,   b_res_sw, 0.14, "L_RES_W_RES_SW", max_i_ka=0.200)

# residential feeder B
l_main_ne = create_lv_line(net, b_lv_main, b_res_ne, 0.18, "L_MAIN_RES_NE", max_i_ka=0.315)
l_ne_e    = create_lv_line(net, b_res_ne,  b_res_e,  0.16, "L_RES_NE_RES_E", max_i_ka=0.250)
l_e_se    = create_lv_line(net, b_res_e,   b_res_se, 0.14, "L_RES_E_RES_SE", max_i_ka=0.200)

# industrial + DER feeders
l_main_ind = create_lv_line(net, b_lv_main, b_ind, 0.12, "L_MAIN_IND", max_i_ka=0.400)
l_main_der = create_lv_line(net, b_lv_main, b_der, 0.10, "L_MAIN_DER", max_i_ka=0.315)

# -----------------------------
# feeder switches / protection zoning
# -----------------------------
# Feeder head circuit breakers
sw_cb_res_a = pp.create_switch(net, bus=b_lv_main, element=l_main_nw, et="l", type="CB",  closed=True, name="CB_FEEDER_RES_A")
sw_cb_res_b = pp.create_switch(net, bus=b_lv_main, element=l_main_ne, et="l", type="CB",  closed=True, name="CB_FEEDER_RES_B")
sw_cb_ind   = pp.create_switch(net, bus=b_lv_main, element=l_main_ind, et="l", type="CB",  closed=True, name="CB_FEEDER_IND")
sw_cb_der   = pp.create_switch(net, bus=b_lv_main, element=l_main_der, et="l", type="CB",  closed=True, name="CB_FEEDER_DER")

# Sectionalizing / load-break switches downstream
sw_lbs_a_1 = pp.create_switch(net, bus=b_res_nw, element=l_nw_w, et="l", type="LBS", closed=True, name="LBS_RES_A_1")
sw_lbs_a_2 = pp.create_switch(net, bus=b_res_w,  element=l_w_sw, et="l", type="LBS", closed=True, name="LBS_RES_A_2")
sw_lbs_b_1 = pp.create_switch(net, bus=b_res_ne, element=l_ne_e, et="l", type="LBS", closed=True, name="LBS_RES_B_1")
sw_lbs_b_2 = pp.create_switch(net, bus=b_res_e,  element=l_e_se, et="l", type="LBS", closed=True, name="LBS_RES_B_2")

# -----------------------------
# optional fuse protection objects
# -----------------------------
protection_available = False
try:
    from pandapower.protection.protection_devices.fuse import Fuse
    protection_available = True

    # Feeder head fuses / backup fuses (illustrative)
    Fuse(net, switch_index=sw_cb_res_a, fuse_type="none", rated_i_a=250, name="FUSE_RES_A")
    Fuse(net, switch_index=sw_cb_res_b, fuse_type="none", rated_i_a=250, name="FUSE_RES_B")
    Fuse(net, switch_index=sw_cb_ind,   fuse_type="none", rated_i_a=400, name="FUSE_IND")
    Fuse(net, switch_index=sw_cb_der,   fuse_type="none", rated_i_a=250, name="FUSE_DER")

    # Section fuses
    Fuse(net, switch_index=sw_lbs_a_1, fuse_type="none", rated_i_a=160, name="FUSE_RES_A_1")
    Fuse(net, switch_index=sw_lbs_a_2, fuse_type="none", rated_i_a=125, name="FUSE_RES_A_2")
    Fuse(net, switch_index=sw_lbs_b_1, fuse_type="none", rated_i_a=160, name="FUSE_RES_B_1")
    Fuse(net, switch_index=sw_lbs_b_2, fuse_type="none", rated_i_a=125, name="FUSE_RES_B_2")

except Exception:
    protection_available = False

# -----------------------------
# residential customers
# 20 homes distributed over 6 residential buses
# each home has:
# - base household load
# - rooftop PV
# - EV charging load
# -----------------------------
residential_buses = [
    b_res_nw, b_res_nw, b_res_nw, b_res_nw,
    b_res_w,  b_res_w,  b_res_w,
    b_res_sw, b_res_sw, b_res_sw,
    b_res_ne, b_res_ne, b_res_ne, b_res_ne,
    b_res_e,  b_res_e,  b_res_e,
    b_res_se, b_res_se, b_res_se
]

home_base_kw = [3.2, 2.8, 4.0, 3.6, 3.1, 2.7, 3.5, 2.9, 3.4, 3.0,
                3.8, 2.6, 3.3, 4.1, 2.9, 3.2, 3.7, 2.8, 3.4, 3.1]

home_pv_kw   = [5.0, 4.2, 5.5, 4.8, 4.5, 4.0, 5.2, 4.3, 4.9, 4.1,
                5.4, 4.0, 4.6, 5.7, 4.2, 4.8, 5.1, 4.3, 4.7, 4.4]

ev_kw        = [2.5, 3.6, 7.2, 1.8, 4.1, 2.2, 6.8, 2.0, 3.3, 4.6,
                5.2, 1.5, 6.1, 7.4, 2.6, 3.9, 4.8, 2.4, 5.7, 3.1]

for i, bus in enumerate(residential_buses, start=1):
    p_home_mw = home_base_kw[i - 1] / 1000.0
    q_home_mvar = pf_to_q_mvar(p_home_mw, pf=0.97)

    p_ev_mw = ev_kw[i - 1] / 1000.0
    q_ev_mvar = pf_to_q_mvar(p_ev_mw, pf=0.98)

    p_pv_mw = home_pv_kw[i - 1] / 1000.0

    pp.create_load(net, bus=bus, p_mw=p_home_mw, q_mvar=q_home_mvar, name=f"HOME_{i:02d}_LOAD")
    pp.create_load(net, bus=bus, p_mw=p_ev_mw,   q_mvar=q_ev_mvar,   name=f"HOME_{i:02d}_EV")
    pp.create_sgen(net, bus=bus, p_mw=p_pv_mw, q_mvar=0.0, name=f"HOME_{i:02d}_PV")

# -----------------------------
# industrial customers
# 5 customers, 50–100 kW each
# -----------------------------
industrial_kw = [60, 75, 90, 55, 100]
industrial_pf = [0.95, 0.96, 0.94, 0.95, 0.93]

for i, (p_kw, pf) in enumerate(zip(industrial_kw, industrial_pf), start=1):
    p_mw = p_kw / 1000.0
    q_mvar = pf_to_q_mvar(p_mw, pf=pf)
    pp.create_load(net, bus=b_ind, p_mw=p_mw, q_mvar=q_mvar, name=f"INDUSTRY_{i}")

# -----------------------------
# separate DER plant bus
# -----------------------------
pp.create_sgen(net, bus=b_der, p_mw=0.050, q_mvar=0.0, name="PV_PLANT_50kWp")
pp.create_sgen(net, bus=b_der, p_mw=0.100, q_mvar=0.0, name="WIND_PLANT_100kW")

# -----------------------------
# optional transformer LV auxiliary load
# -----------------------------
pp.create_load(net, bus=b_lv_main, p_mw=0.004, q_mvar=pf_to_q_mvar(0.004, pf=0.95), name="COMM_AUX_LOAD")

# -----------------------------
# run power flow
# -----------------------------
pp.runpp(net, algorithm="nr", init="auto", calculate_voltage_angles=False)

# -----------------------------
# optional protection result calculation
# (only if protection package is present)
# -----------------------------
protection_results = None
if protection_available:
    try:
        from pandapower.protection.run_protection import calculate_protection_times
        protection_results = calculate_protection_times(net, scenario="pp")
    except Exception:
        protection_results = None

# -----------------------------
# aggregate result metrics
# -----------------------------
total_load_p = net.load.p_mw.sum()
total_gen_p  = net.sgen.p_mw.sum()
trafo_loading = float(net.res_trafo.loading_percent.iloc[0])

# -----------------------------
# plotting
# -----------------------------
fig, ax = plt.subplots(figsize=(17, 10))

# base network drawing
plot.simple_plot(
    net,
    ax=ax,
    show_plot=False,
    respect_switches=True,
    plot_line_switches=True,
    line_width=2.4,
    bus_size=0.11,
    ext_grid_size=0.18,
    trafo_size=0.20,
    switch_size=0.13,
    bus_color="#0B5CAD",
    line_color="#7A7A7A",
    trafo_color="#111111",
    ext_grid_color="#D7263D",
    switch_color="#111111",
    scale_size=False
)

# make bus markers visually stronger
for coll in ax.collections:
    try:
        coll.set_zorder(2)
    except Exception:
        pass

# highlight key buses
highlight_colors = {
    b_hv: "#D7263D",
    b_lv_main: "#1B998B",
    b_ind: "#F4A261",
    b_der: "#2A9D8F"
}
for bus_idx, color in highlight_colors.items():
    x = net.bus_geodata.loc[bus_idx, "x"]
    y = net.bus_geodata.loc[bus_idx, "y"]
    ax.scatter([x], [y], s=180, color=color, edgecolors="black", linewidths=1.2, zorder=5)

# title + subtitle
subtitle = (
    f"20 homes + rooftop PV + EV | 5 industrial customers | 50 kWp PV plant | 100 kW wind plant | "
    f"Transformer loading: {trafo_loading:.1f}%"
)
ax.set_title("Local Low-Voltage Grid with DER and Protection Devices", fontsize=18, weight="bold", pad=16)
ax.text(
    0.5, 1.01, subtitle,
    transform=ax.transAxes, ha="center", va="bottom",
    fontsize=10.5, color="#333333"
)

# -----------------------------
# bus annotations
# voltage, bus power, approximate current
# -----------------------------
for bus in net.bus.index:
    x = net.bus_geodata.loc[bus, "x"]
    y = net.bus_geodata.loc[bus, "y"]

    name = net.bus.at[bus, "name"]
    vm_pu = float(net.res_bus.at[bus, "vm_pu"])
    p_mw = float(net.res_bus.at[bus, "p_mw"])
    q_mvar = float(net.res_bus.at[bus, "q_mvar"])
    i_ka = bus_result_current_ka(net.bus.at[bus, "vn_kv"], p_mw, q_mvar)

    # nicer sign convention for annotation
    p_kw = p_mw * 1000
    q_kvar = q_mvar * 1000

    if bus == b_hv:
        dx, dy = 0.00, 0.42
    elif bus == b_lv_main:
        dx, dy = 0.00, 0.40
    elif x < 0:
        dx, dy = -0.10, 0.28
    else:
        dx, dy = 0.10, 0.28

    text = (
        f"{name}\n"
        f"V = {vm_pu:.3f} pu\n"
        f"P = {p_kw:+.1f} kW\n"
        f"Q = {q_kvar:+.1f} kVAr\n"
        f"I ≈ {i_ka:.3f} kA"
    )

    box_fc = "white"
    if bus == b_ind:
        box_fc = "#FFF3E8"
    elif bus == b_der:
        box_fc = "#EAFBF5"
    elif bus == b_lv_main:
        box_fc = "#E8F7F4"
    elif bus == b_hv:
        box_fc = "#FCEBEC"

    ax.text(
        x + dx, y + dy, text,
        fontsize=8.4,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.28", fc=box_fc, ec="#888888", alpha=0.95),
        zorder=10
    )

# -----------------------------
# line annotations
# distance, current, loading
# -----------------------------
for line in net.line.index:
    fb = net.line.at[line, "from_bus"]
    tb = net.line.at[line, "to_bus"]

    x1, y1 = net.bus_geodata.loc[fb, ["x", "y"]]
    x2, y2 = net.bus_geodata.loc[tb, ["x", "y"]]

    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2

    length_km = float(net.line.at[line, "length_km"])
    i_ka = float(net.res_line.at[line, "i_ka"])
    loading = float(net.res_line.at[line, "loading_percent"])

    # slight offset so labels don't sit exactly on line
    dx = 0.14 if x2 >= x1 else -0.14
    dy = 0.08

    txt = (
        f"{length_km:.2f} km\n"
        f"I = {i_ka:.3f} kA\n"
        f"Load = {loading:.1f}%"
    )

    ax.text(
        xm + dx, ym + dy, txt,
        fontsize=7.6,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#B0B0B0", alpha=0.92),
        zorder=9
    )

# -----------------------------
# switch labels (CB / LBS / DS)
# -----------------------------
for sw in net.switch.index:
    bus = int(net.switch.at[sw, "bus"])
    et = net.switch.at[sw, "et"]
    sw_type = str(net.switch.at[sw, "type"])
    name = str(net.switch.at[sw, "name"])

    x_bus = net.bus_geodata.loc[bus, "x"]
    y_bus = net.bus_geodata.loc[bus, "y"]

    # estimate switch label position near associated element
    if et == "l":
        line = int(net.switch.at[sw, "element"])
        fb = int(net.line.at[line, "from_bus"])
        tb = int(net.line.at[line, "to_bus"])
        x1, y1 = net.bus_geodata.loc[fb, ["x", "y"]]
        x2, y2 = net.bus_geodata.loc[tb, ["x", "y"]]
        xs = x1 + 0.18 * (x2 - x1)
        ys = y1 + 0.18 * (y2 - y1)
    elif et == "t":
        xs = x_bus + 0.22
        ys = y_bus - 0.15
    else:
        xs = x_bus + 0.15
        ys = y_bus + 0.10

    if sw_type == "CB":
        fc = "#FFE9A8"
    elif sw_type == "LBS":
        fc = "#DDEBFF"
    elif sw_type == "DS":
        fc = "#F0F0F0"
    else:
        fc = "white"

    ax.text(
        xs, ys,
        sw_type,
        fontsize=7.3,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.16", fc=fc, ec="#666666", alpha=0.95),
        zorder=11
    )

# -----------------------------
# legend
# -----------------------------
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='HV Source Bus', markerfacecolor='#D7263D', markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='LV Main Bus', markerfacecolor='#1B998B', markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Industrial / DER Key Bus', markerfacecolor='#F4A261', markeredgecolor='black', markersize=10),
    Line2D([0], [0], color='#7A7A7A', lw=2.4, label='Power Line'),
    Patch(facecolor='#FFE9A8', edgecolor='#666666', label='CB = Circuit Breaker'),
    Patch(facecolor='#DDEBFF', edgecolor='#666666', label='LBS = Load Break Switch'),
    Patch(facecolor='#F0F0F0', edgecolor='#666666', label='DS = Disconnector'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, frameon=True)

# -----------------------------
# footer summary box
# -----------------------------
summary_lines = [
    f"Total connected load: {total_load_p*1000:.1f} kW",
    f"Total installed generation: {total_gen_p*1000:.1f} kW",
    f"Transformer loading: {trafo_loading:.1f} %",
    f"Protection objects attached: {'Yes' if protection_available else 'No (switches only)'}"
]
if protection_results is not None:
    summary_lines.append("Protection time calculation: available")
else:
    summary_lines.append("Protection time calculation: not evaluated")

summary_text = "\n".join(summary_lines)
ax.text(
    0.985, 0.02, summary_text,
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=9.2,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#666666", alpha=0.96)
)

# -----------------------------
# cosmetics
# -----------------------------
ax.set_aspect("equal")
ax.grid(False)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()

# Save for presentation use
plt.savefig("local_grid_presentation.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# optional console results
# -----------------------------
print("\n========================")
print("NETWORK SUMMARY")
print("========================")
print(f"Total load       : {total_load_p*1000:.2f} kW")
print(f"Total generation : {total_gen_p*1000:.2f} kW")
print(f"Transformer load : {trafo_loading:.2f} %")
print("\nBus results:")
print(net.res_bus[["vm_pu", "p_mw", "q_mvar"]])

print("\nLine results:")
print(net.res_line[["i_ka", "loading_percent"]])

if protection_results is not None:
    print("\nProtection results:")
    print(protection_results)
elif protection_available:
    print("\nProtection package available, but protection times were not calculated for this version/setup.")
else:
    print("\nProtection package not available in this pandapower installation.")
