import valstore as vt
from config_ursse import get_from_config as get


gamma = get("gamma")
me = get("me_MeV")
ring_alpha = get("ring_alpha")



vt.newval("Vrf", get("Vrf"), '{:.0f}', "V")
vt.newval("gamma", gamma, '{:.1f}', "")
vt.newval("Ezero", gamma*me, '{:.1f}', "MeV")
vt.newval("alphac", ring_alpha, '{:.5f}', "")
vt.newval("etas", ring_alpha-1/gamma**2, '{:.5f}', "")
vt.newval("h", get("RF_q"), '{:.0f}', "")
vt.newval("rho", 100*get("dipole_rho_m"), '{:.0f}', "cm")

