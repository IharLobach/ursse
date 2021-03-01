import valstotex as vtt
from config_ursse import get_from_config as get


gamma = get("gamma")
me = get("me_MeV")
ring_alpha = get("ring_alpha")


vtt.newval("nav", f'{get("nav"):.1f}', "")
vtt.newval("Vrf", f'{get("Vrf"):.0f}', "V")
vtt.newval("gamma", f'{gamma:.1f}', "")
vtt.newval("Ezero", f'{gamma*me:.1f}', "MeV")
vtt.newval("JE", f'{get("damping_partition_JE"):.2f}', "")
vtt.newval("alphac", f'{ring_alpha:.5f}', "")
vtt.newval("etas", f'{ring_alpha-1/gamma**2:.5f}', "")
vtt.newval("phis", f'{get("phis"):.4f}', "rad")
vtt.newval("Uav", f'{get("Et"):.1f}', "eV")
vtt.newval("h", f'{get("RF_q"):.0f}', "")
vtt.newval("Ec", f'{get("Ec"):.1f}', "eV")
vtt.newval("rho", f'{100*get("dipole_rho_m"):.0f}', "cm")
vtt.newval("M", f'{get("M"):.2f}', "")

