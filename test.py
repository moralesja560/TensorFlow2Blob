import pyads

pyads.open_port()
ams_net_id = pyads.get_local_address().netid
print(ams_net_id)
pyads.close_port()
plc=pyads.Connection('10.65.96.185.1.1', 801, '10.65.96.185')
plc.open()
var_handle_actual_hook = plc.get_handle('SCADA.This_hook')
plc.write_by_name("", 1, plc_datatype=pyads.PLCTYPE_UINT,handle=var_handle_actual_hook)
plc.release_handle(var_handle_actual_hook)
plc.close()
