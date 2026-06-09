import sympy as sp
import networkx as nx
import math
import random

def build_physics_graph():
    state = {
        'anode_thickness': 85e-6, 'cathode_thickness': 75e-6, 'separator_thickness': 20e-6,
        'electrode_gap': 1e-4, 'cell_width': 0.15, 'cell_length': 0.3, 'cell_height': 0.02,
        'tab_width': 0.04, 'tab_thickness': 0.002, 'jellyroll_radius': 0.015,
        'particle_size_anode': 12e-6, 'particle_size_cathode': 8e-6, 'binder_ratio': 0.04,
        'conductive_additive_ratio': 0.04, 'separator_porosity': 0.4, 'electrolyte_concentration': 1.15,
        'pre_lithiation_degree': 0.08, 'packing_density': 0.72, 'solid_electrolyte_fraction': 0.02,
        'lithium_excess': 0.05, 'coolant_flow': 0.06, 'chiller_temp_setpoint': 295.15,
        'tim_thickness': 0.0008, 'tim_thermal_conductivity': 3.5, 'base_plate_thickness': 0.012,
        'fin_thickness': 0.0025, 'fin_height': 0.045, 'fin_pitch': 0.006,
        'cooling_channel_width': 0.003, 'cooling_channel_depth': 0.006, 'pump_speed': 3200.0,
        'fan_speed': 4800.0, 'compressor_power': 1600.0, 'coolant_viscosity_ref': 0.001,
        'coolant_density_ref': 1050.0, 'fan_blade_radius': 0.15, 'fan_blade_angle': 0.25,
        'pump_impeller_diameter': 0.05, 'nozzle_diameter': 0.01, 'pipe_roughness': 1e-4,
        'wire_radius': 0.006, 'busbar_cross_section': 0.00015, 'busbar_length': 0.8,
        'inverter_switching_freq': 25000.0, 'gate_resistance': 5.0, 'mosfet_rdson': 0.002,
        'dc_link_cap': 500e-6, 'inductor_volume': 0.002, 'winding_turns': 20.0,
        'core_cross_section': 0.001, 'housing_thickness': 0.005, 'housing_surface_area': 1.5,
        'casing_emissivity': 0.85, 'vibration_mount_stiffness': 50000.0, 'vibration_mount_damping': 200.0,
        'fastener_torque': 15.0, 'weld_seam_length': 2.0, 'potting_compound_thickness': 0.003,
        'gasket_thickness': 0.002, 'gasket_compression': 0.2, 'calendering_pressure': 45e6,
        'formation_current': 12.0, 'drying_temperature': 350.15, 'drying_time': 7200.0,
        'coating_speed': 1.5, 'welding_power': 2000.0, 'assembly_pressure': 10000.0,
        'aging_temperature': 310.15, 'aging_time': 86400.0, 'inspection_rate': 0.95,
        'soc_target': 0.8, 'dod_max': 0.9, 'c_rate_max': 3.0, 'voltage_max': 4.2,
        'voltage_min': 2.8, 'temp_limit_high': 333.15, 'temp_limit_low': 253.15,
        'power_limit_surge': 150000.0, 'current_limit_continuous': 180.0, 'derating_factor': 0.85,
        'cell_spacing': 0.002, 'pack_volume_margin': 0.15, 'bms_sampling_rate': 100.0,
        'sensor_accuracy': 0.98, 'isolation_resistance': 1e6, 'ambient_temp': 298.15
    }

    C_voltage_sys = 400.0
    C_solar_irradiation = 800.0

    s = state
    s['die_area'] = s['cell_width'] * s['cell_length']
    s['cell_volume'] = s['cell_width'] * s['cell_length'] * s['cell_height']
    s['pack_volume'] = s['cell_volume'] * (1.0 + s['pack_volume_margin']) * 1.2
    s['active_material_vol'] = s['cell_volume'] * s['packing_density'] * (
                1.0 - s['binder_ratio'] - s['conductive_additive_ratio'])
    s['anode_active_area'] = s['active_material_vol'] * s['anode_thickness'] / (s['particle_size_anode'] + 1e-6)
    s['cathode_active_area'] = s['active_material_vol'] * s['cathode_thickness'] / (s['particle_size_cathode'] + 1e-6)
    s['fin_surface_area'] = 2.0 * s['fin_height'] * s['fin_thickness'] / s['fin_pitch'] * s['die_area']
    s['thermal_pad_area'] = s['die_area'] * 1.05
    s['cross_section_flow'] = s['cooling_channel_width'] * s['cooling_channel_depth'] * 10.0
    s['hydraulic_diameter'] = 2.0 * s['cooling_channel_width'] * s['cooling_channel_depth'] / (
                s['cooling_channel_width'] + s['cooling_channel_depth'] + 1e-6)

    s['coolant_velocity'] = s['coolant_flow'] / (s['cross_section_flow'] + 1e-6)
    s['reynolds_number'] = s['coolant_density_ref'] * s['coolant_velocity'] * s['hydraulic_diameter'] / (
                s['coolant_viscosity_ref'] + 1e-6)
    s['nusselt_number'] = 0.023 * s['reynolds_number'] ** 0.8 * 4.0 ** 0.4
    s['convective_coeff'] = s['nusselt_number'] * 0.6 / (s['hydraulic_diameter'] + 1e-6) + s['pump_speed'] * 0.01
    s['pressure_drop'] = s['pipe_roughness'] * (s['coolant_velocity'] ** 2) * 1000.0 / (s['hydraulic_diameter'] + 1e-6)
    s['pump_power'] = s['pressure_drop'] * s['coolant_flow'] / 0.8 + 0.5 * s['pump_speed'] ** 3 * 1e-9
    s['fan_power'] = 0.5 * s['fan_speed'] ** 2.5 * s['fan_blade_radius'] ** 2 * 1e-6
    s['thermal_resistance_sink'] = 1.0 / (s['convective_coeff'] * s['fin_surface_area'] + 1e-6)
    s['thermal_resistance_tim'] = s['tim_thickness'] / (s['tim_thermal_conductivity'] * s['thermal_pad_area'] + 1e-6)
    s['total_thermal_resistance'] = s['thermal_resistance_sink'] + s['thermal_resistance_tim'] + s[
        'base_plate_thickness'] / (200.0 * s['die_area'] + 1e-6)

    s['wire_resistance'] = 1.68e-8 * 2.5 / (math.pi * s['wire_radius'] ** 2)
    s['busbar_resistance'] = 1.68e-8 * s['busbar_length'] / (s['busbar_cross_section'] + 1e-6)
    s['tab_resistance'] = 2.82e-8 * s['tab_thickness'] / (s['tab_width'] * s['jellyroll_radius'] + 1e-6)
    s['electrolyte_conductivity'] = 1.5 * s['electrolyte_concentration'] * math.exp(
        -2500.0 / (8.314 * s['chiller_temp_setpoint'] + 1e-6))
    s['separator_resistance'] = s['separator_thickness'] / (
                s['electrolyte_conductivity'] * s['separator_porosity'] ** 1.5 + 1e-9)
    s['anode_resistance'] = s['anode_thickness'] / (
                1e4 * s['anode_active_area'] * s['calendering_pressure'] ** 0.5 + 1e-9)
    s['cathode_resistance'] = s['cathode_thickness'] / (
                5e3 * s['cathode_active_area'] * s['calendering_pressure'] ** 0.4 + 1e-9)
    s['cell_internal_resistance'] = s['separator_resistance'] + s['anode_resistance'] + s['cathode_resistance'] + s[
        'tab_resistance'] + s['electrode_gap'] / (s['electrolyte_conductivity'] + 1e-9)
    s['inverter_resistance'] = s['mosfet_rdson'] * 6.0 + 0.001
    s['total_electrical_resistance'] = s['wire_resistance'] + s['busbar_resistance'] + s['cell_internal_resistance'] + \
                                       s['inverter_resistance']

    s['operating_current'] = s['current_limit_continuous'] * s['derating_factor']
    s['joule_heating_wire'] = s['operating_current'] ** 2 * s['wire_resistance']
    s['busbar_loss'] = s['operating_current'] ** 2 * s['busbar_resistance']
    s['cell_joule_heating'] = s['operating_current'] ** 2 * s['cell_internal_resistance']
    s['switching_loss'] = 0.5 * C_voltage_sys * s['operating_current'] * s['inverter_switching_freq'] * s[
        'gate_resistance'] * 1e-6
    s['core_loss'] = 1.2 * s['inductor_volume'] * (s['inverter_switching_freq'] / 1000.0) ** 1.5 * s[
        'core_cross_section'] ** 2
    s['total_heat_generation'] = s['joule_heating_wire'] + s['busbar_loss'] + s['cell_joule_heating'] + s[
        'switching_loss'] + s['core_loss']
    s['die_temp'] = s['chiller_temp_setpoint'] + s['total_heat_generation'] * s['total_thermal_resistance']
    s['radiation_heat_loss'] = 5.67e-8 * s['casing_emissivity'] * s['housing_surface_area'] * (
                s['die_temp'] ** 4 - 298.15 ** 4)
    s['housing_temp'] = 298.15 + (s['die_temp'] - 298.15) * math.exp(-s['housing_thickness'] / 0.05) - s[
        'radiation_heat_loss'] / 100.0

    s['leakage_current'] = s['operating_current'] * 0.01 * math.exp(min((s['die_temp'] - 298.15) / 25.0, 100.0)) if s[
                                                                                                                        'die_temp'] > 298.15 else \
    s['operating_current'] * 0.01
    s['cell_capacity'] = 3600.0 * (
                s['anode_thickness'] * s['anode_active_area'] + s['cathode_thickness'] * s['cathode_active_area']) * s[
                             'packing_density'] * s['soc_target']
    s['cell_voltage_drop'] = s['operating_current'] * s['total_electrical_resistance']
    s['operating_voltage'] = max(s['voltage_min'], min(s['voltage_max'], C_voltage_sys - s['cell_voltage_drop']))
    s['actual_power_out'] = s['operating_voltage'] * s['operating_current']
    s['cell_energy'] = s['cell_capacity'] * s['operating_voltage']
    s['sei_layer_growth'] = s['formation_current'] ** 0.5 * math.exp(-1000.0 / (8.314 * s['die_temp'] + 1e-6)) * s[
        'aging_time'] ** 0.2
    s['capacity_fade_rate'] = s['sei_layer_growth'] * (s['pre_lithiation_degree'] + 0.01) ** -0.2 * math.log(
        1.0 + s['aging_temperature'])
    s['cycle_life'] = 800.0 / (s['capacity_fade_rate'] + 1e-6) * (1.1 - s['dod_max'])
    s['efficiency_power'] = s['actual_power_out'] / (
                s['actual_power_out'] + s['total_heat_generation'] + s['pump_power'] + s['fan_power'] + s[
            'compressor_power'] + 1e-6)

    s['system_weight_cells'] = s['pack_volume'] * s['packing_density'] * 2500.0
    s['system_weight_housing'] = s['housing_surface_area'] * s['housing_thickness'] * 2700.0 + s[
        'weld_seam_length'] * 0.1
    s['system_weight_coolant'] = s['pack_volume'] * s['coolant_density_ref'] * 0.1
    s['system_weight'] = s['system_weight_cells'] + s['system_weight_housing'] + s['system_weight_coolant'] + s[
        'busbar_length'] * s['busbar_cross_section'] * 8900.0
    s['specific_energy'] = s['cell_energy'] / (s['system_weight'] + 1e-6)
    s['specific_power'] = s['actual_power_out'] / (s['system_weight'] + 1e-6)
    s['energy_density'] = s['cell_energy'] / (s['pack_volume'] + 1e-6)
    s['power_density'] = s['actual_power_out'] / (s['pack_volume'] + 1e-6)
    s['stress_thermal'] = 12e-6 * 70e9 * max(1.0, s['die_temp'] - 298.15) * math.log(
        1.0 + s['base_plate_thickness'] / 0.01)
    s['volumetric_expansion'] = 0.05 * s['pre_lithiation_degree'] * math.log(1.0 + s['cell_capacity'] / 1000.0) + s[
        'thermal_pad_area'] * 0.001

    s['mechanical_stress_expansion'] = s['volumetric_expansion'] * 50e9 * s['housing_thickness'] / (
                s['die_area'] ** 0.5 + 1e-6)
    s['vibration_tolerance'] = 100.0 * s['housing_thickness'] ** 2 * s['vibration_mount_stiffness'] / (
                s['housing_surface_area'] * s['system_weight'] ** 0.5 * s['vibration_mount_damping'] + 1e-6)
    s['fatigue_life'] = 1e12 * (s['stress_thermal'] + s['mechanical_stress_expansion'] + 1e6) ** -3 * math.exp(
        min(298.15 / (s['die_temp'] + 1e-6), 100.0))
    s['short_circuit_risk'] = 1e-3 * math.exp(min((s['die_temp'] - 298.15) / 50.0, 100.0)) * s['separator_porosity'] / (
                s['separator_thickness'] * 1e6 + 1e-6)
    s['leakage_risk'] = 1e-4 * 101325.0 * s['housing_surface_area'] / (
                s['housing_thickness'] * s['gasket_compression'] + 1e-6)
    s['emi_shielding'] = 50.0 * math.log(1.0 + s['housing_thickness'] * 1000.0) * s['conductive_additive_ratio'] ** 0.5
    s['manufacturing_defect_rate'] = max(0.0,
                                         1.0 - math.exp(-s['assembly_pressure'] / 1e6 - s['coating_speed'] / 10.0) * s[
                                             'inspection_rate'])
    s['material_cost'] = 50.0 * s['die_area'] + 200.0 * (s['anode_thickness'] + s['cathode_thickness']) + 15.0 * s[
        'separator_thickness'] / (s['separator_porosity'] + 1e-6) + s['system_weight'] * 2.0
    s['manufacturing_cost'] = 1e8 / (s['calendering_pressure'] + 1.0) + 500.0 * s['formation_current'] ** 0.2 + s[
        'welding_power'] * 0.01 + s['drying_temperature'] * s['drying_time'] * 0.1
    s['cooling_cost'] = 5.0 * s['pump_power'] ** 0.6 + 2.0 * s['fan_power'] ** 0.7 + 10.0 * s['compressor_power'] ** 0.8

    s['recyclability_index'] = max(0.0, 100.0 - 10.0 * s['binder_ratio'] / (
                s['binder_ratio'] + s['conductive_additive_ratio'] + 1e-6) - s['potting_compound_thickness'] * 100.0)
    s['carbon_footprint'] = 0.5 * s['material_cost'] + 0.2 * s['manufacturing_cost'] + 0.1 * s['cooling_cost']
    s['startup_time'] = 1500.0 * s['cell_internal_resistance'] * s['cell_capacity'] / (s['actual_power_out'] + 1e-6) + \
                        s['sensor_accuracy'] * 10.0
    s['dynamic_response'] = s['power_density'] * s['vibration_tolerance'] / (s['startup_time'] + 1e-6) * s[
        'bms_sampling_rate'] ** 0.5
    s['aging_acceleration'] = math.exp(min((s['die_temp'] - 298.15) / 15.0, 100.0)) * (
                s['cell_voltage_drop'] / C_voltage_sys) ** 2
    s['thermal_gradient'] = max(0.1, (s['die_temp'] - s['housing_temp'])) / s['base_plate_thickness']
    s['cooling_efficiency'] = (s['convective_coeff'] * s['fin_surface_area']) / (
                s['pump_power'] + s['fan_power'] + s['compressor_power'] + 1e-6)

    s['solar_heat_gain'] = C_solar_irradiation * s['housing_surface_area'] * 0.8
    s['ambient_heat_transfer'] = (s['housing_temp'] - s['ambient_temp']) * s['housing_surface_area'] * 10.0
    s['total_housing_heat_flux'] = s['solar_heat_gain'] - s['ambient_heat_transfer'] - s['radiation_heat_loss']

    s['Sys_Risk'] = s['short_circuit_risk'] * 100.0 + s['leakage_risk'] * 50.0 + s['aging_acceleration'] * 200.0 + s[
        'mechanical_stress_expansion'] / 1e7 + s['stress_thermal'] / 1e8 + 1000.0 / (
                                s['fatigue_life'] + 1e-6) + 10.0 / (s['emi_shielding'] + 1e-6) + s['die_temp'] / 100.0 + \
                    s['housing_temp'] / 100.0 + s['cell_voltage_drop'] + s['joule_heating_wire'] / 100.0 + s[
                        'leakage_current'] * 10.0 + s['thermal_gradient'] / 1000.0 + s[
                        'manufacturing_defect_rate'] * 500.0

    s['Total_Cost'] = s['cooling_cost'] + s['material_cost'] + s['manufacturing_cost'] + s['carbon_footprint'] * 15.0 + \
                      s['Sys_Risk'] * 50.0 + 1e5 / (s['cycle_life'] + 1e-6) + s['system_weight'] * 5.0 + s[
                          'pump_power'] + s['fan_power'] + s['compressor_power'] / 10.0 + s[
                          'total_electrical_resistance'] * 1e6

    reg_term_vars = [k for k in state.keys() if type(state[k]) in [float, int] and 'sys' not in k.lower()]
    reg_term = sum([math.log(max(1e-9, abs(s[v]))) for v in reg_term_vars])

    s['Global_FoM'] = (s['specific_energy'] ** 0.5 * s['specific_power'] ** 0.5 * s['cycle_life'] ** 0.8 * s[
        'efficiency_power'] * s['dynamic_response'] * s['recyclability_index'] * s['energy_density'] ** 0.2 * s[
                           'power_density'] ** 0.2 * s['cooling_efficiency'] ** 0.1) / (
                                  s['Total_Cost'] ** 0.5 * s['Sys_Risk'] ** 1.2 * s['startup_time'] ** 0.3 * s[
                              'cell_internal_resistance'] ** 0.2 * s[
                                      'total_thermal_resistance'] ** 0.1 + 1e-6) * math.exp(
        max(-100.0, -0.01 * reg_term))

    sy = {k: sp.Symbol(k) for k in state.keys()}
    equations = []

    equations.append(('die_area', sy['cell_width'] * sy['cell_length']))
    equations.append(('cell_volume', sy['cell_width'] * sy['cell_length'] * sy['cell_height']))
    equations.append(('pack_volume', sy['cell_volume'] * (1.0 + sy['pack_volume_margin']) * 1.2))
    equations.append(('active_material_vol', sy['cell_volume'] * sy['packing_density'] * (
                1.0 - sy['binder_ratio'] - sy['conductive_additive_ratio'])))
    equations.append(
        ('anode_active_area', sy['active_material_vol'] * sy['anode_thickness'] / (sy['particle_size_anode'] + 1e-6)))
    equations.append(('cathode_active_area',
                      sy['active_material_vol'] * sy['cathode_thickness'] / (sy['particle_size_cathode'] + 1e-6)))
    equations.append(
        ('fin_surface_area', 2.0 * sy['fin_height'] * sy['fin_thickness'] / sy['fin_pitch'] * sy['die_area']))
    equations.append(('thermal_pad_area', sy['die_area'] * 1.05))
    equations.append(('cross_section_flow', sy['cooling_channel_width'] * sy['cooling_channel_depth'] * 10.0))
    equations.append(('hydraulic_diameter', 2.0 * sy['cooling_channel_width'] * sy['cooling_channel_depth'] / (
                sy['cooling_channel_width'] + sy['cooling_channel_depth'] + 1e-6)))

    equations.append(('coolant_velocity', sy['coolant_flow'] / (sy['cross_section_flow'] + 1e-6)))
    equations.append(('reynolds_number',
                      sy['coolant_density_ref'] * sy['coolant_velocity'] * sy['hydraulic_diameter'] / (
                                  sy['coolant_viscosity_ref'] + 1e-6)))
    equations.append(('nusselt_number', 0.023 * sy['reynolds_number'] ** 0.8 * 4.0 ** 0.4))
    equations.append(
        ('convective_coeff', sy['nusselt_number'] * 0.6 / (sy['hydraulic_diameter'] + 1e-6) + sy['pump_speed'] * 0.01))
    equations.append(('pressure_drop', sy['pipe_roughness'] * (sy['coolant_velocity'] ** 2) * 1000.0 / (
                sy['hydraulic_diameter'] + 1e-6)))
    equations.append(
        ('pump_power', sy['pressure_drop'] * sy['coolant_flow'] / 0.8 + 0.5 * sy['pump_speed'] ** 3 * 1e-9))
    equations.append(('fan_power', 0.5 * sy['fan_speed'] ** 2.5 * sy['fan_blade_radius'] ** 2 * 1e-6))
    equations.append(('thermal_resistance_sink', 1.0 / (sy['convective_coeff'] * sy['fin_surface_area'] + 1e-6)))
    equations.append(('thermal_resistance_tim',
                      sy['tim_thickness'] / (sy['tim_thermal_conductivity'] * sy['thermal_pad_area'] + 1e-6)))
    equations.append(('total_thermal_resistance',
                      sy['thermal_resistance_sink'] + sy['thermal_resistance_tim'] + sy['base_plate_thickness'] / (
                                  200.0 * sy['die_area'] + 1e-6)))

    equations.append(('wire_resistance', 1.68e-8 * 2.5 / (sp.pi * sy['wire_radius'] ** 2)))
    equations.append(('busbar_resistance', 1.68e-8 * sy['busbar_length'] / (sy['busbar_cross_section'] + 1e-6)))
    equations.append(
        ('tab_resistance', 2.82e-8 * sy['tab_thickness'] / (sy['tab_width'] * sy['jellyroll_radius'] + 1e-6)))
    equations.append(('electrolyte_conductivity', 1.5 * sy['electrolyte_concentration'] * sp.exp(
        -2500.0 / (8.314 * sy['chiller_temp_setpoint'] + 1e-6))))
    equations.append(('separator_resistance', sy['separator_thickness'] / (
                sy['electrolyte_conductivity'] * sy['separator_porosity'] ** 1.5 + 1e-9)))
    equations.append(('anode_resistance', sy['anode_thickness'] / (
                1e4 * sy['anode_active_area'] * sy['calendering_pressure'] ** 0.5 + 1e-9)))
    equations.append(('cathode_resistance', sy['cathode_thickness'] / (
                5e3 * sy['cathode_active_area'] * sy['calendering_pressure'] ** 0.4 + 1e-9)))
    equations.append(('cell_internal_resistance',
                      sy['separator_resistance'] + sy['anode_resistance'] + sy['cathode_resistance'] + sy[
                          'tab_resistance'] + sy['electrode_gap'] / (sy['electrolyte_conductivity'] + 1e-9)))
    equations.append(('inverter_resistance', sy['mosfet_rdson'] * 6.0 + 0.001))
    equations.append(('total_electrical_resistance',
                      sy['wire_resistance'] + sy['busbar_resistance'] + sy['cell_internal_resistance'] + sy[
                          'inverter_resistance']))

    equations.append(('operating_current', sy['current_limit_continuous'] * sy['derating_factor']))
    equations.append(('joule_heating_wire', sy['operating_current'] ** 2 * sy['wire_resistance']))
    equations.append(('busbar_loss', sy['operating_current'] ** 2 * sy['busbar_resistance']))
    equations.append(('cell_joule_heating', sy['operating_current'] ** 2 * sy['cell_internal_resistance']))
    equations.append(('switching_loss',
                      0.5 * C_voltage_sys * sy['operating_current'] * sy['inverter_switching_freq'] * sy[
                          'gate_resistance'] * 1e-6))
    equations.append(('core_loss', 1.2 * sy['inductor_volume'] * (sy['inverter_switching_freq'] / 1000.0) ** 1.5 * sy[
        'core_cross_section'] ** 2))
    equations.append(('total_heat_generation',
                      sy['joule_heating_wire'] + sy['busbar_loss'] + sy['cell_joule_heating'] + sy['switching_loss'] +
                      sy['core_loss']))
    equations.append(
        ('die_temp', sy['chiller_temp_setpoint'] + sy['total_heat_generation'] * sy['total_thermal_resistance']))
    equations.append(('radiation_heat_loss', 5.67e-8 * sy['casing_emissivity'] * sy['housing_surface_area'] * (
                sy['die_temp'] ** 4 - 298.15 ** 4)))
    equations.append(('housing_temp', 298.15 + (sy['die_temp'] - 298.15) * sp.exp(-sy['housing_thickness'] / 0.05) - sy[
        'radiation_heat_loss'] / 100.0))

    equations.append(('leakage_current', sp.Piecewise(
        (sy['operating_current'] * 0.01 * sp.exp(sp.Min((sy['die_temp'] - 298.15) / 25.0, 100.0)),
         sy['die_temp'] > 298.15), (sy['operating_current'] * 0.01, True))))
    equations.append(('cell_capacity', 3600.0 * (
                sy['anode_thickness'] * sy['anode_active_area'] + sy['cathode_thickness'] * sy['cathode_active_area']) *
                      sy['packing_density'] * sy['soc_target']))
    equations.append(('cell_voltage_drop', sy['operating_current'] * sy['total_electrical_resistance']))
    equations.append(('operating_voltage',
                      sp.Max(sy['voltage_min'], sp.Min(sy['voltage_max'], C_voltage_sys - sy['cell_voltage_drop']))))
    equations.append(('actual_power_out', sy['operating_voltage'] * sy['operating_current']))
    equations.append(('cell_energy', sy['cell_capacity'] * sy['operating_voltage']))
    equations.append(('sei_layer_growth',
                      sy['formation_current'] ** 0.5 * sp.exp(-1000.0 / (8.314 * sy['die_temp'] + 1e-6)) * sy[
                          'aging_time'] ** 0.2))
    equations.append(('capacity_fade_rate',
                      sy['sei_layer_growth'] * (sy['pre_lithiation_degree'] + 0.01) ** -0.2 * sp.log(
                          1.0 + sy['aging_temperature'])))
    equations.append(('cycle_life', 800.0 / (sy['capacity_fade_rate'] + 1e-6) * (1.1 - sy['dod_max'])))
    equations.append(('efficiency_power', sy['actual_power_out'] / (
                sy['actual_power_out'] + sy['total_heat_generation'] + sy['pump_power'] + sy['fan_power'] + sy[
            'compressor_power'] + 1e-6)))

    equations.append(('system_weight_cells', sy['pack_volume'] * sy['packing_density'] * 2500.0))
    equations.append(('system_weight_housing',
                      sy['housing_surface_area'] * sy['housing_thickness'] * 2700.0 + sy['weld_seam_length'] * 0.1))
    equations.append(('system_weight_coolant', sy['pack_volume'] * sy['coolant_density_ref'] * 0.1))
    equations.append(('system_weight',
                      sy['system_weight_cells'] + sy['system_weight_housing'] + sy['system_weight_coolant'] + sy[
                          'busbar_length'] * sy['busbar_cross_section'] * 8900.0))
    equations.append(('specific_energy', sy['cell_energy'] / (sy['system_weight'] + 1e-6)))
    equations.append(('specific_power', sy['actual_power_out'] / (sy['system_weight'] + 1e-6)))
    equations.append(('energy_density', sy['cell_energy'] / (sy['pack_volume'] + 1e-6)))
    equations.append(('power_density', sy['actual_power_out'] / (sy['pack_volume'] + 1e-6)))
    equations.append(('stress_thermal', 12e-6 * 70e9 * sp.Max(1.0, sy['die_temp'] - 298.15) * sp.log(
        1.0 + sy['base_plate_thickness'] / 0.01)))
    equations.append(('volumetric_expansion',
                      0.05 * sy['pre_lithiation_degree'] * sp.log(1.0 + sy['cell_capacity'] / 1000.0) + sy[
                          'thermal_pad_area'] * 0.001))

    equations.append(('mechanical_stress_expansion',
                      sy['volumetric_expansion'] * 50e9 * sy['housing_thickness'] / (sy['die_area'] ** 0.5 + 1e-6)))
    equations.append(('vibration_tolerance', 100.0 * sy['housing_thickness'] ** 2 * sy['vibration_mount_stiffness'] / (
                sy['housing_surface_area'] * sy['system_weight'] ** 0.5 * sy['vibration_mount_damping'] + 1e-6)))
    equations.append(('fatigue_life',
                      1e12 * (sy['stress_thermal'] + sy['mechanical_stress_expansion'] + 1e6) ** -3 * sp.exp(
                          sp.Min(298.15 / (sy['die_temp'] + 1e-6), 100.0))))
    equations.append(('short_circuit_risk',
                      1e-3 * sp.exp(sp.Min((sy['die_temp'] - 298.15) / 50.0, 100.0)) * sy['separator_porosity'] / (
                                  sy['separator_thickness'] * 1e6 + 1e-6)))
    equations.append(('leakage_risk', 1e-4 * 101325.0 * sy['housing_surface_area'] / (
                sy['housing_thickness'] * sy['gasket_compression'] + 1e-6)))
    equations.append(('emi_shielding',
                      50.0 * sp.log(1.0 + sy['housing_thickness'] * 1000.0) * sy['conductive_additive_ratio'] ** 0.5))
    equations.append(('manufacturing_defect_rate', sp.Max(0.0, 1.0 - sp.exp(
        -sy['assembly_pressure'] / 1e6 - sy['coating_speed'] / 10.0) * sy['inspection_rate'])))
    equations.append(('material_cost',
                      50.0 * sy['die_area'] + 200.0 * (sy['anode_thickness'] + sy['cathode_thickness']) + 15.0 * sy[
                          'separator_thickness'] / (sy['separator_porosity'] + 1e-6) + sy['system_weight'] * 2.0))
    equations.append(('manufacturing_cost',
                      1e8 / (sy['calendering_pressure'] + 1.0) + 500.0 * sy['formation_current'] ** 0.2 + sy[
                          'welding_power'] * 0.01 + sy['drying_temperature'] * sy['drying_time'] * 0.1))
    equations.append(('cooling_cost', 5.0 * sy['pump_power'] ** 0.6 + 2.0 * sy['fan_power'] ** 0.7 + 10.0 * sy[
        'compressor_power'] ** 0.8))

    equations.append(('recyclability_index', sp.Max(0.0, 100.0 - 10.0 * sy['binder_ratio'] / (
                sy['binder_ratio'] + sy['conductive_additive_ratio'] + 1e-6) - sy[
                                                        'potting_compound_thickness'] * 100.0)))
    equations.append(
        ('carbon_footprint', 0.5 * sy['material_cost'] + 0.2 * sy['manufacturing_cost'] + 0.1 * sy['cooling_cost']))
    equations.append(('startup_time',
                      1500.0 * sy['cell_internal_resistance'] * sy['cell_capacity'] / (sy['actual_power_out'] + 1e-6) +
                      sy['sensor_accuracy'] * 10.0))
    equations.append(('dynamic_response',
                      sy['power_density'] * sy['vibration_tolerance'] / (sy['startup_time'] + 1e-6) * sy[
                          'bms_sampling_rate'] ** 0.5))
    equations.append(('aging_acceleration', sp.exp(sp.Min((sy['die_temp'] - 298.15) / 15.0, 100.0)) * (
                sy['cell_voltage_drop'] / C_voltage_sys) ** 2))
    equations.append(
        ('thermal_gradient', sp.Max(0.1, (sy['die_temp'] - sy['housing_temp'])) / sy['base_plate_thickness']))
    equations.append(('cooling_efficiency', (sy['convective_coeff'] * sy['fin_surface_area']) / (
                sy['pump_power'] + sy['fan_power'] + sy['compressor_power'] + 1e-6)))

    equations.append(('solar_heat_gain', C_solar_irradiation * sy['housing_surface_area'] * 0.8))
    equations.append(
        ('ambient_heat_transfer', (sy['housing_temp'] - sy['ambient_temp']) * sy['housing_surface_area'] * 10.0))
    equations.append(
        ('total_housing_heat_flux', sy['solar_heat_gain'] - sy['ambient_heat_transfer'] - sy['radiation_heat_loss']))

    equations.append(('Sys_Risk',
                      sy['short_circuit_risk'] * 100.0 + sy['leakage_risk'] * 50.0 + sy['aging_acceleration'] * 200.0 +
                      sy['mechanical_stress_expansion'] / 1e7 + sy['stress_thermal'] / 1e8 + 1000.0 / (
                                  sy['fatigue_life'] + 1e-6) + 10.0 / (sy['emi_shielding'] + 1e-6) + sy[
                          'die_temp'] / 100.0 + sy['housing_temp'] / 100.0 + sy['cell_voltage_drop'] + sy[
                          'joule_heating_wire'] / 100.0 + sy['leakage_current'] * 10.0 + sy[
                          'thermal_gradient'] / 1000.0 + sy['manufacturing_defect_rate'] * 500.0))
    equations.append(('Total_Cost', sy['cooling_cost'] + sy['material_cost'] + sy['manufacturing_cost'] + sy[
        'carbon_footprint'] * 15.0 + sy['Sys_Risk'] * 50.0 + 1e5 / (sy['cycle_life'] + 1e-6) + sy[
                          'system_weight'] * 5.0 + sy['pump_power'] + sy['fan_power'] + sy['compressor_power'] / 10.0 +
                      sy['total_electrical_resistance'] * 1e6))

    sp_reg_term = sum([sp.log(sp.Max(1e-9, sp.Abs(sy[v]))) for v in reg_term_vars])
    equations.append(('Global_FoM', (
                sy['specific_energy'] ** 0.5 * sy['specific_power'] ** 0.5 * sy['cycle_life'] ** 0.8 * sy[
            'efficiency_power'] * sy['dynamic_response'] * sy['recyclability_index'] * sy['energy_density'] ** 0.2 * sy[
                    'power_density'] ** 0.2 * sy['cooling_efficiency'] ** 0.1) / (
                                  sy['Total_Cost'] ** 0.5 * sy['Sys_Risk'] ** 1.2 * sy['startup_time'] ** 0.3 * sy[
                              'cell_internal_resistance'] ** 0.2 * sy[
                                      'total_thermal_resistance'] ** 0.1 + 1e-6) * sp.exp(
        sp.Max(-100.0, -0.01 * sp_reg_term))))

    G = nx.DiGraph()
    for k in state.keys():
        val = state[k]

        if val == 0.0:
            low, high = 1e-6, 0.1
        else:
            low, high = val * 0.85, val * 1.15

        if any(key in k for key in
               ['ratio', 'porosity', 'degree', 'emissivity', 'fraction', 'excess', 'rate', 'factor', 'accuracy']):
            low = max(0.01, low)
            high = min(0.99, high)
        elif 'packing_density' in k:
            low = max(0.1, low)
            high = min(1.0, high)
        elif any(key in k for key in
                 ['thickness', 'area', 'radius', 'gap', 'size', 'width', 'depth', 'cross_section', 'length', 'volume',
                  'pitch']):
            low = max(1e-7, low)
        elif 'temp' in k:
            low = max(253.15, low)

        G.add_node(k, value=val, lower_bound=low, upper_bound=high)

    raw_edges = []

    for var_name, expr in equations:
        free_syms = list(expr.free_symbols)
        if not free_syms:
            continue

        target_val = state[var_name] if abs(state[var_name]) > 1e-12 else 1e-12

        for source_sym in free_syms:
            source_name = source_sym.name
            source_val = state[source_name]

            deriv_expr = sp.diff(expr, source_sym)

            try:
                fast_func = sp.lambdify(free_syms, deriv_expr, modules=['numpy', 'math', 'sympy'])
                args = [state[sym.name] for sym in free_syms]
                deriv_val = float(fast_func(*args))
            except Exception:
                deriv_val = 0.0

            if math.isnan(deriv_val) or math.isinf(deriv_val):
                deriv_val = 0.0

            elasticity = deriv_val * (source_val / target_val)

            if math.isnan(elasticity) or math.isinf(elasticity):
                continue

            if abs(elasticity) > 1e-9:
                elasticity = max(min(elasticity, 1e15), -1e15)
                raw_edges.append((source_name, var_name, elasticity))

    max_abs_e = max([abs(e[2]) for e in raw_edges]) if raw_edges else 1.0
    if math.isnan(max_abs_e) or math.isinf(max_abs_e) or max_abs_e <= 0:
        max_abs_e = 1.0

    max_log = math.log10(1.0 + max_abs_e)
    if max_log <= 0:
        max_log = 1.0

    for src, dst, e_val in raw_edges:
        sign = 1.0 if e_val >= 0 else -1.0
        norm_w = sign * (math.log10(1.0 + abs(e_val)) / max_log) * 10.0
        if not math.isnan(norm_w) and not math.isinf(norm_w):
            G.add_edge(src, dst, weight=norm_w)

    all_edges = list(G.edges(data=True))
    if all_edges:
        num_edges_to_perturb = int(len(all_edges) * 0.5)
        edges_to_perturb = random.sample(all_edges, num_edges_to_perturb)

        for u, v, data in edges_to_perturb:
            original_weight = data['weight']
            fluctuation_ratio = random.uniform(-0.10, 0.10)
            G[u][v]['weight'] = original_weight * (1.0 + fluctuation_ratio)

    return G, state


def evaluate_system_FoM(x, param_names, base_state):
    s = base_state.copy()
    for i, name in enumerate(param_names):
        s[name] = float(x[i])

    C_voltage_sys = 400.0
    C_solar_irradiation = 800.0

    try:
        s['die_area'] = s['cell_width'] * s['cell_length']
        s['cell_volume'] = s['cell_width'] * s['cell_length'] * s['cell_height']
        s['pack_volume'] = s['cell_volume'] * (1.0 + s['pack_volume_margin']) * 1.2
        s['active_material_vol'] = s['cell_volume'] * s['packing_density'] * (
                    1.0 - s['binder_ratio'] - s['conductive_additive_ratio'])
        s['anode_active_area'] = s['active_material_vol'] * s['anode_thickness'] / (s['particle_size_anode'] + 1e-6)
        s['cathode_active_area'] = s['active_material_vol'] * s['cathode_thickness'] / (
                    s['particle_size_cathode'] + 1e-6)
        s['fin_surface_area'] = 2.0 * s['fin_height'] * s['fin_thickness'] / (s['fin_pitch'] + 1e-6) * s['die_area']
        s['thermal_pad_area'] = s['die_area'] * 1.05
        s['cross_section_flow'] = s['cooling_channel_width'] * s['cooling_channel_depth'] * 10.0
        s['hydraulic_diameter'] = 2.0 * s['cooling_channel_width'] * s['cooling_channel_depth'] / (
                    s['cooling_channel_width'] + s['cooling_channel_depth'] + 1e-6)

        s['coolant_velocity'] = s['coolant_flow'] / (s['cross_section_flow'] + 1e-6)
        s['reynolds_number'] = s['coolant_density_ref'] * s['coolant_velocity'] * s['hydraulic_diameter'] / (
                    s['coolant_viscosity_ref'] + 1e-6)
        s['nusselt_number'] = 0.023 * max(0.0, s['reynolds_number']) ** 0.8 * 4.0 ** 0.4
        s['convective_coeff'] = s['nusselt_number'] * 0.6 / (s['hydraulic_diameter'] + 1e-6) + s['pump_speed'] * 0.01
        s['pressure_drop'] = s['pipe_roughness'] * (s['coolant_velocity'] ** 2) * 1000.0 / (
                    s['hydraulic_diameter'] + 1e-6)
        s['pump_power'] = s['pressure_drop'] * s['coolant_flow'] / 0.8 + 0.5 * s['pump_speed'] ** 3 * 1e-9
        s['fan_power'] = 0.5 * max(0.0, s['fan_speed']) ** 2.5 * s['fan_blade_radius'] ** 2 * 1e-6
        s['thermal_resistance_sink'] = 1.0 / (s['convective_coeff'] * s['fin_surface_area'] + 1e-6)
        s['thermal_resistance_tim'] = s['tim_thickness'] / (
                    s['tim_thermal_conductivity'] * s['thermal_pad_area'] + 1e-6)
        s['total_thermal_resistance'] = s['thermal_resistance_sink'] + s['thermal_resistance_tim'] + s[
            'base_plate_thickness'] / (200.0 * s['die_area'] + 1e-6)

        s['wire_resistance'] = 1.68e-8 * 2.5 / (math.pi * s['wire_radius'] ** 2 + 1e-9)
        s['busbar_resistance'] = 1.68e-8 * s['busbar_length'] / (s['busbar_cross_section'] + 1e-6)
        s['tab_resistance'] = 2.82e-8 * s['tab_thickness'] / (s['tab_width'] * s['jellyroll_radius'] + 1e-6)
        s['electrolyte_conductivity'] = 1.5 * s['electrolyte_concentration'] * math.exp(
            -2500.0 / (8.314 * s['chiller_temp_setpoint'] + 1e-6))
        s['separator_resistance'] = s['separator_thickness'] / (
                    s['electrolyte_conductivity'] * max(0.0, s['separator_porosity']) ** 1.5 + 1e-9)
        s['anode_resistance'] = s['anode_thickness'] / (
                    1e4 * s['anode_active_area'] * max(0.0, s['calendering_pressure']) ** 0.5 + 1e-9)
        s['cathode_resistance'] = s['cathode_thickness'] / (
                    5e3 * s['cathode_active_area'] * max(0.0, s['calendering_pressure']) ** 0.4 + 1e-9)
        s['cell_internal_resistance'] = s['separator_resistance'] + s['anode_resistance'] + s['cathode_resistance'] + s[
            'tab_resistance'] + s['electrode_gap'] / (s['electrolyte_conductivity'] + 1e-9)
        s['inverter_resistance'] = s['mosfet_rdson'] * 6.0 + 0.001
        s['total_electrical_resistance'] = s['wire_resistance'] + s['busbar_resistance'] + s[
            'cell_internal_resistance'] + s['inverter_resistance']

        s['operating_current'] = s['current_limit_continuous'] * s['derating_factor']
        s['joule_heating_wire'] = s['operating_current'] ** 2 * s['wire_resistance']
        s['busbar_loss'] = s['operating_current'] ** 2 * s['busbar_resistance']
        s['cell_joule_heating'] = s['operating_current'] ** 2 * s['cell_internal_resistance']
        s['switching_loss'] = 0.5 * C_voltage_sys * s['operating_current'] * s['inverter_switching_freq'] * s[
            'gate_resistance'] * 1e-6
        s['core_loss'] = 1.2 * s['inductor_volume'] * max(0.0, s['inverter_switching_freq'] / 1000.0) ** 1.5 * s[
            'core_cross_section'] ** 2
        s['total_heat_generation'] = s['joule_heating_wire'] + s['busbar_loss'] + s['cell_joule_heating'] + s[
            'switching_loss'] + s['core_loss']
        s['die_temp'] = s['chiller_temp_setpoint'] + s['total_heat_generation'] * s['total_thermal_resistance']
        s['radiation_heat_loss'] = 5.67e-8 * s['casing_emissivity'] * s['housing_surface_area'] * (
                    s['die_temp'] ** 4 - 298.15 ** 4)
        s['housing_temp'] = 298.15 + (s['die_temp'] - 298.15) * math.exp(max(-100.0, -s['housing_thickness'] / 0.05)) - \
                            s['radiation_heat_loss'] / 100.0

        if s['die_temp'] > 298.15:
            s['leakage_current'] = s['operating_current'] * 0.01 * math.exp(min((s['die_temp'] - 298.15) / 25.0, 100.0))
        else:
            s['leakage_current'] = s['operating_current'] * 0.01

        s['cell_capacity'] = 3600.0 * (
                    s['anode_thickness'] * s['anode_active_area'] + s['cathode_thickness'] * s['cathode_active_area']) * \
                             s['packing_density'] * s['soc_target']
        s['cell_voltage_drop'] = s['operating_current'] * s['total_electrical_resistance']
        s['operating_voltage'] = max(s['voltage_min'], min(s['voltage_max'], C_voltage_sys - s['cell_voltage_drop']))
        s['actual_power_out'] = s['operating_voltage'] * s['operating_current']
        s['cell_energy'] = s['cell_capacity'] * s['operating_voltage']

        s['sei_layer_growth'] = max(0.0, s['formation_current']) ** 0.5 * math.exp(
            -1000.0 / (8.314 * s['die_temp'] + 1e-6)) * max(0.0, s['aging_time']) ** 0.2
        s['capacity_fade_rate'] = s['sei_layer_growth'] * max(1e-6,
                                                              s['pre_lithiation_degree'] + 0.01) ** -0.2 * math.log(
            1.0 + max(0.0, s['aging_temperature']))
        s['cycle_life'] = 800.0 / (s['capacity_fade_rate'] + 1e-6) * (1.1 - s['dod_max'])
        s['efficiency_power'] = s['actual_power_out'] / (
                    s['actual_power_out'] + s['total_heat_generation'] + s['pump_power'] + s['fan_power'] + s[
                'compressor_power'] + 1e-6)

        s['system_weight_cells'] = s['pack_volume'] * s['packing_density'] * 2500.0
        s['system_weight_housing'] = s['housing_surface_area'] * s['housing_thickness'] * 2700.0 + s[
            'weld_seam_length'] * 0.1
        s['system_weight_coolant'] = s['pack_volume'] * s['coolant_density_ref'] * 0.1
        s['system_weight'] = s['system_weight_cells'] + s['system_weight_housing'] + s['system_weight_coolant'] + s[
            'busbar_length'] * s['busbar_cross_section'] * 8900.0
        s['specific_energy'] = s['cell_energy'] / (s['system_weight'] + 1e-6)
        s['specific_power'] = s['actual_power_out'] / (s['system_weight'] + 1e-6)
        s['energy_density'] = s['cell_energy'] / (s['pack_volume'] + 1e-6)
        s['power_density'] = s['actual_power_out'] / (s['pack_volume'] + 1e-6)
        s['stress_thermal'] = 12e-6 * 70e9 * max(1.0, s['die_temp'] - 298.15) * math.log(
            1.0 + max(0.0, s['base_plate_thickness']) / 0.01)
        s['volumetric_expansion'] = 0.05 * s['pre_lithiation_degree'] * math.log(
            1.0 + max(0.0, s['cell_capacity']) / 1000.0) + s['thermal_pad_area'] * 0.001

        s['mechanical_stress_expansion'] = s['volumetric_expansion'] * 50e9 * s['housing_thickness'] / (
                    max(0.0, s['die_area']) ** 0.5 + 1e-6)
        s['vibration_tolerance'] = 100.0 * s['housing_thickness'] ** 2 * s['vibration_mount_stiffness'] / (
                    s['housing_surface_area'] * max(0.0, s['system_weight']) ** 0.5 * s[
                'vibration_mount_damping'] + 1e-6)
        s['fatigue_life'] = 1e12 * max(1e-6,
                                       s['stress_thermal'] + s['mechanical_stress_expansion'] + 1e6) ** -3 * math.exp(
            min(298.15 / (s['die_temp'] + 1e-6), 100.0))
        s['short_circuit_risk'] = 1e-3 * math.exp(min((s['die_temp'] - 298.15) / 50.0, 100.0)) * s[
            'separator_porosity'] / (s['separator_thickness'] * 1e6 + 1e-6)
        s['leakage_risk'] = 1e-4 * 101325.0 * s['housing_surface_area'] / (
                    s['housing_thickness'] * s['gasket_compression'] + 1e-6)
        s['emi_shielding'] = 50.0 * math.log(1.0 + max(0.0, s['housing_thickness']) * 1000.0) * max(0.0, s[
            'conductive_additive_ratio']) ** 0.5
        s['manufacturing_defect_rate'] = max(0.0, 1.0 - math.exp(
            max(-100.0, -s['assembly_pressure'] / 1e6 - s['coating_speed'] / 10.0)) * s['inspection_rate'])

        s['material_cost'] = 50.0 * s['die_area'] + 200.0 * (s['anode_thickness'] + s['cathode_thickness']) + 15.0 * s[
            'separator_thickness'] / (s['separator_porosity'] + 1e-6) + s['system_weight'] * 2.0
        s['manufacturing_cost'] = 1e8 / (s['calendering_pressure'] + 1.0) + 500.0 * max(0.0,
                                                                                        s['formation_current']) ** 0.2 + \
                                  s['welding_power'] * 0.01 + s['drying_temperature'] * s['drying_time'] * 0.1
        s['cooling_cost'] = 5.0 * max(0.0, s['pump_power']) ** 0.6 + 2.0 * max(0.0, s['fan_power']) ** 0.7 + 10.0 * max(
            0.0, s['compressor_power']) ** 0.8

        s['recyclability_index'] = max(0.0, 100.0 - 10.0 * s['binder_ratio'] / (
                    s['binder_ratio'] + s['conductive_additive_ratio'] + 1e-6) - s[
                                           'potting_compound_thickness'] * 100.0)
        s['carbon_footprint'] = 0.5 * s['material_cost'] + 0.2 * s['manufacturing_cost'] + 0.1 * s['cooling_cost']
        s['startup_time'] = 1500.0 * s['cell_internal_resistance'] * s['cell_capacity'] / (
                    s['actual_power_out'] + 1e-6) + s['sensor_accuracy'] * 10.0
        s['dynamic_response'] = s['power_density'] * s['vibration_tolerance'] / (s['startup_time'] + 1e-6) * max(0.0, s[
            'bms_sampling_rate']) ** 0.5
        s['aging_acceleration'] = math.exp(min((s['die_temp'] - 298.15) / 15.0, 100.0)) * (
                    s['cell_voltage_drop'] / C_voltage_sys) ** 2
        s['thermal_gradient'] = max(0.1, (s['die_temp'] - s['housing_temp'])) / s['base_plate_thickness']
        s['cooling_efficiency'] = (s['convective_coeff'] * s['fin_surface_area']) / (
                    s['pump_power'] + s['fan_power'] + s['compressor_power'] + 1e-6)

        s['solar_heat_gain'] = C_solar_irradiation * s['housing_surface_area'] * 0.8
        s['ambient_heat_transfer'] = (s['housing_temp'] - s['ambient_temp']) * s['housing_surface_area'] * 10.0
        s['total_housing_heat_flux'] = s['solar_heat_gain'] - s['ambient_heat_transfer'] - s['radiation_heat_loss']

        s['Sys_Risk'] = s['short_circuit_risk'] * 100.0 + s['leakage_risk'] * 50.0 + s['aging_acceleration'] * 200.0 + \
                        s['mechanical_stress_expansion'] / 1e7 + s['stress_thermal'] / 1e8 + 1000.0 / (
                                    s['fatigue_life'] + 1e-6) + 10.0 / (s['emi_shielding'] + 1e-6) + s[
                            'die_temp'] / 100.0 + s['housing_temp'] / 100.0 + s['cell_voltage_drop'] + s[
                            'joule_heating_wire'] / 100.0 + s['leakage_current'] * 10.0 + s[
                            'thermal_gradient'] / 1000.0 + s['manufacturing_defect_rate'] * 500.0

        s['Total_Cost'] = s['cooling_cost'] + s['material_cost'] + s['manufacturing_cost'] + s[
            'carbon_footprint'] * 15.0 + s['Sys_Risk'] * 50.0 + 1e5 / (s['cycle_life'] + 1e-6) + s[
                              'system_weight'] * 5.0 + s['pump_power'] + s['fan_power'] + s['compressor_power'] / 10.0 + \
                          s['total_electrical_resistance'] * 1e6

        reg_term_vars = [k for k in base_state.keys() if type(base_state[k]) in [float, int] and 'sys' not in k.lower()]
        reg_term = sum([math.log(max(1e-9, abs(s[v]))) for v in reg_term_vars])

        numerator = (max(0.0, s['specific_energy']) ** 0.5 * max(0.0, s['specific_power']) ** 0.5 * max(0.0, s[
            'cycle_life']) ** 0.8 * s['efficiency_power'] * s['dynamic_response'] * s['recyclability_index'] * max(0.0,
                                                                                                                   s[
                                                                                                                       'energy_density']) ** 0.2 * max(
            0.0, s['power_density']) ** 0.2 * max(0.0, s['cooling_efficiency']) ** 0.1)
        denominator = (max(0.0, s['Total_Cost']) ** 0.5 * max(0.0, s['Sys_Risk']) ** 1.2 * max(0.0, s[
            'startup_time']) ** 0.3 * max(0.0, s['cell_internal_resistance']) ** 0.2 * max(0.0, s[
            'total_thermal_resistance']) ** 0.1 + 1e-6)

        s['Global_FoM'] = (numerator / denominator) * math.exp(max(-100.0, -0.01 * reg_term))

        if math.isnan(s['Global_FoM']) or math.isinf(s['Global_FoM']):
            return 1e-6

        return s['Global_FoM'] * 1e11

    except Exception as e:

        return -1.0


if __name__ == "__main__":
    G, state = build_physics_graph()
    edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    edges.sort(key=lambda x: x[2], reverse=True)