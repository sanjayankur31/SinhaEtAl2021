#!/usr/bin/env python3
"""
Single neuron simulations including structural plasticity.
Pre-print: https://www.biorxiv.org/content/early/2019/10/21/810846

File: Sinha2020-single.py

Copyright 2020 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import nest
import numpy


class Sinha2020single():

    """Single neuron simulation code for Sinha2020"""

    def __init__(self):
        """Initialise everything """
        self.dt = 0.1

        # Growth curve parameters
        self.nu = 0.001

        # Initial values
        # Different values for butz neuron, and our neuron
        self.psi = 10.
        self.eps_den_e_new = self.psi
        self.eps_den_i_new = self.psi * 1.75
        self.eta_den_e_new = self.psi * 0.25
        self.eta_den_i_new = self.psi

        # identical curves
        self.eps_den_butz = self.psi
        self.eta_den_butz = self.psi * 0.25

        # synapses
        self.weightE = 0.5
        self.weightI = 3.

        self.base_current = 180.
        self.current_delta = 30.

        self.neuronDict = {'V_m': -60.,
                           't_ref': 5.0, 'V_reset': -60.,
                           'V_th': -50., 'C_m': 200.,
                           'E_L': -60., 'g_L': 10.,
                           'E_ex': 0., 'E_in': -80.,
                           'tau_syn_ex': 5., 'tau_syn_in': 10.,
                           'beta_Ca': 0.010, 'tau_Ca': 50000.,
                           'I_e': self.base_current
                           }

        # record every 5 seconds
        self.steps = 5.

    def __create_neurons(self, sim_time=500.):
        """Setup simulation
        :returns: nothing

        """
        # Our neuron only needs dendritic elements
        new_growth_curve_dendritic_e_new = {
            'growth_curve': "gaussian",
            'growth_rate': 0.,
            'tau_vacant': 0.000000000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_e_new,
            'eps': self.eps_den_e_new
        }
        new_growth_curve_dendritic_i_new = {
            'growth_curve': "gaussian",
            'growth_rate': 0.,
            'tau_vacant': 0.000000000000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_i_new,
            'eps': self.eps_den_i_new
        }

        structural_p_elements_E_new = {
            'Den_ex': new_growth_curve_dendritic_e_new,
            'Den_in': new_growth_curve_dendritic_i_new,
        }

        self.neuron_new = nest.Create(
            "tif_neuronE", 1,
            {'synaptic_elements': structural_p_elements_E_new})

        new_growth_curve_dendritic_e_butz = {
            'growth_curve': "gaussian",
            'growth_rate': 0.,
            'tau_vacant': 0.00000000000000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_butz,
            'eps': self.eps_den_butz
        }
        new_growth_curve_dendritic_i_butz = {
            'growth_curve': "gaussian",
            'growth_rate': 0.,
            'tau_vacant': 0.000000000000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_butz,
            'eps': self.eps_den_butz
        }

        structural_p_elements_E_butz = {
            'Den_ex': new_growth_curve_dendritic_e_butz,
            'Den_in': new_growth_curve_dendritic_i_butz,
        }

        self.neuron_butz = nest.Create(
            "tif_neuronE", 1,
            {'synaptic_elements': structural_p_elements_E_butz})

    def __get_optimal_activity(self, sim_time=500.):
        """Get the optimal activity with default external current."""
        # Run for some time, see what activity level the neuron gets to
        self.__run_sim(sim_time, False, self.steps)
        ca = nest.GetStatus(self.neuron_new, ['Ca'])[0][0]

        self.psi = ca
        self.eps_den_e_new = self.psi
        self.eps_den_i_new = self.psi * 1.75
        self.eta_den_e_new = self.psi * 0.25
        self.eta_den_i_new = self.psi

        # identical curves
        self.eps_den_butz = self.psi
        self.eta_den_butz = self.psi * 0.25

        with open("Parameters.txt", 'w') as f:
            print("Psi: {}".format(ca), file=f)
            print("eps_den_e_new: {}".format(self.eps_den_e_new), file=f)
            print("eps_den_i_new: {}".format(self.eps_den_i_new), file=f)
            print("eps_den_butz: {}".format(self.eps_den_butz), file=f)
            print("eps_den_butz: {}".format(self.eps_den_butz), file=f)

    def __grow_initial_elements(self, sim_time=500.):
        """Grow intial elements."""
        # Growing some elements with activity less than the psi value
        nest.SetStatus(self.neuron_new, {'I_e': self.base_current -
                                         self.current_delta})
        nest.SetStatus(self.neuron_butz, {'I_e': self.base_current -
                                          self.current_delta})
        # set the nu to 4 times here, because in general, the indegree of
        # neurons is 4 times more for excitatory elements than inhibitory
        # elements
        new_growth_curve_dendritic_e_new = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu * 4.,
            'tau_vacant': 0.000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_e_new,
            'eps': self.eps_den_e_new
        }
        # use same parameters here to get both denE and denI elements to form
        # as required
        new_growth_curve_dendritic_i_new = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,
            'tau_vacant': 0.0000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_e_new,
            'eps': self.eps_den_e_new
        }
        structural_p_elements_E_new = {
            'Den_ex': new_growth_curve_dendritic_e_new,
            'Den_in': new_growth_curve_dendritic_i_new,
        }
        nest.SetStatus(self.neuron_new, 'synaptic_elements_param',
                       structural_p_elements_E_new)

        new_growth_curve_dendritic_e_butz = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu * 4.,
            'tau_vacant': 0.000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_butz,
            'eps': self.eps_den_butz
        }
        new_growth_curve_dendritic_i_butz = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,
            'tau_vacant': 0.000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_butz,
            'eps': self.eps_den_butz
        }
        structural_p_elements_E_butz = {
            'Den_ex': new_growth_curve_dendritic_e_butz,
            'Den_in': new_growth_curve_dendritic_i_butz,
        }
        nest.SetStatus(self.neuron_butz, 'synaptic_elements_param',
                       structural_p_elements_E_butz)
        self.__run_sim(sim_time, False, self.steps)

    def __return_activity_to_hom(self, sim_time=500.):
        """Run for a bit at the fixed point to check that elements are stable.

        :sim_time: simulation time
        :returns: nothing

        """
        new_growth_curve_dendritic_e_new = {
            'growth_curve': "gaussian",
            'growth_rate': 0.,
            'tau_vacant': 0.00000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_e_new,
            'eps': self.eps_den_e_new
        }
        new_growth_curve_dendritic_i_new = {
            'growth_curve': "gaussian",
            'growth_rate': 0.,
            'tau_vacant': 0.000000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_i_new,
            'eps': self.eps_den_i_new
        }
        structural_p_elements_E_new = {
            'Den_ex': new_growth_curve_dendritic_e_new,
            'Den_in': new_growth_curve_dendritic_i_new,
        }
        nest.SetStatus(self.neuron_new, 'synaptic_elements_param',
                       structural_p_elements_E_new)

        new_growth_curve_dendritic_e_butz = {
            'growth_curve': "gaussian",
            'growth_rate': 0.,
            'tau_vacant': 0.0000000000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_butz,
            'eps': self.eps_den_butz
        }
        new_growth_curve_dendritic_i_butz = {
            'growth_curve': "gaussian",
            'growth_rate': 0.,
            'tau_vacant': 0.000000000000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_butz,
            'eps': self.eps_den_butz
        }
        structural_p_elements_E_butz = {
            'Den_ex': new_growth_curve_dendritic_e_butz,
            'Den_in': new_growth_curve_dendritic_i_butz,
        }
        nest.SetStatus(self.neuron_butz, 'synaptic_elements_param',
                       structural_p_elements_E_butz)

        nest.SetStatus(self.neuron_new, {'I_e': self.base_current})
        nest.SetStatus(self.neuron_butz, {'I_e': self.base_current})
        self.__run_sim(sim_time, True, self.steps)

    def __prepare_hypotheses(self):
        """Prepare real growth curves."""
        # Neurons are now ready
        # remove background input
        nest.SetStatus(self.neuron_new, {'I_e': 0.})
        nest.SetStatus(self.neuron_butz, {'I_e': 0.})

        new_growth_curve_dendritic_e_new = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,
            'tau_vacant': 0.000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_e_new,
            'eps': self.eps_den_e_new
        }
        new_growth_curve_dendritic_i_new = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,
            'tau_vacant': 0.000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_i_new,
            'eps': self.eps_den_i_new
        }
        structural_p_elements_E_new = {
            'Den_ex': new_growth_curve_dendritic_e_new,
            'Den_in': new_growth_curve_dendritic_i_new,
        }
        nest.SetStatus(self.neuron_new, 'synaptic_elements_param',
                       structural_p_elements_E_new)

        new_growth_curve_dendritic_e_butz = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,
            'tau_vacant': 0.000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_butz,
            'eps': self.eps_den_butz
        }
        new_growth_curve_dendritic_i_butz = {
            'growth_curve': "gaussian",
            'growth_rate': self.nu,
            'tau_vacant': 0.000000000000000000000000000000000000001,
            'continuous': False,
            'eta': self.eta_den_butz,
            'eps': self.eps_den_butz
        }
        structural_p_elements_E_butz = {
            'Den_ex': new_growth_curve_dendritic_e_butz,
            'Den_in': new_growth_curve_dendritic_i_butz,
        }
        nest.SetStatus(self.neuron_butz, 'synaptic_elements_param',
                       structural_p_elements_E_butz)

    def __run_sim(self, sim_time, str_p, step=1.):
        "Run the sim in bits."
        update_steps = numpy.arange(0, sim_time, step)
        for i in update_steps:
            nest.Simulate(step * 1000.)

            ca = nest.GetStatus(self.neuron_new, ['Ca'])[0][0]
            synelms = nest.GetStatus(self.neuron_new,
                                     ['synaptic_elements'])[0][0]

            print(
                "{}\t{}\t{}\t{}\t{}".format(
                    nest.GetKernelStatus()['time'],
                    ca,
                    synelms['Den_ex']['z'],
                    synelms['Den_in']['z'],
                    self.weightE * synelms['Den_ex']['z'] -
                    self.weightI * synelms['Den_in']['z']),
                file=self.fh_neuron_new
            )

            ca = nest.GetStatus(self.neuron_butz, ['Ca'])[0][0]
            synelms = nest.GetStatus(self.neuron_butz,
                                     ['synaptic_elements'])[0][0]
            # In the butz model, the conductances were all 1nS
            print(
                "{}\t{}\t{}\t{}\t{}".format(
                    nest.GetKernelStatus()['time'],
                    ca,
                    synelms['Den_ex']['z'],
                    synelms['Den_in']['z'],
                    synelms['Den_ex']['z'] -
                    synelms['Den_in']['z']),
                file=self.fh_neuron_butz
            )

    def __setup(self, sim_time=500.):
        """Setup the simulation

        :sim_time: time for each simulation phase
        :file_prefix: output file prefix
        :returns: nothing

        """
        nest.ResetKernel()
        # http://www.nest-simulator.org/sli/setverbosity/
        nest.set_verbosity('M_INFO')
        nest.SetKernelStatus(
            {
                'resolution': self.dt,
                'local_num_threads': 1,
                'overwrite_files': True
            }
        )
        # Since I've patched NEST, this doesn't actually update connectivity
        # But, it's required to ensure that synaptic elements are connected
        # correctly when I form or delete new connections
        nest.EnableStructuralPlasticity()
        nest.CopyModel("iaf_cond_exp", "tif_neuronE")
        nest.SetDefaults('tif_neuronE', self.neuronDict)

        # do the bits
        self.__create_neurons()
        self.__get_optimal_activity()
        self.__grow_initial_elements()
        self.__return_activity_to_hom()
        self.__prepare_hypotheses()

    def __close_files(self):
        """Close files."""
        self.fh_neuron_butz.close()
        self.fh_neuron_new.close()

    def modulated_ext_input_run(self, sim_time=500., file_prefix=""):
        """Excite neuron to  its activity.

        Here we excite the neuron to make its activity more than optimal.

        :sim_time: time to run, in seconds
        :returns: nothing

        """
        # set up output file handles
        self.fh_neuron_butz = open("{}butz.csv".format(file_prefix), 'w')
        self.fh_neuron_new = open("{}new.csv".format(file_prefix), 'w')

        self.__setup()

        modulatory_input_dict = {
            'amplitude': self.current_delta,
            'offset': self.base_current,
            'frequency': 1/400., 'phase': 0.
        }
        modulatory_input = nest.Create('ac_generator', 1,
                                       params=modulatory_input_dict)

        nest.Connect(modulatory_input, self.neuron_new,
                     syn_spec={'model': 'static_synapse',
                               'delay': 0.1})
        nest.Connect(modulatory_input, self.neuron_butz,
                     syn_spec={'model': 'static_synapse',
                               'delay': 0.1})
        self.__run_sim(sim_time * 4., True, self.steps)

        self.__close_files()


if __name__ == "__main__":
    sim = Sinha2020single()
    sim.modulated_ext_input_run()
