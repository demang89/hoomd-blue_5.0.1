# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement BoxResize."""

from hoomd.operation import Updater
from hoomd.data.parameterdicts import ParameterDict
from hoomd.variant import Variant, Constant
from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All
from hoomd.trigger import Periodic


class BoxShear(Updater):

    def __init__(self, trigger, vinf, deltaT, vscale, flip, filter=All()):
        params = ParameterDict(vinf=Variant,
                               deltaT = float,
                               vscale = bool,
                               flip=bool,
                               filter=ParticleFilter)
        params['vinf'] = vinf
        params['trigger'] = trigger
        params['deltaT'] = deltaT
        params['vscale'] = vscale
        params['flip'] = flip
        params['filter'] = filter
        self._param_dict.update(params)
        super().__init__(trigger)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self.filter)
        self._cpp_obj = _hoomd.BoxShearUpdater(
            self._simulation.state._cpp_sys_def, self.trigger, self.vinf, self.deltaT, self.vscale, self.flip, group)

    @staticmethod
    def update(state, deltaT, flip, filter=All()):
        group = state._get_group(filter)
        updater = _hoomd.BoxShearUpdater(
             state._cpp_sys_def, Periodic(1), Constant(0), deltaT, vscale, flip, group)
