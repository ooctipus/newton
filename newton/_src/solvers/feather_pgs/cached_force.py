# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Author current-force arithmetic for the cached predictor mappings.

Operands are native expressions supplied by the owner, never extracted from
another kernel. Owners retain reductions, drive production, held operator
actions, validation and publication transactions. These operations allocate
no storage and introduce no launch or synchronization boundary.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ForceOperands:
    """Bind current scalar force inputs separately from predictor velocity.

    All array operands are native pointer expressions. q and the spring reference use a
    coordinate index; all other arrays use physical DOF indices, never factor indices.
    """

    q: str
    state_qd: str
    predictor_qd: str
    stiffness: str
    reference: str
    damping: str
    dt: str

    def passive(self, q_index, dof):
        """Evaluate the scalar spring/damper law using current state velocity."""
        return (
            f"{self.stiffness}[{dof}]*({self.reference}[{q_index}]-{self.q}[{q_index}])"
            f"-{self.damping}[{dof}]*{self.state_qd}[{dof}]"
        )

    def velocity(self, dof, acceleration):
        """Increment the declared predictor velocity with the held action."""
        return f"{self.predictor_qd}[{dof}]+{self.dt}*{acceleration}"

    def transport(self, first_dof, output):
        """Add free-root transport (:func:`.spatial_laws.free_root_transport`); the owner guards prescribed roots.

        Output is a component expression using ``k`` and may address scratch
        for deferred publication or the owner's existing output bank.
        """
        qd, d = self.predictor_qd, first_dof
        return f"""{{
            const wp::vec3 v({qd}[{d}],{qd}[{d}+1],{qd}[{d}+2]);
            const wp::vec3 w({qd}[{d}+3],{qd}[{d}+4],{qd}[{d}+5]);
            const wp::vec3 c=Spatial::free_root_transport(v,w);
            for(int k=0;k<3;++k){output}+=c[k]*{self.dt};
        }}\n"""
