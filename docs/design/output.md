# Output registry

First template for discussion, November 26, 2026. Design only: there is no code for this note. Nothing here is settled. Part of the set introduced in [remaining_areas.md](remaining_areas.md).

**Scope.** How the variables a FEBio input asks for, in its plot file and its log file, are produced from a State. The writers themselves (`.xplt`, VTKHDF) are in pull request 5 and are not changed.

## 1. Summary

- **The writers take any variable; only three are produced.** Displacement, stress and reaction force.
- **The two application inputs ask for nine plot variables between them**, and the mitral input also asks for log data.
- **Log data matters most for verification.** FEBio's log has 12 digits; its plot file is single precision.
- **Proposal: a registry from FEBio's variable names to a kernel on the State**, with data moved to the host at output steps only.

## 2. From a State to a file

```mermaid
flowchart LR
    req["Output section of the input<br/>plot variables, log data"] --> reg["Registry<br/>FEBio name to producer"]
    state(["State at an output time"]) --> prod["Producer<br/>a kernel on the device,<br/>owned by a domain, a contact term<br/>or the system"]
    reg --> prod
    prod --> host["Copy to the host<br/>at output steps only"]
    host --> xplt[".xplt"]
    host --> vtk["VTKHDF"]
    host --> log["Log data<br/>for comparison with FEBio"]
    reg --> cov["Coverage report<br/>names not registered"]
```

## 3. What the inputs ask for

| Plot variable | Mitral | TPVR (file 3599) | FEBio class | Produced today | Would come from |
| --- | --- | --- | --- | --- | --- |
| `displacement` | yes | yes | `FEPlotNodeDisplacement` | yes | The State |
| `stress` | yes | yes | `FEPlotElementStress` | yes | Domain point data |
| `reaction forces` | | yes | `FEPlotNodeReactionForces` | yes | The system |
| `relative volume` | yes | yes | `FEPlotRelativeVolume` | no | Domain point data (J is already computed) |
| `Lagrange strain` | yes | yes | `FEPlotLagrangeStrain` | no | Domain point data (from F) |
| `shell thickness` | yes | yes | `FEPlotShellThickness` | no | Shell domain |
| `contact gap` | yes | yes | `FEPlotContactGap` | no | Contact term |
| `contact pressure` | yes | yes | `FEPlotContactPressure` | no | Contact term |
| `discrete element force` | yes | | `FEPlotDiscreteElementForce` | no | Spring domain |

The mitral input also has log data: `Ux;Uy;Uz;s1;E1` per element for seven element sets, `contact gap` per face, and `contact area` for the contact surface.

For the contact potential, FEBio reports as the gap the smallest pair distance found at an integration point, as the pressure the magnitude of the traction there, and as the area the sum over points whose traction is not zero (`FEContactPotential.cpp`). The contact draft reports contact area and contact fraction today.

## 4. Proposed

| Topic | Proposal | Why |
| --- | --- | --- |
| Registry | A table from FEBio's name to a producer, with the data class the writer needs (per node, per element, per face) | The plot file writer of pull request 5 already takes any variable with its class |
| Producers | Each term offers the variables it owns through one call, `plot(state, name)` | Stress belongs to the domain that computed it; the gap belongs to the contact term |
| Element values from point values | FEBio's rule for each variable, as the run path already follows for stress | The comparison is with FEBio's numbers |
| Log data | The same producers, with FEBio's log names (`Ux`, `s1`, `E1`) mapped to components and principal values | Log data is what the tests compare |
| Timing | Producers run on the device at output steps. One copy to the host per variable per output step | Follows the run path's rule for what crosses to the host |
| Memory | Results are streamed to the writer per output step | The run path keeps all results in memory, which the 200 states of an application would make large |
| Unknown names | Listed by the coverage report, and the run proceeds without them | An input should not be refused for a plot variable |

**Alternative considered.** Compute every derived variable on the host from displacement and stress after the run. It keeps the device code smaller. It cannot give contact or spring variables, and it repeats kinematics the domains already have.

## 5. Verification

Each producer is compared with FEBio's log data on the small problem that exercises its owner. The stored references log nodal displacements, velocities and reactions and element stress components. References for `relative volume`, `Lagrange strain`, `contact gap` and `discrete element force` would need to be regenerated with those names added to the log.

## 6. Reserved

- `beam stress` and other beam variables.
- Variables for sticky and sliding contact.
- Output to OpenUSD, which the model note mentions.

## 7. Depends on

The results store and writers of pull request 5, and the `Term` contract of the run path.
