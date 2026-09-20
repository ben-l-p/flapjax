# Tutorials

The below tutorials showcase the capabilities of the framework, whilst giving clean working examples on how to use the
package. Each tutorial is a Jupyter notebook, and can be run interactively in a local environment. Some examples will
write ``.vtu`` and ``.pvd`` files to disk, which can be visualised in ParaView.

- Structure
    - [Geradin beam static deformation](geradin_beam_static.ipynb) - Clamped cantilever beam subject to a tip load,
      causing large deflections. The tutorial demonstrates how to set up a simple structural problem, and use the beam
      ``static_solve()`` routine to solve for the deformation when subject to follower and dead external forces.
    - [Geradin beam adjoint gradients](geradin_beam_adjoint.ipynb) - Follow-on to the static deformation tutorial that
      uses the beam ``static_adjoint()`` routine to compute the gradient of the tip vertical displacement with respect
      to the tip load and the beam bending stiffness, then verifies each against finite differences.
        - [Flying spaghetti free dynamics](flying_spaghetti.ipynb) - Free-flying beam subject to time-dependent external
          forces, making use of the beam ``dynamic_solve()`` routine to find the time-doman response.
    - [Flying spaghetti mass optimisation](flying_spaghetti_optimise.ipynb) - Follow-on to the free dynamics tutorial
      that uses the beam ``dynamic_adjoint()`` routine to drive an SLSQP outer loop, redistributing per-element mass to
      minimise the final-time strain energy of the beam subject to a total-mass constraint.
    - [Flexible double pendulum](double_pendulum.ipynb) - Two very flexible beam segments connected by a hinge joint,
      with the root mounted to a hinge. Demonstrates multibody dynamics with ``GroundedHinge`` and ``MultibodyHinge``
      constraints using the ``dynamic_solve()`` routine.
- Aeroelastic
    - [Simple HALE gust response](simple_hale_gust.ipynb) - Free-flying high aspect ratio aircraft configuration subject
      to a one-minus-cosine gust. This first uses the aeroelastic``trim()``
      routine to find the thrust and elevator deflection that satisfy trim conditions, before using the aeroelastic
      ``dynamic_solve()`` routine to find the time-domain response of the aircraft. This tutorial demonstrates using
      batching to efficiently parallelise for multiple gust cases at once.s
    - [Simple HALE hinged wingtip gust response](simple_hale_hinged_gust.ipynb) - Variant of the above with
      free-hinging wingtips, using ``MultibodyHinge`` constraints. Demonstrates trimming a multibody aircraft, where
      the trim solver also finds the equilibrium hinge angles, before flying it through the gust as a free-flying
      body.
        - [Cantilever wing adjoint](cantilever_wing_adjoint.ipynb) — Cantilever wing subject to a one-minus-cosine
          gust, using the coupled ``static_adjoint()`` and ``dynamic_adjoint()`` routines to compute the gradient of
          the peak wing root bending moment with respect to structural and aerodynamic design variables.
        - [Patil wing open-loop control](patil_wing_control.ipynb) — Open-loop control of ailerons for a pair of very
          flexible wings mounted on a central hinge, performing a roll manoeuvre. Makes use of the aeroelastic
          ``trim()``
          and
          ``dynamic_solve()`` routines to find the time-domain response.
        - [Patil wing optimal roll manoeuvre](patil_wing_optimal_roll.ipynb) — Follow-on to the open-loop control
          tutorial that instead optimises the control-surface velocity profiles with an SLSQP outer loop, using
          gradients of a target roll angle objective from the coupled dynamic adjoint.
        - Pazy wing (both straight and swept configurations)
            - Static
              deflection [straight](straight_pazy_static_deflection.ipynb), [swept](swept_pazy_static_deflection.ipynb) —
              Grid of cases with varying root angles of attack and velocity, running the ``static_solve()`` routine in
              parallel. Evolution of the tip deflection is plotted.
            - Deformed mode
              frequencies [straight](straight_pazy_deformed_modes.ipynb), [swept](swept_pazy_deformed_modes.ipynb) —
              Evolution of the first five natural frequencies with tip displacement, controlled via freestream velocity
              and
              making use of the ``static_solve()``
              and structural ``modal()`` routines.
            - Flutter analysis [straight](straight_pazy_flutter.ipynb), [swept](swept_pazy_flutter.ipynb) — Compute the
              stability of the wing across a grid of angles of attack and freestream velocities using the coupled
              ``linearise()`` routine.
            - Time-domain LCO [straight](straight_pazy_lco.ipynb), [swept](swept_pazy_lco.ipynb) — Time-domain
              limit-cycle
              oscillation computation for the wing, plotting the deflection of the beam tip over time, obtained using
              the
              ``static_solve()`` and ``dynamic_solve()`` routines.
