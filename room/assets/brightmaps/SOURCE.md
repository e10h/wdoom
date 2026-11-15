Exact emission masks from the Brightmaps Plus PNGs and material definitions
already used by the native renderer in octree-tracer/files/gzdoom_brightmaps.

Regenerate with tools/pack-brightmaps.py and the native
bmplus_textures/brightmaps/textures/doom directory. The native parity test
compares every packed pixel with the native mask loader, including liquids.
