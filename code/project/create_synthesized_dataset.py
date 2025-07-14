import SimpleITK as sitk

import disptools.displacements as dsp
import disptools.drawing as drw

# Create an example Jacobian map
# A spherical ROI with a Jacobian of 1.1 (expansion)
jacobian = drw.create_sphere(10, 40, fg_val=1.1, bg_val=1.0)

# Create a binary mask for the ROI
mask = drw.create_sphere(10, 40) > 0

# Generate the displacement
displacement = dsp.displacement(jacobian, mask=mask)

# Check the correctness of the result within the ROI
error = jacobian - dsp.jacobian(displacement)
error = sitk.Mask(error, mask)
