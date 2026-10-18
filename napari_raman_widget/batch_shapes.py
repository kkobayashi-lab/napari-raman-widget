"""Native Shapes layer with a local napari 0.5.6 multidimensional picking fix."""
import napari
import numpy as np
from napari.layers import Shapes


class BatchShapes(Shapes):
    def _get_value(self, position):
        # 0.5.6's ShapeList._visible_shapes chains array comparisons, raising
        # during cursor status updates and picking with multiple hidden axes.
        # Populate its normal cache with an elementwise comparison. ShapeList
        # already invalidates this cache on edits, deletion and slice changes;
        # fetching _data_view here also handles replacement of the layer data.
        # Keep this workaround on this layer and leave other versions untouched.
        if napari.__version__ == "0.5.6":
            view = self._data_view
            if "_visible_shapes" not in view.__dict__:
                key = np.asarray(view.slice_key)
                view.__dict__["_visible_shapes"] = [
                    (index, shape) for index, shape in enumerate(view.shapes)
                    if not key.size or (
                        np.all(shape.slice_key[0] <= key)
                        and np.all(key <= shape.slice_key[1])
                    )
                ]
        return super()._get_value(position)
