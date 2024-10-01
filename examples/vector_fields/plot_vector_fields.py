"""Example to plot vector fields on an O0 atlas."""

from neurocollage import get_layer_annotation
from neurocollage.mesh_helper import MeshHelper
from neurocollage.planes import get_atlas

if __name__ == "__main__":
    atlas_path = {"atlas": "atlas", "structure": "region_structure.yaml"}
    region = "O0"

    n_vec = 10
    length = 5

    atlas = get_atlas(atlas_path)
    layer_annotation = get_layer_annotation(atlas_path, region=region)

    mesh_helper = MeshHelper(atlas_path, region)
    mesh_helper.layer_annotation = layer_annotation
    vector_fields = mesh_helper.get_vector_field(n_vec=10, length=5)

    mesh_helper.render(data=vector_fields)
    mesh_helper.show()
